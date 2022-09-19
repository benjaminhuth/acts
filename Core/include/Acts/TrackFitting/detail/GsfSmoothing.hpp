// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/TrackFitting/GsfError.hpp"
#include "Acts/TrackFitting/KalmanFitter.hpp"
#include "Acts/TrackFitting/detail/GsfUtils.hpp"
#include "Acts/Utilities/detail/gaussian_mixture_helpers.hpp"

namespace Acts {

namespace detail {

/// @brief Smoothing function, which takes two ranges of
/// MultiTrajectory-indices and the corresponding projectors.
template <typename fwd_it, typename bwd_it, typename fwd_projector_t = Identity,
          typename bwd_projector_t = Identity>
auto bayesianSmoothing(fwd_it fwdBegin, fwd_it fwdEnd, bwd_it bwdBegin,
                       bwd_it bwdEnd,
                       fwd_projector_t fwdProjector = fwd_projector_t{},
                       bwd_projector_t bwdProjector = bwd_projector_t{}) {
  std::vector<std::tuple<double, BoundVector, std::optional<BoundSymMatrix>>>
      smoothedState;

  using ResType = Result<decltype(smoothedState)>;

  for (auto fwd = fwdBegin; fwd != fwdEnd; ++fwd) {
    const auto &[weight_a, pars_a, cov_a] = fwdProjector(*fwd);
    throw_assert(cov_a, "for now we require a covariance here");

    for (auto bwd = bwdBegin; bwd != bwdEnd; ++bwd) {
      const auto &[weight_b, pars_b, cov_b] = bwdProjector(*bwd);
      throw_assert(cov_b, "for now we require a covariance here");

      const auto summedCov = *cov_a + *cov_b;

      const auto K = *cov_a * summedCov.inverse();
      const auto new_pars = (pars_a + K * (pars_b - pars_a)).eval();
      const auto new_cov = (K * *cov_b).eval();

      const auto diff = pars_a - pars_b;
      const ActsScalar exponent = diff.transpose() * summedCov.inverse() * diff;

      const auto new_weight = std::exp(-0.5 * exponent) * weight_a * weight_b;

      if (new_weight == 0) {
        return ResType(GsfError::SmoothingFailed);
      }

      smoothedState.push_back({new_weight, new_pars, new_cov});
    }
  }

  normalizeWeights(smoothedState, [](auto &tuple) -> decltype(auto) {
    return std::get<double>(tuple);
  });

  throw_assert(weightsAreNormalized(
                   smoothedState,
                   [](const auto &tuple) { return std::get<double>(tuple); }),
               "smoothed state not normalized");

  return ResType(smoothedState);
}

/// @brief Projector type which maps a MultiTrajectory-Index to a tuple of
/// [weight, parameters, covariance]. Therefore, it contains a MultiTrajectory
/// and for now a std::map for the weights
template <StatesType type, typename traj_t>
struct MultiTrajectoryProjector {
  const MultiTrajectory<traj_t> &mt;
  const std::map<MultiTrajectoryTraits::IndexType, double> &weights;

  auto operator()(MultiTrajectoryTraits::IndexType idx) const {
    const auto proxy = mt.getTrackState(idx);
    switch (type) {
      case StatesType::ePredicted:
        return std::make_tuple(weights.at(idx), proxy.predicted(),
                               std::optional{proxy.predictedCovariance()});
      case StatesType::eFiltered:
        return std::make_tuple(weights.at(idx), proxy.filtered(),
                               std::optional{proxy.filteredCovariance()});
      case StatesType::eSmoothed:
        return std::make_tuple(weights.at(idx), proxy.smoothed(),
                               std::optional{proxy.smoothedCovariance()});
    }
  }
};

template <typename D, typename Visitor>
auto visitMultiTrajectoryMultiComponents(
    const MultiTrajectory<D> &traj,
    std::vector<MultiTrajectoryTraits::IndexType> tips, Visitor &&visitor) {
  auto sortUniqueValidateTips = [&]() {
    std::sort(tips.begin(), tips.end());
    tips.erase(std::unique(tips.begin(), tips.end()), tips.end());

    auto invalid_it = std::find(tips.begin(), tips.end(),
                                std::numeric_limits<uint16_t>::max());
    if (invalid_it != tips.end()) {
      tips.erase(invalid_it);
    }
  };

  sortUniqueValidateTips();

  while (!tips.empty()) {
    visitor(tips);

    for (auto &tip : tips) {
      tip = traj.getTrackState(tip).previous();
    }

    sortUniqueValidateTips();
  }
}

/// @brief This function takes two MultiTrajectory objects and corresponding
/// index lists (one of the backward pass, one of the forward pass), combines
/// them, applies smoothing, and returns a new, single-component MultiTrajectory
/// TODO this function does not handle outliers correctly at the moment I think
/// TODO change std::vector< size_t > to boost::small_vector for better
/// performance
template <typename traj_t>
auto smoothTrajectory(
    const MultiTrajectory<traj_t> &fwd,
    const std::vector<MultiTrajectoryTraits::IndexType> &fwdStartTips,
    const std::map<MultiTrajectoryTraits::IndexType, double> &fwdWeights,
    const MultiTrajectory<traj_t> &bwd,
    const std::vector<MultiTrajectoryTraits::IndexType> &bwdStartTips,
    const std::map<MultiTrajectoryTraits::IndexType, double> &bwdWeights,
    MultiTrajectory<traj_t> &res, MultiTrajectoryTraits::IndexType resTip,
    LoggerWrapper logger = getDummyLogger()) {
  visitMultiTrajectoryMultiComponents(
      bwd, bwdStartTips, [&](const auto &bwdTips) {
        const auto firstBwdState = bwd.getTrackState(bwdTips.front());
        const auto &currentSurface = firstBwdState.referenceSurface();

        // Search corresponding forward tips
        const auto bwdGeoId = currentSurface.geometryId();
        std::vector<MultiTrajectoryTraits::IndexType> fwdTips;

        for (const auto tip : fwdStartTips) {
          fwd.visitBackwards(tip, [&](const auto &state) {
            if (state.referenceSurface().geometryId() == bwdGeoId) {
              fwdTips.push_back(state.index());
            }
          });
        }

        // Check if we have forward tips
        if (fwdTips.empty()) {
          ACTS_WARNING("Did not find forward states on surface " << bwdGeoId);
          return;
        }

        // Ensure we have no duplicates
        std::sort(fwdTips.begin(), fwdTips.end());
        fwdTips.erase(std::unique(fwdTips.begin(), fwdTips.end()),
                      fwdTips.end());

        // Search state in result trajectory
        auto resIndex = MultiTrajectoryTraits::kInvalid;
        res.visitBackwards(resTip, [&](const auto &proxy) {
          if (proxy.referenceSurface().geometryId() == bwdGeoId) {
            resIndex = proxy.index();
          }
        });

        if (resIndex == MultiTrajectoryTraits::kInvalid) {
          ACTS_ERROR("Did not find " << bwdGeoId << " in result trajectory");
        }

        auto proxy = res.getTrackState(resIndex);

        using PredProjector =
            MultiTrajectoryProjector<StatesType::ePredicted, traj_t>;
        using FiltProjector =
            MultiTrajectoryProjector<StatesType::eFiltered, traj_t>;

        auto smoothedStateResult = bayesianSmoothing(
            fwdTips.begin(), fwdTips.end(), bwdTips.begin(), bwdTips.end(),
            PredProjector{fwd, fwdWeights}, FiltProjector{bwd, bwdWeights});

        if (!smoothedStateResult.ok()) {
          ACTS_WARNING("Smoothing failed on " << bwdGeoId);
          return;
        }

        const auto &smoothedState = *smoothedStateResult;

        const auto [smoothedMean, smoothedCov] =
            angleDescriptionSwitch(currentSurface, [&](const auto &desc) {
              return combineGaussianMixture(smoothedState, Acts::Identity{}, desc);
            });

        proxy.smoothed() = smoothedMean;
        proxy.smoothedCovariance() = smoothedCov.value();
        ACTS_VERBOSE("Added smoothed state to MultiTrajectory");
      });
}

}  // namespace detail
}  // namespace Acts
