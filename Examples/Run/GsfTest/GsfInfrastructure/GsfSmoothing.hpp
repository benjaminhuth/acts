// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "GsfError.hpp"
#include "GsfUtils.hpp"

namespace Acts {

namespace detail {

/// @brief Smoothing function, which takes two ranges of
/// MultiTrajectory-indices and the corresponding projectors.
template <typename component_iterator_t, typename fwd_projector_t = Identity,
          typename bwd_projector_t = Identity>
auto bayesianSmoothing(component_iterator_t fwdBegin,
                       component_iterator_t fwdEnd,
                       component_iterator_t bwdBegin,
                       component_iterator_t bwdEnd,
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

  throw_assert(componentWeightsAreNormalized(
                   smoothedState,
                   [](const auto &tuple) { return std::get<double>(tuple); }),
               "smoothed state not normalized");

  return ResType(smoothedState);
}

/// @brief Projector type which maps a MultiTrajectory-Index to a tuple of
/// [weight, parameters, covariance]. Therefore, it contains a MultiTrajectory
/// and for now a std::map for the weights
template <StatesType type>
struct MultiTrajectoryProjector {
  const MultiTrajectory &mt;
  const std::map<std::size_t, double> &weights;

  auto operator()(std::size_t idx) const {
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

/// @brief This function takes two MultiTrajectory objects and corresponding
/// index lists (one of the backward pass, one of the forward pass), combines
/// them, applies smoothing, and returns a new, single-component MultiTrajectory
template <bool ReturnSmootedStates = false>
auto smoothAndCombineTrajectories(
    const MultiTrajectory &fwd,
    const std::vector<std::size_t> &fwdStartTips,
    const std::map<std::size_t, double> &fwdWeights,
    const MultiTrajectory &bwd,
    const std::vector<std::size_t> &bwdStartTips,
    const std::map<std::size_t, double> &bwdWeights,
    LoggerWrapper logger = getDummyLogger()) {
  // This vector gets only filled if ReturnSmootedStates is true
  std::vector<std::pair<const Surface *,
                        std::vector<std::tuple<double, BoundVector,
                                               std::optional<BoundSymMatrix>>>>>
      smoothedStates;

  // Use backward trajectory as basic trajectory, so that final trajectory is
  // ordered correctly. We ensure also that they are unique.
  std::vector<std::size_t> bwdTips = bwdStartTips;
  std::sort(bwdTips.begin(), bwdTips.end());
  bwdTips.erase(std::unique(bwdTips.begin(), bwdTips.end()), bwdTips.end());

  MultiTrajectory finalTrajectory;

  std::size_t lastTip = SIZE_MAX;

  // MultiTrajectory uses uint16_t internally TODO is none_of here correct?
  while (std::none_of(bwdTips.begin(), bwdTips.end(), [](auto i) {
    return i == std::numeric_limits<uint16_t>::max();
  })) {
    const auto firstBwdState = bwd.getTrackState(bwdTips.front());
    const auto &currentSurface = firstBwdState.referenceSurface();

    // Search corresponding forward tips
    const auto bwdGeoId = currentSurface.geometryId();
    std::vector<std::size_t> fwdTips;

    std::vector<std::size_t> visited;

    for (const auto tip : fwdStartTips) {
      fwd.visitBackwards(tip, [&](const auto &state) {
        std::cout << "idx " << state.index() << "\n";
        throw_assert(std::find(visited.begin(), visited.end(), state.index()) == visited.end(), "where at idx " << state.index() << "twice" );
        visited.push_back(state.index());

        if (state.referenceSurface().geometryId() == bwdGeoId) {
          fwdTips.push_back(state.index());
        }
      });
    }

    // Check if we have forward tips
    if (!fwdTips.empty()) {
      // Ensure we have no duplicates
      std::sort(fwdTips.begin(), fwdTips.end());
      fwdTips.erase(std::unique(fwdTips.begin(), fwdTips.end()), fwdTips.end());

      // Define some Projector types we need in the following
      using PredProjector = MultiTrajectoryProjector<StatesType::ePredicted>;
      using FiltProjector = MultiTrajectoryProjector<StatesType::eFiltered>;

      // Do the smoothing
      auto smoothedStateResult = bayesianSmoothing(
          fwdTips.begin(), fwdTips.end(), bwdTips.begin(), bwdTips.end(),
          PredProjector{fwd, fwdWeights}, FiltProjector{bwd, bwdWeights});

      if (smoothedStateResult.ok()) {
        const auto &smoothedState = *smoothedStateResult;

        if constexpr (ReturnSmootedStates) {
          smoothedStates.push_back({&currentSurface, smoothedState});
        }

        // Add state to MultiTrajectory
        lastTip =
            finalTrajectory.addTrackState(TrackStatePropMask::All, lastTip);
        auto proxy = finalTrajectory.getTrackState(lastTip);

        // This way I hope we copy all relevant flags and the calibrated field
        proxy.copyFrom(firstBwdState);

        // The predicted state is the forward pass
        const auto [fwdMeanPred, fwdCovPred] = combineComponentRange(
            fwdTips.begin(), fwdTips.end(), PredProjector{fwd, fwdWeights});
        proxy.predicted() = fwdMeanPred;
        proxy.predictedCovariance() = fwdCovPred.value();

        // The filtered state is the backward pass
        const auto [bwdMeanFilt, bwdCovFilt] = combineComponentRange(
            bwdTips.begin(), bwdTips.end(), FiltProjector{bwd, bwdWeights});
        proxy.filtered() = bwdMeanFilt;
        proxy.filteredCovariance() = bwdCovFilt.value();

        // The smoothed state is a combination
        const auto [smoothedMean, smoothedCov] =
            combineComponentRange(smoothedState.begin(), smoothedState.end());
        proxy.smoothed() = smoothedMean;
        proxy.smoothedCovariance() = smoothedCov.value();
      }
    } else {
      ACTS_WARNING("Did not find forward states on surface " << bwdGeoId);
    }

    // Update bwdTips to the next state
    for (auto &tip : bwdTips) {
      const auto p = bwd.getTrackState(tip);
      tip = p.previous();
    }

    std::sort(bwdTips.begin(), bwdTips.end());
    bwdTips.erase(std::unique(bwdTips.begin(), bwdTips.end()), bwdTips.end());
  }

  if constexpr (ReturnSmootedStates) {
    return std::make_tuple(finalTrajectory, lastTip, smoothedStates);
  } else {
    return std::make_tuple(finalTrajectory, lastTip);
  }
}

}  // namespace detail
}  // namespace Acts
