// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/MultiComponentBoundTrackParameters.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/TrackParameters.hpp"

namespace Acts {

namespace detail {

/// @brief A multi component state
using MultiComponentState =
    std::pair<std::shared_ptr<const Surface>,
              std::vector<std::tuple<ActsScalar, BoundVector,
                                     std::optional<BoundSymMatrix>>>>;

/// @brief Struct which contains all needed information throughout the
/// processing of the components in the GSF
template <typename source_link_t>
struct GsfComponentCache {
  /// Where to find the parent component in the MultiTrajectory
  std::size_t parentIndex;

  /// The weight of the component
  ActsScalar weight;

  /// The predicted track state from the stepper
  BoundVector predictedPars;
  std::optional<BoundSymMatrix> predictedCov;

  /// The proxyTrackState from the MultiTrajectory
  using Proxy = typename MultiTrajectory<source_link_t>::TrackStateProxy;
  std::optional<Proxy> trackStateProxy;

  /// Other quantities TODO are they really needed here? seems they are
  /// reinitialized to Identity etc.
  BoundMatrix jacobian;
  BoundToFreeMatrix jacToGlobal;
  FreeMatrix jacTransport;
  FreeVector derivative;

  /// We need to preserve the path length
  ActsScalar pathLength;
};

/// @brief reweight MultiComponentState
void normalizeMultiComponentState(MultiComponentState &state)
{
    ActsScalar sum{0.0};

    for(const auto &[weight, pars, cov] : state.second) {
        sum += weight;
    }

    for(auto &[weight, pars, cov] : state.second) {
        weight /= sum;
    }
}

/// @brief Combine multiple components into one representative track state
/// object. The function takes iterators to allow for arbitrary ranges to be
/// combined
template <typename component_iterator_t>
BoundTrackParameters combineMultiComponentState(
    const component_iterator_t begin, const component_iterator_t end,
    const Surface &surface) {
  BoundVector mean = BoundVector::Zero();
  BoundSymMatrix cov = BoundSymMatrix::Zero();
  double sumOfWeights{0.0};

  const double referencePhi = std::get<BoundVector>(*begin)[eBoundPhi];

  // clang-format off
  // x = \sum_{l} w_l * x_l
  // C = \sum_{l} w_l * C_l + \sum_{l} \sum_{m>l} w_l * w_m * (x_l - x_m)(x_l - x_m)^T
  // clang-format on
  for (auto l = begin; l != end; ++l) {
    throw_assert(std::get<std::optional<BoundSymMatrix>>(*l),
                 "we require a covariance here");

    BoundVector pars_l = std::get<BoundVector>(*l);

    // Avoid problems with cyclic phi
    const double deltaPhi = referencePhi - pars_l[eBoundPhi];

    if (deltaPhi > M_PI) {
      pars_l[eBoundPhi] += 2 * M_PI;
    } else if (deltaPhi < -M_PI) {
      pars_l[eBoundPhi] -= 2 * M_PI;
    }

    sumOfWeights += std::get<ActsScalar>(*l);
    mean += std::get<ActsScalar>(*l) * pars_l;
    cov +=
        std::get<ActsScalar>(*l) * *std::get<std::optional<BoundSymMatrix>>(*l);

    for (auto m = std::next(l); m != end; ++m) {
      throw_assert(std::get<std::optional<BoundSymMatrix>>(*m),
                   "we require a covariance here");

      const BoundVector diff = pars_l - std::get<BoundVector>(*m);

      cov += std::get<ActsScalar>(*l) * std::get<ActsScalar>(*m) * diff *
             diff.transpose();
    }
  }

  throw_assert(std::abs(sumOfWeights - 1.0) < 1.e-8,
               "weights are not normalized");

  return BoundTrackParameters(surface.getSharedPtr(), mean, cov);
}

/// @brief Function that reduces the number of components. at the moment,
/// this is just by erasing the components with the lowest weights.
/// Finally, the components are reweighted so the sum of the weights is
/// still 1
/// TODO If we create new components here to preserve the mean or if we do some
/// component merging, how can this be applied in the MultiTrajectory
template <typename source_link_t>
void reduceNumberOfComponents(
    std::vector<GsfComponentCache<source_link_t>> &components,
    std::size_t maxRemainingComponents) {
  // The elements should be sorted by weight (high -> low)
  std::sort(begin(components), end(components),
            [](const auto &a, const auto &b) { return a.weight > b.weight; });

  // Remove elements by resize
  components.erase(begin(components) + maxRemainingComponents, end(components));

  // Reweight after removal
  const auto sum_of_weights = std::accumulate(
      begin(components), end(components), 0.0,
      [](auto sum, const auto &cmp) { return sum + cmp.weight; });

  for (auto &cmp : components) {
    cmp.weight /= sum_of_weights;
  }
}

/// @brief Reweight the components according to `R. Frühwirth, "Track fitting
/// with non-Gaussian noise"`. See also the implementation in Athena at
/// PosteriorWeightsCalculator.cxx
template <typename source_link_t>
void reweightComponents(std::vector<GsfComponentCache<source_link_t>> &cmps) {
  // Helper Function to compute detR
  auto computeDetR = [](const auto &trackState) -> ActsScalar {
    const auto predictedCovariance = trackState.predictedCovariance();

    return visit_measurement(
        trackState.calibrated(), trackState.calibratedCovariance(),
        trackState.calibratedSize(),
        [&](const auto calibrated, const auto calibratedCovariance) {
          constexpr size_t kMeasurementSize =
              decltype(calibrated)::RowsAtCompileTime;
          const auto H =
              trackState.projector()
                  .template topLeftCorner<kMeasurementSize, eBoundSize>()
                  .eval();

          return (H * predictedCovariance * H.transpose() +
                  calibratedCovariance)
              .determinant();
        });
  };

  // Find minChi2, this can be used to factor some things later in the
  // exponentiation
  const auto minChi2 = std::min_element(cmps.begin(), cmps.end(),
                                        [](const auto &a, const auto &b) {
                                          return a.trackStateProxy->chi2() <
                                                 b.trackStateProxy->chi2();
                                        })
                           ->trackStateProxy->chi2();

  // Compute new weights and reweight
  double sumOfWeights = 0.;

  for (auto &cmp : cmps) {
    const double chi2 = cmp.trackStateProxy->chi2() - minChi2;
    const double detR = computeDetR(*cmp.trackStateProxy);

    cmp.weight *= std::sqrt(1. / detR) * std::exp(-0.5 * chi2);
    sumOfWeights += cmp.weight;
  }

  throw_assert(sumOfWeights > 0.,
               "The sum of the weights needs to be positive");

  for (auto &cmp : cmps) {
    cmp.weight *= (1. / sumOfWeights);
  }
}

/// @brief Combine a forward pass and a backward pass to a smoothed trajectory.
/// This is part of the Weighted-Mean-Smoother implementation for the GSF
std::vector<BoundTrackParameters> combineForwardAndBackwardPass(
    const std::vector<BoundTrackParameters> &forward,
    const std::vector<BoundTrackParameters> &backward) {
  throw_assert(forward.size() == backward.size(),
               "forward and backwards pass must match in size");

  std::vector<BoundTrackParameters> ret;
  ret.reserve(forward.size());

  for (auto i = 0ul; i < forward.size(); ++i) {
    throw_assert(forward[i].referenceSurface().geometryId() ==
                     backward[i].referenceSurface().geometryId(),
                 "ID must be equal");

    //     const BoundSymMatrix covFwdInv = forward[i].covariance()->inverse();
    //     const BoundSymMatrix covBwdInv = backward[i].covariance()->inverse();
    //
    //     const BoundSymMatrix covInv = covFwdInv + covBwdInv;
    //
    //     const BoundVector params = covInv * (covFwdInv *
    //     forward[i].parameters() +
    //                                          covBwdInv *
    //                                          backward[i].parameters());
    //         ret.push_back(
    //         BoundTrackParameters(forward[i].referenceSurface().getSharedPtr(),
    //                              params, covInv.inverse()));

    // Where do these equations come from???
    const BoundSymMatrix covSummed =
        *forward[i].covariance() + *backward[i].covariance();
    const BoundSymMatrix K = *forward[i].covariance() * covSummed.inverse();
    const BoundSymMatrix newCov = K * *backward[i].covariance();

    const BoundVector xNew =
        forward[i].parameters() +
        K * (backward[i].parameters() - forward[i].parameters());

    ret.push_back(BoundTrackParameters(
        forward[i].referenceSurface().getSharedPtr(), xNew, newCov));
  }

  return ret;
}

/// @brief Extracts a MultiComponentState from a MultiTrajectory.
///
/// @param usePredicted Wether to use the predicted state (true) or the
/// filtered state (false) from the MultiTrajectory.
template <typename source_link_t>
auto extractMultiComponentStates(const MultiTrajectory<source_link_t> &traj,
                                 std::vector<size_t> tips,
                                 const std::map<size_t, ActsScalar> &weights,
                                 bool usePredicted) {
  std::vector<MultiComponentState> ret;

  // MultiTrajectory uses uint16_t internally
  while (std::none_of(tips.begin(), tips.end(), [](auto i) {
    return i == std::numeric_limits<uint16_t>::max();
  })) {
    std::sort(tips.begin(), tips.end());
    tips.erase(std::unique(tips.begin(), tips.end()), tips.end());

    MultiComponentState state;
    std::get<1>(state).reserve(tips.size());

    for (auto &tip : tips) {
      const auto proxy = traj.getTrackState(tip);

      throw_assert(weights.find(tip) != weights.end(),
                   "Could not find weight for idx " << tip);

      if (usePredicted) {
        std::get<1>(state).push_back(
            {weights.at(tip), proxy.predicted(), proxy.predictedCovariance()});
      } else {
        std::get<1>(state).push_back(
            {weights.at(tip), proxy.filtered(), proxy.filteredCovariance()});
      }

      if (!std::get<0>(state)) {
        std::get<0>(state) = proxy.referenceSurface().getSharedPtr();
      } else {
        throw_assert(std::get<0>(state)->geometryId() ==
                         proxy.referenceSurface().geometryId(),
                     "surface mismatch");
      }

      tip = proxy.previous();
    }

    ret.push_back(state);
  }

  return ret;
}

/// @brief This function applies the bayesian smoothing by combining a
/// forward MultiComponentState and a backward MultiComponentState into a new
/// MultiComponentState. The result is not normalized, and also not component
/// reduction is done
auto bayesianSmoothing(const MultiComponentState &fwd,
                       const MultiComponentState &bwd) {
  MultiComponentState smoothedState;
  std::get<1>(smoothedState)
      .reserve(std::get<1>(fwd).size() + std::get<1>(bwd).size());

  throw_assert(std::get<0>(fwd)->geometryId() == std::get<0>(bwd)->geometryId(),
               "surface mismatch");
  std::get<0>(smoothedState) = std::get<0>(fwd);

  for (const auto &[weight_a, pars_a, cov_a] : std::get<1>(fwd)) {
    throw_assert(cov_a, "for now we require a covariance here");

    for (const auto &[weight_b, pars_b, cov_b] : std::get<1>(bwd)) {
      throw_assert(cov_b, "for now we require a covariance here");

      const auto summedCov = *cov_a + *cov_b;
      const auto K = *cov_a * summedCov.inverse();
      const auto new_pars = (pars_a + K * (pars_b - pars_a)).eval();
      const auto new_cov = (K * *cov_b).eval();

      const auto diff = pars_a - pars_b;
      const ActsScalar exponent = diff.transpose() * summedCov.inverse() * diff;

      const auto new_weight = std::exp(-0.5 * exponent) * weight_a * weight_b;

      std::get<1>(smoothedState).push_back({new_weight, new_pars, new_cov});
    }
  }

  return smoothedState;
}

}  // namespace detail

}  // namespace Acts
