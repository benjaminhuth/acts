// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/TrackParameters.hpp"

namespace Acts {

namespace detail {

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

/// @brief Combine multiple components into one representative track state
/// object
template <typename source_link_t>
BoundTrackParameters combineMultiComponentState(
    const std::vector<GsfComponentCache<source_link_t>> &cmps,
    const Surface &surface) {
  BoundVector mean = BoundVector::Zero();
  BoundSymMatrix cov = BoundSymMatrix::Zero();
  double sumOfWeights{0.0};

  const double referencePhi =
      cmps.front().trackStateProxy->filtered()[eBoundPhi];

  // clang-format off
  // x = \sum_{l} w_l * x_l
  // C = \sum_{l} w_l * C_l + \sum_{l} \sum_{m>l} w_l * w_m * (x_l - x_m)(x_l - x_m)^T
  // clang-format on
  for (auto l = cmps.cbegin(); l != cmps.cend(); ++l) {
    BoundVector pars_l = l->trackStateProxy->filtered();

    // Avoid problems with cyclic phi
    const double deltaPhi = referencePhi - pars_l[eBoundPhi];

    if (deltaPhi > M_PI) {
      pars_l[eBoundPhi] += 2 * M_PI;
    } else if (deltaPhi < -M_PI) {
      pars_l[eBoundPhi] -= 2 * M_PI;
    }

    sumOfWeights += l->weight;
    mean += l->weight * pars_l;
    cov += l->weight * l->trackStateProxy->filteredCovariance();

    for (auto m = std::next(l); m != cmps.end(); ++m) {
      const BoundVector diff = pars_l - m->trackStateProxy->filtered();

      cov += l->weight * m->weight * diff * diff.transpose();
    }

    throw_assert(surface.geometryId() ==
                     l->trackStateProxy->referenceSurface().geometryId(),
                 "surface mismatch");
  }

  throw_assert(std::abs(sumOfWeights - 1.0) < 1.e-8,
               "weights are not normalized");

  return BoundTrackParameters(surface.getSharedPtr(), mean, cov);
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

    const auto covFwdInv = forward[i].covariance()->inverse();
    const auto covBwdInv = backward[i].covariance()->inverse();

    const BoundSymMatrix covInv = covFwdInv + covBwdInv;

    const BoundVector params = covInv * (covFwdInv * forward[i].parameters() +
                                         covBwdInv * backward[i].parameters());

    ret.push_back(
        BoundTrackParameters(forward[i].referenceSurface().getSharedPtr(),
                             params, covInv.inverse()));
  }

  return ret;
}

}  // namespace detail

}  // namespace Acts
