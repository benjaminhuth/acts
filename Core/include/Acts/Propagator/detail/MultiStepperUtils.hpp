// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/EventData/MultiComponentTrackParameters.hpp"
#include "Acts/EventData/detail/CorrectedTransformationFreeToBound.hpp"
#include "Acts/Propagator/MultiStepperError.hpp"
#include "Acts/Surfaces/Surface.hpp"

#include <vector>

#pragma once

namespace Acts::detail {

using MultiStepperBoundState = std::tuple<MultiComponentBoundTrackParameters,
                                          BoundSquareMatrix, ActsScalar>;
using MultiStepperCurvilinearState =
    std::tuple<MultiComponentCurvilinearTrackParameters, BoundSquareMatrix,
               ActsScalar>;

template <typename multistepper_t>
Acts::Result<MultiStepperBoundState> multiComponentBoundState(
    const multistepper_t& stepper, typename multistepper_t::State& state,
    const Surface& surface, bool transportCov,
    const FreeToBoundCorrection& freeToBoundCorrection) {
  std::vector<std::tuple<double, BoundVector, BoundSquareMatrix>> cmps;
  cmps.reserve(stepper.numberComponents(state));
  double accumulatedPathLength = 0.0;
  
  using R = Acts::Result<MultiStepperBoundState>;

  auto components = stepper.componentIterable(state);
  for (auto cmp : components) {
    // Force the component to be on the surface
    // This needs to be done because of the `averageOnSurface`-option of the
    // `MultiStepperSurfaceReached`-Aborter, which can be configured to end the
    // propagation when the mean of all components reached the destination
    // surface. Thus, it is not garantueed that all states are actually
    // onSurface.
    cmp.pars().template segment<3>(eFreePos0) =
        surface
            .intersect(state.geoContext,
                       cmp.pars().template segment<3>(eFreePos0),
                       cmp.pars().template segment<3>(eFreeDir0), false)
            .closest()
            .position();

    auto bs = cmp.boundState(surface, transportCov, freeToBoundCorrection);

    if (bs.ok()) {
      const auto& btp = std::get<BoundTrackParameters>(*bs);
      cmps.emplace_back(
          cmp.weight(), btp.parameters(),
          btp.covariance().value_or(Acts::BoundSquareMatrix::Zero()));
      accumulatedPathLength += std::get<double>(*bs) * cmp.weight();
    }
  }

  if (cmps.empty()) {
    return R{MultiStepperError::AllComponentsConversionToBoundFailed};
  }

  MultiComponentBoundTrackParameters mcbtp(surface.getSharedPtr(), cmps,
                                           state.particleHypothesis);

  return MultiStepperBoundState{std::move(mcbtp), BoundSquareMatrix::Zero(),
                                accumulatedPathLength};
}

template <typename multistepper_t>
MultiStepperCurvilinearState multiComponentCurvilinearState(
    const multistepper_t& stepper, typename multistepper_t::State& state,
    bool transportCov) {
  std::vector<
      std::tuple<double, Vector4, Vector3, ActsScalar, BoundSquareMatrix>>
      cmps;
  cmps.reserve(stepper.numberComponents(state));
  double accumulatedPathLength = 0.0;

  auto components = stepper.componentIterable(state);
  for (auto cmp : components) {
    const auto [cp, jac, pl] = cmp.curvilinearState(transportCov);

    cmps.emplace_back(cmp.weight(), cp.fourPosition(state.geoContext),
                      cp.direction(), (cp.charge() / cp.absoluteMomentum()),
                      cp.covariance().value_or(BoundSquareMatrix::Zero()));
    accumulatedPathLength += cmp.weight() * pl;
  }

  MultiComponentCurvilinearTrackParameters mcctp(cmps,
                                                 state.particleHypothesis);
  return MultiStepperCurvilinearState{std::move(mcctp), BoundSquareMatrix::Zero(),
                                      accumulatedPathLength};
}

}  // namespace Acts::detail
