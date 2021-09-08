// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Surfaces/Surface.hpp"

namespace Acts {

struct MultiStepperSurfaceReached {
  MultiStepperSurfaceReached() = default;

  /// boolean operator for abort condition without using the result
  ///
  /// @tparam propagator_state_t Type of the propagator state
  /// @tparam stepper_t Type of the stepper
  ///
  /// @param [in,out] state The propagation state object
  /// @param [in] stepper Stepper used for propagation
  template <typename propagator_state_t, typename stepper_t>
  bool operator()(propagator_state_t& state, const stepper_t& stepper) const {
    return (*this)(state, stepper, *state.navigation.targetSurface);
  }

  /// boolean operator for abort condition without using the result
  ///
  /// @tparam propagator_state_t Type of the propagator state
  /// @tparam stepper_t Type of the stepper
  ///
  /// @param [in,out] state The propagation state object
  /// @param [in] stepper Stepper used for the progation
  /// @param [in] targetSurface The target surface
  template <typename propagator_state_t, typename stepper_t>
  bool operator()(propagator_state_t& state, const stepper_t& stepper,
                  const Surface& targetSurface) const {
    const auto& logger = state.options.logger;
    if (state.navigation.targetReached) {
      return true;
    }
    // Check if the cache filled the currentSurface - or if we are on the
    // surface
    if ((state.navigation.currentSurface &&
         state.navigation.currentSurface == &targetSurface)) {
      ACTS_VERBOSE("Target: x | "
                   << "Target surface reached.");
      // reaching the target calls a navigation break
      state.navigation.targetReached = true;
      return true;
    }
    
    auto status = stepper.updateSurfaceStatusAsAborter(state.stepping, targetSurface, true);
    
    // The target is reached
    bool targetReached = (status ==
                          Intersection3D::Status::onSurface);

    // Return true if you fall below tolerance
    if (targetReached) {
      ACTS_VERBOSE("Target: x | "
                   << "Target surface reached by all components");
      
      state.navigation.currentSurface = &targetSurface;
      ACTS_VERBOSE("Target: x | "
                   << "Current surface set to target surface  "
                   << state.navigation.currentSurface->geometryId());
      state.navigation.targetReached = true;
    } else {
      ACTS_VERBOSE("Target: 0 | "
                   << "Target stepSize (surface) updated to "
                   << stepper.outputStepSize(state.stepping));
    }
    
    // path limit check
    return targetReached;
  }
};
}  // namespace Acts
