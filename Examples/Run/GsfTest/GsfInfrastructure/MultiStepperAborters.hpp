// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Surfaces/Surface.hpp"

namespace Acts {

/// This
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
    using SingleStepper = typename stepper_t::SingleStepper;

    struct DummyState {
      typename SingleStepper::State& stepping;
      decltype(state.navigation)& navigation;
      decltype(state.options)& options;
      GeometryContext geoContext;
    };

    bool reached = true;
    for (auto& cmp_state : state.stepping.components) {
      DummyState dummyState{cmp_state.state, state.navigation, state.options,
                            state.geoContext};
      if (!SurfaceReached{}(dummyState,
                            static_cast<const SingleStepper&>(stepper),
                            targetSurface)) {
        reached = false;
      }
    }

    return reached;
  }
};
}  // namespace Acts
