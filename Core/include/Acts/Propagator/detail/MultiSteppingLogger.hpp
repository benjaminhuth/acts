// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Common.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Propagator/ConstrainedStep.hpp"
#include "Acts/Propagator/detail/SteppingLogger.hpp"

#include <memory>
#include <optional>
#include <vector>

#include "nlohmann/json.hpp"

namespace Acts {

class Surface;
class Layer;
class TrackingVolume;

namespace detail {

/// @brief a step-logger for the multi-stepping
///
/// It simply logs the constrained step length per step
struct MultiSteppingLogger {
  /// Simple result struct to be returned
  struct this_result {
    std::vector<std::vector<std::pair<double, Step>>> component_steps;
    std::vector<Step> mean_steps;
  };

  using result_type = this_result;

  /// Set the Logger to sterile
  bool sterile = false;

  /// SteppingLogger action for the ActionList of the Propagator
  ///
  /// @tparam stepper_t is the type of the Stepper
  /// @tparam propagator_state_t is the type of Propagator state
  ///
  /// @param [in,out] state is the mutable stepper state object
  /// @param [in,out] result is the mutable result object
  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t& state, const stepper_t& stepper,
                  result_type& result) const {
    // don't log if you have reached the target
    if (sterile or state.navigation.targetReached) {
      return;
    }

    {
      // record the propagation state
      Step step;
      step.position = stepper.position(state.stepping);
      step.momentum =
          stepper.momentum(state.stepping) * stepper.direction(state.stepping);

      if (state.navigation.currentSurface != nullptr) {
        // hang on to surface
        step.surface = state.navigation.currentSurface->getSharedPtr();
      }

      step.volume = state.navigation.currentVolume;
      result.mean_steps.push_back(std::move(step));
    }

    typename decltype(result.component_steps)::value_type components;
    for (auto cmp : stepper.constComponentIterable(state.stepping)) {
      Step step;
      step.stepSize = cmp.singleState(state).stepping.stepSize;
      step.position = cmp.pars().template segment<3>(eFreePos0);
      step.momentum =
          stepper.momentum(state.stepping) * stepper.direction(state.stepping);

      if (state.navigation.currentSurface != nullptr) {
        // hang on to surface
        step.surface = state.navigation.currentSurface->getSharedPtr();
      }
      step.volume = state.navigation.currentVolume;
      components.push_back({cmp.weight(), std::move(step)});
    }

    result.component_steps.push_back(components);
  }

  /// Pure observer interface
  /// - this does not apply to the logger
  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t& /*unused*/,
                  const stepper_t& /*unused*/) const {}
};

void to_json(nlohmann::json& data,
             const MultiSteppingLogger::result_type& res) {
  auto extract_pos = [](std::array<std::vector<float>, 3>& out,
                        const Vector3& in) {
    for (int i = 0; i < 3; ++i) {
      out[i].push_back(in[i]);
    }
  };

  data["mean_steps"] = [&]() {
    std::array<std::vector<float>, 3> r;
    for (const auto& s : res.mean_steps) {
      if (s.surface) {
        extract_pos(r, s.position);
      }
    }
    return r;
  }();

  data["component_steps"] = [&]() {
    std::vector<std::array<std::vector<float>, 3>> r;

    for (const auto& vs : res.component_steps) {
      r.resize(std::max(r.size(), vs.size()));

      for (auto i = 0ul; i < vs.size(); ++i) {
        if (vs[i].second.surface) {
          extract_pos(r[i], vs[i].second.position);
        }
      }
    }

    return r;
  }();
}

}  // namespace detail
}  // namespace Acts
