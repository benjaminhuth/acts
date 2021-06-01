// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/TrackParametrization.hpp"

#include "SimdHelpers.hpp"

namespace Acts {

/// @brief Reducer struct which reduces the multicomponent state to simply by
/// summing the weighted values
struct WeightedComponentReducer {
  //////////////////////////
  // For the SIMD Stepper //
  //////////////////////////
  template <int N>
  using SimdVector3 = Eigen::Matrix<SimdType<N>, 3, 1>;

  template <int N>
  using SimdFreeVector = Eigen::Matrix<SimdType<N>, eFreeSize, 1>;

  template <int N>
  static Vector3 toVector3(const SimdFreeVector<N>& f, const SimdType<N>& w,
                           const FreeIndices i) {
    SimdVector3<N> multi = f.template segment<3>(i);
    multi[0] *= w;
    multi[1] *= w;
    multi[2] *= w;

    Vector3 ret;
    ret << SimdHelpers::sum(multi[0]), SimdHelpers::sum(multi[1]),
        SimdHelpers::sum(multi[2]);

    return ret;
  }

  template <int N>
  static Vector3 position(const SimdFreeVector<N>& f, const SimdType<N>& w) {
    return toVector3(f, w, eFreePos0);
  }

  template <int N>
  static Vector3 direction(const SimdFreeVector<N>& f, const SimdType<N>& w) {
    return toVector3(f, w, eFreeDir0).normalized();
  }

  template <int N>
  static ActsScalar momentum(const SimdFreeVector<N>& f, const SimdType<N>& w,
                             const ActsScalar q) {
    return SimdHelpers::sum((1 / (f[eFreeQOverP] / q)) * w);
  }

  template <int N>
  static ActsScalar time(const SimdFreeVector<N>& f, const SimdType<N>& w) {
    return SimdHelpers::sum(f[eFreeTime] * w);
  }

  //////////////////////////
  // For the Loop Stepper //
  //////////////////////////
  template <typename component_t>
  static Vector3 toVector3(const std::vector<component_t>& comps,
                           const FreeIndices i) {
    return std::accumulate(
        begin(comps), end(comps), Vector3{Vector3::Zero()},
        [i](const auto& sum, const auto& cmp) -> Vector3 {
          return sum + cmp.weight * cmp.state.pars.template segment<3>(i);
        });
  }

  template <typename component_t>
  static ActsScalar toScalar(const std::vector<component_t>& comps,
                             const FreeIndices i) {
    return std::accumulate(begin(comps), end(comps), ActsScalar{0.},
                           [i](const auto& sum, const auto& cmp) -> ActsScalar {
                             return sum + cmp.weight * cmp.state.pars[i];
                           });
  }

  template <typename component_t>
  static Vector3 position(const std::vector<component_t>& comps) {
    return toVector3(comps, eFreePos0);
  }

  template <typename component_t>
  static Vector3 direction(const std::vector<component_t>& comps) {
    return toVector3(comps, eFreeDir0).normalized();
  }

  template <typename component_t>
  static ActsScalar momentum(const std::vector<component_t>& comps,
                             const ActsScalar q) {
    return 1 / (toScalar(comps, eFreeQOverP) / q);
  }

  template <typename component_t>
  static ActsScalar time(const std::vector<component_t>& comps) {
    return toScalar(comps, eFreeTime);
  }
};

}  // namespace Acts
