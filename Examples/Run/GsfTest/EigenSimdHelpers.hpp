// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"

/// This file contains functions overloads or wrappers, which allow common
/// mathematical operations for normal Eigen types (based on ActsScalar which is
/// probably double) and as well for nested Eigen types (e.g., Eigen::Matrix<
/// Eigen::Array4d, N, 1 >).

namespace Acts {

namespace SimdHelpers {

template <typename T>
bool operator<(const Eigen::ArrayBase<T>& a, const typename T::Scalar b) {
  return (a < b).all();
}

template <typename A, typename B>
auto cross(const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b) {
  using ScalarA = typename A::Scalar;
  using ScalarB = typename B::Scalar;

  static_assert(A::RowsAtCompileTime == 3 && A::ColsAtCompileTime == 1);
  static_assert(B::RowsAtCompileTime == 3 && B::ColsAtCompileTime == 1);
  static_assert(std::is_same_v<ScalarA, ScalarB>);

  if constexpr (std::is_same_v<ScalarA, ActsScalar>) {
    return a.cross(b);
  } else {
    Eigen::Matrix<ScalarA, 3, 1> ret;

    ret[0] = a[1] * b[2] - a[2] * b[1];
    ret[1] = a[2] * b[0] - a[0] * b[2];
    ret[2] = a[0] * b[1] - a[1] * b[0];

    return ret;
  }
}

template <typename A, typename B>
auto hypot(const Eigen::ArrayBase<A>& a, const Eigen::ArrayBase<B>& b) {
  // Compute TODO better use loop and std::hypot for comonents? because of
  // std::hypot's internal checking?
  return sqrt(a * a + b * b);
}

}  // namespace SimdHelpers

}  // namespace Acts
