// This file is part of the Acts project.
//
// Copyright (C) 2016-2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Material/Interactions.hpp"

#include <xsimd/xsimd.hpp>

#define SIMD_EIGEN

/// This file contains functions overloads or wrappers, which allow common
/// mathematical operations for normal Eigen types (based on ActsScalar which is
/// probably double) and as well for nested Eigen types (e.g., Eigen::Matrix<
/// Eigen::Array4d, N, 1 >).

namespace Acts {
    
////////////////////////////////////
////// DEFINE THE SIMD TYPE ////////
////////////////////////////////////
    
#ifdef SIMD_EIGEN
template<int N>
using SimdType = Eigen::Array<ActsScalar, N, 1>;
#endif
    
#ifdef SIMD_XSIMD
template<int N>
using SimdType = xsimd::batch<ActsScalar, N>;
#endif

namespace SimdHelpers {

#ifdef SIMD_XSIMD
template<int N>
auto sum(const SimdType<N> &a)
{
    return xsimd::hadd(a);
}
#endif

#ifdef SIMD_EIGEN
template<typename T>
auto sum(const Eigen::ArrayBase<T> &a)
{
    return a.sum();
}
#endif

template <typename A, typename B>
auto cross(const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b) {
  using ScalarA = typename A::Scalar;
  using ScalarB = typename B::Scalar;

  static_assert(A::RowsAtCompileTime == 3);
  static_assert(B::RowsAtCompileTime == 3 && B::ColsAtCompileTime == 1);
  static_assert(std::is_same_v<ScalarA, ScalarB>);

  // Standard Cross Product
  if constexpr (std::is_same_v<ScalarA, ActsScalar>) {
    return a.cross(b);
  } 
  // Manual Cross Product for non default-Scalars
  else if constexpr( A::ColsAtCompileTime == 1 ) {
    Eigen::Matrix<ScalarA, 3, 1> ret;

    ret[0] = a[1] * b[2] - a[2] * b[1];
    ret[1] = a[2] * b[0] - a[0] * b[2];
    ret[2] = a[0] * b[1] - a[1] * b[0];

    return ret;
  }
  // Columnwise Cross Product if A is a 3x3 Matrix
  else if constexpr( A::ColsAtCompileTime == 3 ){
    Eigen::Matrix<ScalarA, 3, 3> ret;
    
    ret.col(0) = cross(a.col(0), b);
    ret.col(1) = cross(a.col(1), b);
    ret.col(2) = cross(a.col(2), b);
    
    return ret;
  }
}

template <typename A, typename B>
auto hypot(const Eigen::ArrayBase<A>& a, const Eigen::ArrayBase<B>& b) {
  // Compute TODO better use loop and std::hypot for comonents? because of
  // std::hypot's internal checking?
  return sqrt(a * a + b * b);
}

template <int N>
auto computeEnergyLossMode(const MaterialSlab& slab, int pdg, float m,
                           const SimdType<N>& qOverP,
                           float q = UnitConstants::e) {
  SimdType<N> ret;

  for (int i = 0; i < N; ++i)
    ret[0] = computeEnergyLossMode(slab, pdg, m, qOverP[i], q);

  return ret;
}

template <int N>
auto computeEnergyLossMean(const MaterialSlab& slab, int pdg, float m,
                           const SimdType<N>& qOverP,
                           float q = UnitConstants::e) {
  SimdType<N> ret;

  for (int i = 0; i < N; ++i)
    ret[0] = computeEnergyLossMean(slab, pdg, m, qOverP[i], q);

  return ret;
}

template <int N>
auto deriveEnergyLossMeanQOverP(const MaterialSlab& slab, int pdg, float m,
                                const SimdType<N>& qOverP,
                                float q = UnitConstants::e) {
  SimdType<N> ret;

  for (int i = 0; i < N; ++i)
    ret[0] = deriveEnergyLossMeanQOverP(slab, pdg, m, qOverP[i], q);

  return ret;
}

template <int N>
auto deriveEnergyLossModeQOverP(const MaterialSlab& slab, int pdg, float m,
                                const SimdType<N>& qOverP,
                                float q = UnitConstants::e) {
  SimdType<N> ret;

  for (int i = 0; i < N; ++i)
    ret[0] = deriveEnergyLossModeQOverP(slab, pdg, m, qOverP[i], q);

  return ret;
}

}  // namespace SimdHelpers

}  // namespace Acts
