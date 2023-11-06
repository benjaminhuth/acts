// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <Eigen/Dense>
#include <experimental/simd>

namespace Acts {

template <int N>
using SimdType = std::experimental::fixed_size_simd<double, N>;

}  // namespace Acts

template <int N>
bool operator||(bool a, const Acts::SimdType<N>& s) {
  bool r = true;
  for (std::size_t i = 0; i < s.size(); ++i) {
    r = r && (a || s[i]);
  }
  return r;
}

namespace Eigen {

template <int N>
struct NumTraits<Acts::SimdType<N>> {
  using T = Acts::SimdType<N>;
  using Real = T;
  using Literal = T;
  using NonInteger = T;
  constexpr static int IsInteger = 0;
  constexpr static int IsSigned = 1;
  constexpr static int IsComplex = 0;
  constexpr static int RequireInitialization = 0;
  constexpr static int ReadCost = Eigen::HugeCost;
  constexpr static int AddCost = Eigen::HugeCost;
  constexpr static int MulCost = Eigen::HugeCost;

  static int digits10() { return 5; }
};

}  // namespace Eigen

namespace Acts {

template <int N, int Rows, int Cols>
auto extract(const Eigen::Matrix<SimdType<N>, Rows, Cols>& m, int i) {
  Eigen::Matrix<double, Rows, Cols> ret;
  for (int j = 0; j < Rows; ++j) {
    for (int k = 0; k < Cols; ++k) {
      ret(j, k) = m(j, k)[i];
    }
  }

  return ret;
}

template <int N>
std::ostream& operator<<(std::ostream& os, const SimdType<N>& s) {
  os << "Simd[ ";
  for (int i = 0; i < N; ++i) {
    os << s[i] << " ";
  }
  os << "]";
  return os;
}

template <int N>
double sum(const SimdType<N>& a) {
  return std::experimental::reduce(a);

}  // namespace SimdHelpers

}  // namespace Acts
