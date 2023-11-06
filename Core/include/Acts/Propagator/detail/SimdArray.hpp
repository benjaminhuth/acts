// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <array>
#include <ostream>

#include <Eigen/Dense>

namespace Acts {

#define ACTS_SIMD_PLAIN_ARRAY

#if defined(__AVX512__)
#define SIMD_ALIGNMENT 64
#pragma message "AVX512"
#elif defined(__AVX2__)
#define SIMD_ALIGNMENT 32
#pragma message "AVX2"
#elif defined(__AVX__) || defined(__SSE__) || defined(__MMX__)
#define SIMD_ALIGNMENT 16
#pragma message "AVX/SSE/MMX"
#else
#define SIMD_ALIGNMENT 8
#pragma message "no SIMD"
#endif

#ifdef __clang__
#define ACTS_LOOP_IVDEP                                      \
  _Pragma("vectorize(enable)") _Pragma("interleave(enable)") \
      _Pragma("vectorize_width(4)")
#else
#define ACTS_LOOP_IVDEP _Pragma("GCC ivdep")
#endif
// #define ACTS_LOOP_IVDEP

#define ACTS_RESTRICT __restrict__
// #define ACTS_RESTRICT

#define ACTS_INLINE inline
// #define ACTS_INLINE

// #define ACTS_ALIGNMENT alignas(SIMD_ALIGNMENT)
#define ACTS_ALIGNMENT

template <int N>
class SimdType {
  ACTS_ALIGNMENT double m_data[N];

 public:
  constexpr static int Size = N;

  SimdType() = default;

  SimdType(double v) { std::fill(std::begin(m_data), std::end(m_data), v); }

  ACTS_INLINE double& operator[](std::size_t i) { return m_data[i]; }
  ACTS_INLINE double operator[](std::size_t i) const { return m_data[i]; }

  ACTS_INLINE double* data() { return m_data; }
  ACTS_INLINE const double* data() const { return m_data; }

  ACTS_INLINE void operator+=(const SimdType& ACTS_RESTRICT other) {
    ACTS_LOOP_IVDEP
    for (auto i = 0ul; i < N; ++i) {
      (*this)[i] += other[i];
    }
  }
  ACTS_INLINE void operator+=(double other) {
    ACTS_LOOP_IVDEP
    for (auto i = 0ul; i < N; ++i) {
      (*this)[i] += other;
    }
  }

  ACTS_INLINE void operator-=(const SimdType& ACTS_RESTRICT other) {
    ACTS_LOOP_IVDEP
    for (auto i = 0ul; i < N; ++i) {
      (*this)[i] -= other[i];
    }
  }

  ACTS_INLINE void operator-=(double other) {
    ACTS_LOOP_IVDEP
    for (auto i = 0ul; i < N; ++i) {
      (*this)[i] -= other;
    }
  }

  ACTS_INLINE void operator*=(const SimdType& ACTS_RESTRICT other) {
    ACTS_LOOP_IVDEP
    for (auto i = 0ul; i < N; ++i) {
      (*this)[i] *= other[i];
    }
  }

  ACTS_INLINE void operator*=(double other) {
    ACTS_LOOP_IVDEP
    for (auto i = 0ul; i < N; ++i) {
      (*this)[i] *= other;
    }
  }

  ACTS_INLINE void operator/=(const SimdType& ACTS_RESTRICT other) {
    ACTS_LOOP_IVDEP
    for (auto i = 0ul; i < N; ++i) {
      (*this)[i] /= other[i];
    }
  }

  ACTS_INLINE void operator/=(double other) {
    ACTS_LOOP_IVDEP
    for (auto i = 0ul; i < N; ++i) {
      (*this)[i] /= other;
    }
  }
};

#undef SIMD_ALIGNMENT

#define DEFINE_BINARY_VEC_OPERATOR(OP, SYM)                 \
  template <int N>                                          \
  ACTS_INLINE auto OP(const SimdType<N>& ACTS_RESTRICT a,   \
                      const SimdType<N>& ACTS_RESTRICT b) { \
    SimdType<N> r;                                          \
    ACTS_LOOP_IVDEP                                         \
    for (auto i = 0ul; i < N; ++i) {                        \
      r[i] = a[i] SYM b[i];                                 \
    }                                                       \
    return r;                                               \
  }

DEFINE_BINARY_VEC_OPERATOR(operator*, *)
DEFINE_BINARY_VEC_OPERATOR(operator+, +)
DEFINE_BINARY_VEC_OPERATOR(operator-, -)
DEFINE_BINARY_VEC_OPERATOR(operator/, /)

#define DEFINE_BINARY_LSCALAR_OPERATOR(OP, SYM)                       \
  template <int N>                                                    \
  ACTS_INLINE auto OP(double a, const SimdType<N>& ACTS_RESTRICT b) { \
    SimdType<N> r;                                                    \
    ACTS_LOOP_IVDEP                                                   \
    for (auto i = 0ul; i < N; ++i) {                                  \
      r[i] = a SYM b[i];                                              \
    }                                                                 \
    return r;                                                         \
  }

DEFINE_BINARY_LSCALAR_OPERATOR(operator*, *)
DEFINE_BINARY_LSCALAR_OPERATOR(operator+, +)
DEFINE_BINARY_LSCALAR_OPERATOR(operator-, -)
DEFINE_BINARY_LSCALAR_OPERATOR(operator/, /)

#define DEFINE_BINARY_RSCALAR_OPERATOR(OP, SYM)                       \
  template <int N>                                                    \
  ACTS_INLINE auto OP(const SimdType<N>& ACTS_RESTRICT a, double b) { \
    SimdType<N> r;                                                    \
    ACTS_LOOP_IVDEP                                                   \
    for (auto i = 0ul; i < N; ++i) {                                  \
      r[i] = a[i] SYM b;                                              \
    }                                                                 \
    return r;                                                         \
  }

DEFINE_BINARY_RSCALAR_OPERATOR(operator*, *)
DEFINE_BINARY_RSCALAR_OPERATOR(operator+, +)
DEFINE_BINARY_RSCALAR_OPERATOR(operator-, -)
DEFINE_BINARY_RSCALAR_OPERATOR(operator/, /)

template <int N>
ACTS_INLINE auto operator-(const SimdType<N>& ACTS_RESTRICT a) {
  SimdType<N> r;
  ACTS_LOOP_IVDEP
  for (auto i = 0ul; i < N; ++i) {
    r[i] = -a[i];
  }
  return r;
}

template <int N>
ACTS_INLINE bool operator==(const SimdType<N>& ACTS_RESTRICT a,
                            const SimdType<N>& ACTS_RESTRICT b) {
  ACTS_LOOP_IVDEP
  for (auto i = 0ul; i < N; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

template <int N>
ACTS_INLINE bool operator!=(const SimdType<N>& ACTS_RESTRICT a,
                            const SimdType<N>& ACTS_RESTRICT b) {
  ACTS_LOOP_IVDEP
  for (auto i = 0ul; i < N; ++i) {
    if (a[i] == b[i]) {
      return false;
    }
  }
  return true;
}

template <int N>
ACTS_INLINE auto hypot(const SimdType<N>& ACTS_RESTRICT a,
                       const SimdType<N>& ACTS_RESTRICT b) {
  SimdType<N> r;
  ACTS_LOOP_IVDEP
  for (auto i = 0ul; i < N; ++i) {
    r[i] = std::hypot(a[i], b[i]);
  }
  return r;
}

template <int N>
ACTS_INLINE auto hypot(const SimdType<N>& ACTS_RESTRICT a, double b) {
  SimdType<N> r;
  ACTS_LOOP_IVDEP
  for (auto i = 0ul; i < N; ++i) {
    r[i] = std::hypot(a[i], b);
  }
  return r;
}

template <int N>
ACTS_INLINE auto hypot(double a, const SimdType<N>& b) {
  SimdType<N> r;
  ACTS_LOOP_IVDEP
  for (auto i = 0ul; i < N; ++i) {
    r[i] = std::hypot(a, b[i]);
  }
  return r;
}

template <int N>
ACTS_INLINE auto sqrt(const SimdType<N>& a) {
  SimdType<N> r;
  ACTS_LOOP_IVDEP
  for (auto i = 0ul; i < N; ++i) {
    r[i] = std::sqrt(a[i]);
  }
  return r;
}

template <int N>
ACTS_INLINE auto sum(const SimdType<N>& a) {
  double r = 0.0;
  ACTS_LOOP_IVDEP
  for (auto i = 0ul; i < N; ++i) {
    r += a[i];
  }
  return r;
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

template <typename T>
ACTS_INLINE auto extract(T& m, int i) {
  constexpr int Rows = std::decay_t<decltype(m)>::RowsAtCompileTime;
  constexpr int Cols = std::decay_t<decltype(m)>::ColsAtCompileTime;
  static_assert(Rows != Eigen::Dynamic && Cols != Eigen::Dynamic);
  constexpr int iStride = std::decay_t<T>::Scalar::Size;
  constexpr int oStride = iStride * Rows;

  if constexpr (std::is_const_v<T>) {
    return Eigen::Map<const Eigen::Matrix<double, Rows, Cols>, Eigen::Unaligned,
                      Eigen::Stride<oStride, iStride>>(m(0, 0).data() + i);
  } else {
    return Eigen::Map<Eigen::Matrix<double, Rows, Cols>, Eigen::Unaligned,
                      Eigen::Stride<oStride, iStride>>(m(0, 0).data() + i);
  }
}

// template <typename A, typename B>
// auto cross(const Eigen::MatrixBase<A>& a, const Eigen::MatrixBase<B>& b) {
//   using ScalarA = typename A::Scalar;
//   using ScalarB = typename B::Scalar;
//
//   static_assert(A::RowsAtCompileTime == 3);
//   static_assert(B::RowsAtCompileTime == 3 && B::ColsAtCompileTime == 1);
//   static_assert(std::is_same_v<ScalarA, ScalarB>);
//
//   // Standard Cross Product
//   if constexpr (std::is_same_v<ScalarA, double>) {
//     return a.cross(b);
//   }
//   // Manual Cross Product for non default-Scalars
//   else if constexpr (A::ColsAtCompileTime == 1) {
//     Eigen::Matrix<ScalarA, 3, 1> ret;
//
//     ret[0] = a[1] * b[2] - a[2] * b[1];
//     ret[1] = a[2] * b[0] - a[0] * b[2];
//     ret[2] = a[0] * b[1] - a[1] * b[0];
//
//     return ret;
//   }
//   // Columnwise Cross Product if A is a 3x3 Matrix
//   else if constexpr (A::ColsAtCompileTime == 3) {
//     Eigen::Matrix<ScalarA, 3, 3> ret;
//
//     ret.col(0) = cross(a.col(0), b);
//     ret.col(1) = cross(a.col(1), b);
//     ret.col(2) = cross(a.col(2), b);
//
//     return ret;
//   }
// }

}  // namespace Acts
