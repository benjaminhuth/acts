// This file is part of the Acts project.
//
// Copyright (C) 2018 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/Propagator/detail/Auctioneer.hpp"
#include "Acts/Utilities/detail/Extendable.hpp"
#include "Acts/Utilities/detail/MPL/all_of.hpp"
#include "Acts/Utilities/detail/MPL/has_duplicates.hpp"

namespace Acts {

template <typename... extensions>
struct NewStepperExtensionList : private detail::Extendable<extensions...> {
 private:
  // Checkout for duplicates in the extensions
  static_assert(not detail::has_duplicates_v<extensions...>,
                "same extension type specified several times");

  static constexpr unsigned int nExtensions = sizeof...(extensions);

  static_assert(nExtensions != 0, "no extension type specified");

  // Access to all extensions
  using detail::Extendable<extensions...>::tuple;

  // Vector3 template
  template <typename T>
  using Vector3 = Eigen::Matrix<T, 3, 1>;

  // FreeMatrix template
  template <typename T>
  using FreeMatrix = Eigen::Matrix<T, eFreeSize, eFreeSize>;

  // Vector of valid extensions for a step
  std::array<bool, nExtensions> validExtensions;
  
 public:
  // Access to an extension
  using detail::Extendable<extensions...>::get;

  /// @brief Evaluation function to set valid extensions for an upcoming
  /// integration step
  ///
  /// @tparam propagator_state_t Type of the state of the propagator
  /// @tparam stepper_t Type of the stepper
  /// @param [in] state State of the propagator
  /// @param [in] stepper Stepper of the propagation
  template <typename propagator_state_t, typename stepper_t>
  bool validExtensionForStep(const propagator_state_t& state,
                             const stepper_t& stepper) {
    std::array<int, nExtensions> bids = std::apply(
        [&](const auto&... ext) {
          return std::array<int, nExtensions>{ext.bid(state, stepper)...};
        },
        tuple());

    validExtensions = state.stepping.auctioneer(std::move(bids));

    return (std::find(validExtensions.begin(), validExtensions.end(), true) !=
            validExtensions.end());
  }

  template <typename propagator_state_t, typename stepper_t, typename scalar_t>
  bool k1(const propagator_state_t& state, const stepper_t& stepper,
          Vector3<scalar_t>& knew, const Vector3<scalar_t>& bField,
          std::array<scalar_t, 4>& kQoP) {
    return k(state, stepper, knew, bField, kQoP);
  }

  template <typename propagator_state_t, typename stepper_t, typename scalar_t>
  bool k2(const propagator_state_t& state, const stepper_t& stepper,
          Vector3<scalar_t>& knew, const Vector3<scalar_t>& bField,
          std::array<scalar_t, 4>& kQoP, const scalar_t h = scalar_t(0.),
          const Vector3<scalar_t>& kprev = Vector3<scalar_t>()) {
    return k(state, stepper, knew, bField, kQoP, 1, h, kprev);
  }

  template <typename propagator_state_t, typename stepper_t, typename scalar_t>
  bool k3(const propagator_state_t& state, const stepper_t& stepper,
          Vector3<scalar_t>& knew, const Vector3<scalar_t>& bField,
          std::array<scalar_t, 4>& kQoP, const scalar_t h = scalar_t(0.),
          const Vector3<scalar_t>& kprev = Vector3<scalar_t>()) {
    return k(state, stepper, knew, bField, kQoP, 2, h, kprev);
  }

  template <typename propagator_state_t, typename stepper_t, typename scalar_t>
  bool k4(const propagator_state_t& state, const stepper_t& stepper,
          Vector3<scalar_t>& knew, const Vector3<scalar_t>& bField,
          std::array<scalar_t, 4>& kQoP, const scalar_t h = scalar_t(0.),
          const Vector3<scalar_t>& kprev = Vector3<scalar_t>()) {
    return k(state, stepper, knew, bField, kQoP, 3, h, kprev);
  }

  template <typename propagator_state_t, typename stepper_t,
            typename scalar_t>
  bool k(const propagator_state_t& state, const stepper_t& stepper,
         Vector3<scalar_t>& knew, const Vector3<scalar_t>& bField,
         std::array<scalar_t, 4>& kQoP, const int i = 0, const scalar_t h = scalar_t(0.),
         const Vector3<scalar_t>& kprev = Vector3<scalar_t>()) {
    // TODO replace with integer-templated lambda with C++20
    auto impl = [&](auto intType, auto& impl_ref) {
      constexpr static std::size_t N = decltype(intType)::value;

      if constexpr (N == 0)
        return true;
      else {
        // If element is invalid: continue
        if (!std::get<N - 1>(validExtensions))
          return impl_ref(std::integral_constant<std::size_t, N-1>{}, impl_ref);

        // Continue as long as evaluations are 'true'
        if (std::get<N - 1>(this->tuple())
                .template k(state, stepper, knew, bField, kQoP, i, h, kprev)) {
          return impl_ref(std::integral_constant<std::size_t, N-1>{}, impl_ref);
        } else {
          // Break at false
          return false;
        }
      }
    };

    return impl(std::integral_constant<std::size_t, nExtensions>{}, impl);
  }

  /// @brief This functions broadcasts the call of the method finalize(). It
  /// collects all extensions and arguments and passes them forward for
  /// evaluation and returns a boolean.
  template <typename propagator_state_t, typename stepper_t, typename scalar_t>
  bool finalize(propagator_state_t& state, const stepper_t& stepper,
                const scalar_t h, FreeMatrix<scalar_t>& D) {
    auto impl = [&](auto intType, auto& impl_ref) {
      constexpr static std::size_t N = decltype(intType)::value;

      if constexpr (N == 0)
        return true;
      else {
        // If element is invalid: continue
        if (!std::get<N - 1>(validExtensions))
          return impl_ref(std::integral_constant<std::size_t, N-1>{}, impl_ref);

        // Continue as long as evaluations are 'true'
        if (std::get<N - 1>(this->tuple()).finalize(state, stepper, h, D)) {
          return impl_ref(std::integral_constant<std::size_t, N-1>{}, impl_ref);
        } else {
          // Break at false
          return false;
        }
      }
    };

    return impl(std::integral_constant<std::size_t, nExtensions>{}, impl);
  }

  /// @brief This functions broadcasts the call of the method finalize(). It
  /// collects all extensions and arguments and passes them forward for
  /// evaluation and returns a boolean.
  template <typename propagator_state_t, typename stepper_t, typename scalar_t>
  bool finalize(propagator_state_t& state, const stepper_t& stepper,
                const scalar_t h) {
    auto impl = [&](auto intType, auto& impl_ref) {
      constexpr static std::size_t N = decltype(intType)::value;

      if constexpr (N == 0)
        return true;
      else {
        // If element is invalid: continue
        if (!std::get<N - 1>(validExtensions))
          return impl_ref(std::integral_constant<std::size_t, N-1>{}, impl_ref);

        // Continue as long as evaluations are 'true'
        if (std::get<N - 1>(this->tuple()).finalize(state, stepper, h)) {
          return impl_ref(std::integral_constant<std::size_t, N-1>{}, impl_ref);
        } else {
          // Break at false
          return false;
        }
      }
    };

    return impl(std::integral_constant<int, nExtensions>{}, impl);
  }
};

}  // namespace Acts
