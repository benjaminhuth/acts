// This file is part of the Acts project.
//
// Copyright (C) 2018-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <experimental/simd>

// TODO where to put this so it is seen by Eigen???
template <typename T, typename Abi>
bool operator||(bool a, const std::experimental::simd_mask<T, Abi> &s) {
  bool r = true;
  for (std::size_t i = 0; i < s.size(); ++i) {
    r = r && (a || s[i]);
  }
  return r;
}

#include <Eigen/Dense>

template <typename Derived>
bool operator||(bool a, const Eigen::DenseBase<Derived> &s) {
  bool r = true;
  for (std::size_t i = 0; i < s.size(); ++i) {
    r = r && (a || s[i]);
  }
  return r;
}

#include "Acts/Propagator/MultiEigenStepperSIMD.hpp"

#include "MultiStepperTests.hpp"

using SimdExtension = Acts::detail::GenericDefaultExtension<Acts::SimdType<4>>;

using MultiStepper =
    MultiEigenStepperSIMD<4, StepperExtensionList<SimdExtension>>;

double epsilon = 1.e-1;
const MultiStepperTester tester(epsilon);

BOOST_AUTO_TEST_SUITE(multistepper_simd_test)

BOOST_AUTO_TEST_CASE(simd_cross_product_test) {
  using Vector3 = Eigen::Matrix<Acts::SimdType<4>, 1, 3>;

  Vector3 a = Vector3::Ones();
  Vector3 b = Vector3::Ones();
  Vector3 c = a.cross(b);
  BOOST_CHECK(std::isfinite(c[0][0]));
}

BOOST_AUTO_TEST_CASE(check_zero_ones) {
  using Vector3 = Eigen::Matrix<Acts::SimdType<4>, 1, 3>;

  Vector3 ones = Vector3::Ones();
  Vector3 zero = Vector3::Zero();

  for (auto vi = 0ul; vi < 3; ++vi) {
    for (auto si = 0ul; si < 4; ++si) {
      BOOST_CHECK_CLOSE(static_cast<double>(zero[vi][si]), 0.0, epsilon);
      BOOST_CHECK_CLOSE(static_cast<double>(ones[vi][si]), 1.0, epsilon);
    }
  }
}

//////////////////////////////////////////////////////
/// Test the construction of the MultiStepper::State
//////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(multi_stepper_state_charged_no_cov) {
  tester.test_multi_stepper_state<MultiStepper, false>();
}

BOOST_AUTO_TEST_CASE(multi_stepper_state_charged_cov) {
  tester.test_multi_stepper_state<MultiStepper, true>();
}

BOOST_AUTO_TEST_CASE(multi_stepper_state_neutral_no_cov) {
  tester.test_multi_stepper_state<MultiStepper, false>();
}

BOOST_AUTO_TEST_CASE(multi_eigen_stepper_state_invalid) {
  tester.test_multi_stepper_state_invalid<MultiStepper>();
}

// This check does not work since the stepsize estimation is not implmented
// (yet) BOOST_AUTO_TEST_CASE(multi_eigen_vs_single_eigen) {
//   tester.test_multi_stepper_vs_eigen_stepper<MultiStepper>();
// }

//////////////////////////////////////////////////
// Test the modifying accessors to the components
//////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(multi_eigen_component_iterable_with_modification) {
  tester.test_components_modifying_accessors<MultiStepper>();
}

/////////////////////////////////////////////
// Test if the surface status update works
/////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(test_surface_status_and_cmpwise_bound_state) {
  tester.test_multi_stepper_surface_status_update<MultiStepper>();
}

//////////////////////////////////
// Test Bound state computations
//////////////////////////////////
BOOST_AUTO_TEST_CASE(test_component_wise_bound_state) {
  tester.test_component_bound_state<MultiStepper>();
}

BOOST_AUTO_TEST_CASE(test_combined_bound_state) {
  tester.test_combined_bound_state_function<MultiStepper>();
}

//////////////////////////////////////////////////
// Test the combined curvilinear state function
//////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(test_curvilinear_state) {
  tester.test_combined_curvilinear_state_function<MultiStepper>();
}

////////////////////////////////////
// Test single component interface
////////////////////////////////////
BOOST_AUTO_TEST_CASE(test_single_component_interface) {
  tester.test_single_component_interface_function<MultiStepper>();
}

//////////////////////////////
// Remove and add components
//////////////////////////////
BOOST_AUTO_TEST_CASE(remove_add_components_test) {
  tester.remove_add_components_function<MultiStepper>();
}

//////////////////////////////////////////////////
// Instatiate a Propagator with the MultiStepper
//////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(propagator_instatiation_test) {
  tester.propagator_instatiation_test_function<MultiStepper>();
}

BOOST_AUTO_TEST_SUITE_END()
