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

#include "Acts/Propagator/MultiEigenStepperSIMD.hpp"

#include "MultiStepperTests.hpp"

using SimdExtension = Acts::detail::GenericDefaultExtension<Acts::SimdType<4>>;

using MultiStepper =
    MultiEigenStepperSIMD<4, StepperExtensionList<SimdExtension>>;

BOOST_AUTO_TEST_SUITE(multistepper_simd_test)

BOOST_AUTO_TEST_CASE(simd_cross_product_test) {
  using Vector3 = Eigen::Matrix<Acts::SimdType<4>, 1, 3>;

  Vector3 a = Vector3::Ones();
  Vector3 b = Vector3::Ones();
  Vector3 c = a.cross(b);
  BOOST_CHECK(std::isfinite(c[0][0]));
}

//////////////////////////////////////////////////////
/// Test the construction of the MultiStepper::State
//////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(multi_stepper_state_charged_no_cov) {
  test_multi_stepper_state<MultiStepper, false>();
}

BOOST_AUTO_TEST_CASE(multi_stepper_state_neutral_no_cov) {
  test_multi_stepper_state<MultiStepper, false>();
}

BOOST_AUTO_TEST_CASE(multi_eigen_stepper_state_invalid) {
  test_multi_stepper_state_invalid<MultiStepper>();
}

////////////////////////////////////////////////////////////////////////
// Compare the Multi-Stepper against the Eigen-Stepper for consistency
////////////////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(multi_eigen_vs_single_eigen) {
  test_multi_stepper_vs_eigen_stepper<MultiStepper>();
}

/////////////////////////////
// Test stepsize accessors
/////////////////////////////

// TODO do this later, when we introduce the MultiEigenStepperSIMD, which there
// needs new interfaces...

//////////////////////////////////////////////////
// Test the modifying accessors to the components
//////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(multi_eigen_component_iterable_with_modification) {
  test_components_modifying_accessors<MultiStepper>();
}

/////////////////////////////////////////////
// Test if the surface status update works
/////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(test_surface_status_and_cmpwise_bound_state) {
  test_multi_stepper_surface_status_update<MultiStepper>();
}

//////////////////////////////////
// Test Bound state computations
//////////////////////////////////
BOOST_AUTO_TEST_CASE(test_component_wise_bound_state) {
  test_component_bound_state<MultiStepper>();
}

BOOST_AUTO_TEST_CASE(test_combined_bound_state) {
  test_combined_bound_state_function<MultiStepper>();
}

//////////////////////////////////////////////////
// Test the combined curvilinear state function
//////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(test_curvilinear_state) {
  test_combined_curvilinear_state_function<MultiStepper>();
}

////////////////////////////////////
// Test single component interface
////////////////////////////////////
BOOST_AUTO_TEST_CASE(test_single_component_interface) {
  test_single_component_interface_function<MultiStepper>();
}

//////////////////////////////
// Remove and add components
//////////////////////////////
// BOOST_AUTO_TEST_CASE(remove_add_components_test) {
//   remove_add_components_function<MultiStepper>();
// }

//////////////////////////////////////////////////
// Instatiate a Propagator with the MultiStepper
//////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(propagator_instatiation_test) {
  propagator_instatiation_test_function<MultiStepper>();
}

BOOST_AUTO_TEST_SUITE_END()
