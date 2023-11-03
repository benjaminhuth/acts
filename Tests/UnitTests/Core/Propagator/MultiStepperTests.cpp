// This file is part of the Acts project.
//
// Copyright (C) 2018-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "MultiStepperTests.hpp"

using MultiStepper =
    MultiEigenStepperLoop<StepperExtensionList<DefaultExtension>>;
    
const MultiStepperTester t;

BOOST_AUTO_TEST_SUITE(multistepper_loop_test)

//////////////////////////////////////////////////////
/// Test the construction of the MultiStepper::State
//////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(multi_stepper_state_charged_no_cov) {
  t.test_multi_stepper_state<MultiStepper, false>();
}

BOOST_AUTO_TEST_CASE(multi_stepper_state_neutral_no_cov) {
  t.test_multi_stepper_state<MultiStepper, false>();
}

BOOST_AUTO_TEST_CASE(multi_stepper_state_charged_cov) {
  t.test_multi_stepper_state<MultiStepper, true>();
}

BOOST_AUTO_TEST_CASE(multi_eigen_stepper_state_invalid) {
  t.test_multi_stepper_state_invalid<MultiStepper>();
}

////////////////////////////////////////////////////////////////////////
// Compare the Multi-Stepper against the Eigen-Stepper for consistency
////////////////////////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(multi_eigen_vs_single_eigen) {
  t.test_multi_stepper_vs_eigen_stepper<MultiStepper>();
}

/////////////////////////////
// Test stepsize accessors
/////////////////////////////

// TODO do this later, when we introduce the MultiEigenStepperSIMD, which there
// needs new interfaces...

////////////////////////////////////////////////////
// Test the modifying accessors to the components
////////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(multi_eigen_component_iterable_with_modification) {
  t.test_components_modifying_accessors<MultiStepper>();
}

/////////////////////////////////////////////
// Test if the surface status update works
/////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(test_surface_status_and_cmpwise_bound_state) {
  t.test_multi_stepper_surface_status_update<MultiStepper>();
}

//////////////////////////////////
// Test Bound state computations
//////////////////////////////////
BOOST_AUTO_TEST_CASE(test_component_wise_bound_state) {
  t.test_component_bound_state<MultiStepper>();
}

BOOST_AUTO_TEST_CASE(test_combined_bound_state) {
  t.test_combined_bound_state_function<MultiStepper>();
}

//////////////////////////////////////////////////
// Test the combined curvilinear state function
//////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(test_curvilinear_state) {
  t.test_combined_curvilinear_state_function<MultiStepper>();
}

////////////////////////////////////
// Test single component interface
////////////////////////////////////
BOOST_AUTO_TEST_CASE(test_single_component_interface) {
  t.test_single_component_interface_function<MultiStepper>();
}

//////////////////////////////
// Remove and add components
//////////////////////////////
BOOST_AUTO_TEST_CASE(remove_add_components_test) {
  t.remove_add_components_function<MultiStepper>();
}

//////////////////////////////////////////////////
// Instatiate a Propagator with the MultiStepper
//////////////////////////////////////////////////
BOOST_AUTO_TEST_CASE(propagator_instatiation_test) {
  t.propagator_instatiation_test_function<MultiStepper>();
}

BOOST_AUTO_TEST_SUITE_END()
