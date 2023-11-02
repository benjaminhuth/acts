// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

namespace Acts {
  

template <int N, typename AA, typename BB, typename CC, typename DD>
  void MultiEigenStepperSIMD<N, AA, BB, CC, DD>::transportCovarianceToBound(
      State& state, const Surface& surface,
      const FreeToBoundCorrection& freeToBoundCorrection) const {
    auto components = componentIterable(state);
    for (auto cmp : components) {
      FreeVector pars = cmp.pars();
      FreeVector derivative = cmp.derivative();
      BoundSquareMatrix cov = cmp.cov();
      BoundSquareMatrix jacobian = cmp.jacobian();
      BoundToFreeMatrix jacToGlobal = cmp.jacToGlobal();
      FreeMatrix jacTransport = cmp.jacTransport();

      detail::transportCovarianceToBound(state.geoContext, cov, jacobian,
                                         jacTransport, derivative, jacToGlobal,
                                         pars, surface, freeToBoundCorrection);

      cmp.pars() = pars;
      cmp.derivative() = derivative;
      cmp.cov() = cov;
      cmp.jacobian() = jacobian;
      cmp.jacToGlobal() = jacToGlobal;
      cmp.jacTransport() = jacTransport;
      cmp.derivative() = derivative;
    }
  }
  
template <int N, typename AA, typename BB, typename CC, typename DD>
  void MultiEigenStepperSIMD<N, AA, BB, CC, DD>::transportCovarianceToCurvilinear(State& state) const {
    auto components = componentIterable(state);
    for (auto cmp : components) {
      FreeVector derivative = cmp.derivative();
      BoundSquareMatrix cov = cmp.cov();
      BoundSquareMatrix jacobian = cmp.jacobian();
      BoundToFreeMatrix jacToGlobal = cmp.jacToGlobal();
      FreeMatrix jacTransport = cmp.jacTransport();

      auto singleStepper = cmp.singleStepper(*this);
      detail::transportCovarianceToCurvilinear(cov, jacobian, jacTransport,
                                               derivative, jacToGlobal,
                                               singleStepper.direction(state));

      cmp.derivative() = derivative;
      cmp.cov() = cov;
      cmp.jacobian() = jacobian;
      cmp.jacToGlobal() = jacToGlobal;
      cmp.jacTransport() = jacTransport;
      cmp.derivative() = derivative;
    }
  }

// Seperated stepsize estimate from Single eigen stepper
/// TODO state should be constant, but then magne
template <int N, typename AA, typename BB, typename CC, typename DD>
template <typename propagator_state_t, typename navigator_t>
Result<double> MultiEigenStepperSIMD<N, AA, BB, CC, DD>::estimate_step_size(
    const propagator_state_t& state, const navigator_t& navigator,
    const Vector3& k1, MagneticFieldProvider::Cache& fieldCache,
    const SingleProxyStepper& stepper, const ConstrainedStep step_size) const {
  double error_estimate = 0.;
  auto current_estimate = step_size;

  // Create SingleExtension locally here
  SingleExtension extension;

  // Don't forget to bid, so that everything works
  extension.validExtensionForStep(state, stepper);

  // If not initialized to zero, we get undefined behaviour
  struct {
    Vector3 B_first = Vector3::Zero();
    Vector3 B_middle = Vector3::Zero();
    Vector3 B_last = Vector3::Zero();
    Vector3 k1 = Vector3::Zero();
    Vector3 k2 = Vector3::Zero();
    Vector3 k3 = Vector3::Zero();
    Vector3 k4 = Vector3::Zero();
    std::array<double, 4> kQoP = {0., 0., 0., 0.};
  } sd;

  sd.k1 = k1;

  const auto pos = stepper.position(state.stepping);
  const auto dir = stepper.direction(state.stepping);

  const auto tryRungeKuttaStep = [&](const ConstrainedStep& h) -> bool {
    // State the square and half of the step size
    const double h2 = h.value() * h.value();
    const double half_h = h.value() * 0.5;

    // Second Runge-Kutta point
    const Vector3 pos1 = pos + half_h * dir + h2 * 0.125 * sd.k1;
    sd.B_middle = *SingleStepper::m_bField->getField(pos1, fieldCache);
    if (!extension.k2(state, stepper, navigator, sd.k2, sd.B_middle, sd.kQoP,
                      half_h, sd.k1)) {
      return false;
    }

    // Third Runge-Kutta point
    if (!extension.k3(state, stepper, navigator, sd.k3, sd.B_middle, sd.kQoP,
                      half_h, sd.k2)) {
      return false;
    }

    // Last Runge-Kutta point
    const Vector3 pos2 = pos + h * dir + h2 * 0.5 * sd.k3;
    sd.B_last = *SingleStepper::m_bField->getField(pos2, fieldCache);
    if (!extension.k4(state, stepper, navigator, sd.k4, sd.B_last, sd.kQoP, h,
                      sd.k3)) {
      return false;
    }

    // Compute and check the local integration error estimate
    error_estimate = std::max(
        h2 * ((sd.k1 - sd.k2 - sd.k3 + sd.k4).template lpNorm<1>() +
              std::abs(sd.kQoP[0] - sd.kQoP[1] - sd.kQoP[2] + sd.kQoP[3])),
        1e-20);

    return (error_estimate <= state.options.tolerance);
  };

  double stepSizeScaling = 1.;
  size_t nStepTrials = 0;
  // Select and adjust the appropriate Runge-Kutta step size as given
  // ATL-SOFT-PUB-2009-001
  while (!tryRungeKuttaStep(current_estimate)) {
    stepSizeScaling =
        std::min(std::max(0.25, std::pow((state.options.tolerance /
                                          std::abs(2. * error_estimate)),
                                         0.25)),
                 4.);

    current_estimate.setValue(current_estimate.value() * stepSizeScaling);

    // If step size becomes too small the particle remains at the initial
    // place
    if (std::abs(current_estimate.value()) <
        std::abs(state.options.stepSizeCutOff)) {
      // Not moving due to too low momentum needs an aborter
      return EigenStepperError::StepSizeStalled;
    }

    // If the parameter is off track too much or given stepSize is not
    // appropriate
    if (nStepTrials > state.options.maxRungeKuttaStepTrials) {
      // Too many trials, have to abort
      return EigenStepperError::StepSizeAdjustmentFailed;
    }
    nStepTrials++;
  }

  return current_estimate;
}

template <int N, typename AA, typename BB, typename CC, typename DD>
template <typename propagator_state_t, typename navigator_t>
Result<double> MultiEigenStepperSIMD<N, AA, BB, CC, DD>::step(
    propagator_state_t& state, const navigator_t& navigator) const {
  auto& sd = state.stepping.stepData;
  auto& stepping = state.stepping;

  const auto pos = multiPosition(stepping);
  const auto dir = multiDirection(stepping);

  // First Runge-Kutta point
  sd.B_first = getMultiField(stepping, pos);
  if (!stepping.extension.validExtensionForStep(state, MultiProxyStepper{},
                                                navigator) ||
      !stepping.extension.k1(state, MultiProxyStepper{}, navigator, sd.k1,
                             sd.B_first, sd.kQoP)) {
    return EigenStepperError::StepInvalid;
  }

  // check for nan
  for (int i = 0; i < 3; ++i) {
    assert(!sd.k1[i].isNaN().any() && "k1 contains nan");
  }

  // Now do stepsize estimate, use the minimum momentum component for this
  // auto estimated_h = [&]() {
  //   Eigen::Index r, c;
  //   multiabsoluteMomentum(stepping).minCoeff(&r, &c);
  //
  //   const Vector3 k1{sd.k1[0][r], sd.k1[1][r], sd.k1[2][r]};
  //   const ConstrainedStep h = stepping.stepSizes[r];
  //
  //   return estimate_step_size(state, navigator, k1, stepping.fieldCache,
  //                             SingleProxyStepper{static_cast<std::size_t>(r)},
  //                             h);
  // }();
  //
  // if (!estimated_h.ok())
  //   return estimated_h.error();

  // Constant stepsize at the moment
  // SimdScalar h = [&]() {
  //   SimdScalar s = SimdScalar::Zero();
  //
  //   for (auto i = 0ul; i < stepping.numComponents; ++i) {
  //     // h = 0 if surface not reachable, effectively suppress any progress
  //     if (stepping.status[i] == Intersection3D::Status::reachable) {
  //       // make sure we get the correct minimal stepsize
  //       s[i] = std::min(*estimated_h,
  //                       static_cast<ActsScalar>(stepping.stepSizes[i]));
  //     }
  //   }
  //
  //   return s;
  // }();
  
  SimdScalar h = 0;
  for(int i=0; i<stepping.stepSizes.size(); ++i) {
    h[i] = stepping.stepSizes[i].value();
  }
  
  std::cout << "Perform step with h=" << h << std::endl;

  // If everything is zero, nothing to do (TODO should this happen?)
  if (SimdHelpers::sum(h) == 0.0) {
    return 0.0;
  }
  
  

  const SimdScalar h2 = h * h;
  const SimdScalar half_h = h * SimdScalar(0.5);

  // Second Runge-Kutta point
  const SimdVector3 pos1 = pos + half_h * dir + h2 * 0.125 * sd.k1;
  sd.B_middle = getMultiField(stepping, pos1);

  if (!stepping.extension.k2(state, MultiProxyStepper{}, navigator, sd.k2,
                             sd.B_middle, sd.kQoP, half_h, sd.k1)) {
    return EigenStepperError::StepInvalid;
  }

  // check for nan
  for (int i = 0; i < 3; ++i) {
    assert(!sd.k2[i].isNaN().any() && "k2 contains nan");
  }

  // Third Runge-Kutta point
  if (!stepping.extension.k3(state, MultiProxyStepper{}, navigator, sd.k3,
                             sd.B_middle, sd.kQoP, half_h, sd.k2)) {
    return EigenStepperError::StepInvalid;
  }

  // check for nan
  for (int i = 0; i < 3; ++i) {
    assert(!sd.k3[i].isNaN().any() && "k3 contains nan");
  }

  // Last Runge-Kutta point
  const SimdVector3 pos2 = pos + h * dir + h2 * 0.5 * sd.k3;
  sd.B_last = getMultiField(stepping, pos2);

  if (!stepping.extension.k4(state, MultiProxyStepper{}, navigator, sd.k4,
                             sd.B_last, sd.kQoP, h, sd.k3)) {
    return EigenStepperError::StepInvalid;
  }

  // check for nan
  for (int i = 0; i < 3; ++i) {
    assert(!sd.k4[i].isNaN().any() && "k4 contains nan");
  }

  // When doing error propagation, update the associated Jacobian matrix
  if (stepping.covTransport) {
    // The step transport matrix in global coordinates
    SimdFreeMatrix D;
    if (!stepping.extension.finalize(state, MultiProxyStepper{}, navigator, h,
                                     D)) {
      return EigenStepperError::StepInvalid;
    }

    // for moment, only update the transport part
    stepping.jacTransport = D * stepping.jacTransport;
  } else {
    if (!stepping.extension.finalize(state, MultiProxyStepper{}, navigator,
                                     h)) {
      return EigenStepperError::StepInvalid;
    }
  }

  // Update the track parameters according to the equations of motion
  stepping.pars.template segment<3>(eFreePos0) +=
      h * dir + h2 / SimdScalar(6.) * (sd.k1 + sd.k2 + sd.k3);
  stepping.pars.template segment<3>(eFreeDir0) +=
      h / SimdScalar(6.) * (sd.k1 + SimdScalar(2.) * (sd.k2 + sd.k3) + sd.k4);

  // Normalize the direction (TODO this can for sure be done smarter...)
  for (auto i = 0ul; i < N; ++i) {
    Vector3 d{stepping.pars[eFreeDir0][i], stepping.pars[eFreeDir1][i],
              stepping.pars[eFreeDir2][i]};

    d.normalize();

    stepping.pars[eFreeDir0][i] = d[0];
    stepping.pars[eFreeDir1][i] = d[1];
    stepping.pars[eFreeDir2][i] = d[2];
  }

  // check for nan
  for (auto i = 0ul; i < eFreeSize; ++i)
    assert(!stepping.pars[i].isNaN().any() && "free parameters contain nan");

  // Compute average step and return
  const auto avg_step = SimdHelpers::sum(h) / N;

  stepping.pathAccumulated += avg_step;
  return avg_step;
}

}  // namespace Acts
