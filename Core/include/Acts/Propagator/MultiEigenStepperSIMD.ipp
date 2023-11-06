// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Propagator/detail/MultiStepperUtils.hpp"
#include "Acts/Utilities/Zip.hpp"

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
void MultiEigenStepperSIMD<N, AA, BB, CC, DD>::transportCovarianceToCurvilinear(
    State& state) const {
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

template <int N, typename AA, typename BB, typename CC, typename DD>
Intersection3D::Status
MultiEigenStepperSIMD<N, AA, BB, CC, DD>::updateSurfaceStatus(
    State& state, const Surface& surface, std::uint8_t index, Direction navDir,
    const BoundaryCheck& bcheck, const Logger& /*l*/) const {
  std::array<int, 4> counts = {0, 0, 0, 0};

  auto components = componentIterable(state);
  for (auto [cmp, stepSize] : Acts::zip(components, state.stepSizes)) {
    // TODO part of the hack: set the stepsize to the correct value before
    // calling updateSingleSurfaceStatus
    state.stepSize = stepSize;

    const auto prevStatus = cmp.status();

    cmp.status() = detail::updateSingleSurfaceStatus<SingleProxyStepper>(
        cmp.singleStepper(*this), state, surface, index, navDir, bcheck,
        s_onSurfaceTolerance, logger() /*Acts::getDummyLogger()*/);

    ACTS_VERBOSE("  cmp" << cmp.index() << ": " << prevStatus << " -> "
                         << cmp.status());

    ++counts[static_cast<std::size_t>(cmp.status())];
  }

  // This is a 'any_of' criterium. As long as any of the components has a
  // certain state, this determines the total state (in the order of a
  // somewhat importance)
  using Status = Intersection3D::Status;

  if (counts[static_cast<std::size_t>(Status::reachable)] > 0)
    return Status::reachable;
  else if (counts[static_cast<std::size_t>(Status::onSurface)] > 0)
    return Status::onSurface;
  else if (counts[static_cast<std::size_t>(Status::unreachable)] > 0)
    return Status::unreachable;
  else
    return Status::missed;
}

template <int N, typename AA, typename BB, typename CC, typename DD>
auto MultiEigenStepperSIMD<N, AA, BB, CC, DD>::boundState(
    State& state, const Surface& surface, bool transportCov,
    const FreeToBoundCorrection& freeToBoundCorrection) const
    -> Result<BoundState> {
  return detail::multiComponentBoundState(*this, state, surface, transportCov,
                                          freeToBoundCorrection);
}

template <int N, typename AA, typename BB, typename CC, typename DD>
auto MultiEigenStepperSIMD<N, AA, BB, CC, DD>::curvilinearState(
    State& state, bool transportCov) const -> CurvilinearState {
  return detail::multiComponentCurvilinearState(*this, state, transportCov);
}

#if 0
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

    current_estimate.update(current_estimate.value() * stepSizeScaling, ConstrainedStep::actor);

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
#endif

template <int N, typename AA, typename BB, typename CC, typename DD>
template <typename propagator_state_t, typename navigator_t>
Result<double> MultiEigenStepperSIMD<N, AA, BB, CC, DD>::step(
    propagator_state_t& state, const navigator_t& navigator) const {
  auto& sd = state.stepping.stepData;
  auto& stepping = state.stepping;

  const auto pos = multiPosition(stepping);
  const auto dir = multiDirection(stepping);

  // TODO
  // - make this configurable
  // - check how we can do stepsize estimation...
  SimdScalar h = SimdScalar{30.0};
  for (auto i = 0ul; i < stepping.stepSizes.size(); ++i) {
    h[i] = std::min(static_cast<double>(h[i]), stepping.stepSizes[i].value());
  }
  ACTS_VERBOSE("Perform step with h=" << h);

  ACTS_VERBOSE("Pos before step:");
  ACTS_VERBOSE("  x = " << pos[0]);
  ACTS_VERBOSE("  y = " << pos[1]);
  ACTS_VERBOSE("  z = " << pos[2]);

  ACTS_VERBOSE("Dir before step:");
  ACTS_VERBOSE("  dx = " << dir[0]);
  ACTS_VERBOSE("  dy = " << dir[1]);
  ACTS_VERBOSE("  dz = " << dir[2]);

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

  // If everything is zero, nothing to do (TODO should this happen?)
  if (sum(h) == 0.0) {
    return 0.0;
  }

  const SimdScalar h2 = h * h;
  const SimdScalar half_h = h * SimdScalar(0.5);

  // Second Runge-Kutta point
  const SimdVector3 pos1 = pos + half_h * dir + h2 * SimdScalar(0.125) * sd.k1;
  sd.B_middle = getMultiField(stepping, pos1);

  ACTS_VERBOSE("Runge-Kutta k1:");
  ACTS_VERBOSE("  x = " << sd.k1[0]);
  ACTS_VERBOSE("  y = " << sd.k1[1]);
  ACTS_VERBOSE("  z = " << sd.k1[2]);

  if (!stepping.extension.k2(state, MultiProxyStepper{}, navigator, sd.k2,
                             sd.B_middle, sd.kQoP, half_h, sd.k1)) {
    return EigenStepperError::StepInvalid;
  }

  ACTS_VERBOSE("Runge-Kutta k2:");
  ACTS_VERBOSE("  x = " << sd.k2[0]);
  ACTS_VERBOSE("  y = " << sd.k2[1]);
  ACTS_VERBOSE("  z = " << sd.k2[2]);

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

  ACTS_VERBOSE("Runge-Kutta k3:");
  ACTS_VERBOSE("  x = " << sd.k3[0]);
  ACTS_VERBOSE("  y = " << sd.k3[1]);
  ACTS_VERBOSE("  z = " << sd.k3[2]);

  if (!stepping.extension.k4(state, MultiProxyStepper{}, navigator, sd.k4,
                             sd.B_last, sd.kQoP, h, sd.k3)) {
    return EigenStepperError::StepInvalid;
  }

  ACTS_VERBOSE("Runge-Kutta k4:");
  ACTS_VERBOSE("  x = " << sd.k4[0]);
  ACTS_VERBOSE("  y = " << sd.k4[1]);
  ACTS_VERBOSE("  z = " << sd.k4[2]);

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

  // .normalize() does not work for some reason
  SimdScalar dirNorm = stepping.pars.template segment<3>(eFreeDir0).norm();
  stepping.pars.template segment<3>(eFreeDir0) /= dirNorm;
  // stepping.pars.template segment<3>(eFreeDir0).normalize();

  ACTS_VERBOSE("Pos after step:");
  ACTS_VERBOSE("  x = " << multiPosition(stepping)[0]);
  ACTS_VERBOSE("  y = " << multiPosition(stepping)[1]);
  ACTS_VERBOSE("  z = " << multiPosition(stepping)[2]);

  // check for nan
  for (auto i = 0ul; i < eFreeSize; ++i) {
    assert(!stepping.pars[i].isNaN().any() && "free parameters contain nan");
  }

  // Compute average step and return
  const auto avgStep = sum(h) / N;

  stepping.pathAccumulated += avgStep;
  return avgStep;
}

}  // namespace Acts
