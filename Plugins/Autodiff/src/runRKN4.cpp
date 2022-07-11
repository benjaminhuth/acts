#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Plugins/Autodiff/AutodiffExtensionWrapper.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/detail/GenericDefaultExtension.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"

using namespace Acts;
using namespace Acts::UnitLiterals;

struct StepData {
  Vector3 B_first, B_middle, B_last;
  // Vector3 k1, k2, k3, k4;
  // std::array<double, 4> kQoP;
};

struct Options {
  double mass = 1.0;
  double stepSizeCutOff = 0.0;
  double tolerance = 1e-4;
  double maxRungeKuttaStepTrials = 1.0;
};

struct NavigationState {
  // no entry so far
};

using Scalar = double;
using Extension = detail::GenericDefaultExtension<double>;

const double step_size = 1.0;
const Vector3 direction = {1, 0, 0};
const Vector3 bfield_vec = {0, 0, 2_T};

void run_rkn4_from_autodiff_header() {
  // Initial params
  FreeVector pars = FreeVector::Zero();
  pars.segment<4>(eFreePos0) = Vector4::Zero();
  pars.segment<3>(eFreeDir0) = direction;
  pars(eFreeQOverP) = 1 / 1_GeV;  // charge divided by momentum

  // Just use a 2T B-field in z-direction here
  StepData stepData;
  stepData.B_first = bfield_vec;
  stepData.B_middle = bfield_vec;
  stepData.B_last = bfield_vec;

  // Create dummy states
  Options opts;
  NavigationState navState;
  detail::AutodiffFakeStepperState<Scalar> stepperState;

  detail::AutodiffFakePropState<Scalar, Options, NavigationState> propState{
      stepperState, opts, navState};

  // this function needs to be differentiated
  std::cout << "pars beginning: " << pars.transpose() << "\n";
  const auto out =
      detail::rkn4Step<Scalar, Extension>(pars, stepData, propState, step_size);
  std::cout << "pars after:     " << out.transpose() << "\n";
}

void run_eigen_stepper_step() {
  // Setup magnetic field & Stepper
  const std::shared_ptr<Acts::MagneticFieldProvider> bfield =
      std::make_shared<Acts::ConstantBField>(bfield_vec);
  const EigenStepper stepper(bfield);

  // Setup initial parameters bound to a dummy surface
  const auto surface =
      Surface::makeShared<PlaneSurface>(Vector3::Zero(), direction);
  BoundVector pars = BoundVector::Zero();
  pars(eBoundQOverP) = 1 / 1_GeV;
  pars(eBoundTheta) = M_PI_2;
  //   pars(eBoundPhi) = M_PI_2;

  // Setup the propagator state
  Options opts;
  NavigationState navState;
  auto stepperState = decltype(stepper)::State(
      GeometryContext{}, bfield->makeCache(MagneticFieldContext{}),
      BoundTrackParameters(surface, pars, BoundSymMatrix::Identity()),
      NavigationDirection::Forward, step_size);

  struct PropState {
    decltype(stepper)::State stepping;
    Options options;
    NavigationState navigation;
  };

  PropState propState{std::move(stepperState), opts, navState};

  // Do the step
  std::cout << "pars beginning: "
            << propState.stepping.pars.transpose().cast<float>() << "\n";
  stepper.step(propState);
  std::cout << "pars after:     "
            << propState.stepping.pars.transpose().cast<float>() << "\n";
  std::cout << "path accumulated: " << propState.stepping.pathAccumulated
            << "\n";

  // This is the hand-computed Jacobian (if you do only one step)
  [[maybe_unused]] const auto D = propState.stepping.jacTransport;
}

int main() {
  std::cout << std::setprecision(5) << std::fixed;
  std::cout << "Default Stepper:\n";
  run_eigen_stepper_step();
  std::cout << "\n";

  std::cout << "Autodiff function:\n";
  run_rkn4_from_autodiff_header();
}
