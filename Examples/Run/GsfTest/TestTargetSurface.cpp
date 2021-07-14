
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Utilities/PdgParticle.hpp"
#include "ActsExamples/Digitization/DigitizationConfig.hpp"
#include "ActsExamples/EventData/Trajectories.hpp"
#include "ActsExamples/Fatras/FatrasAlgorithm.hpp"
#include "ActsExamples/Framework/RandomNumbers.hpp"
#include "ActsExamples/Framework/Sequencer.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/Generators/EventGenerator.hpp"
#include "ActsExamples/Generators/MultiplicityGenerators.hpp"
#include "ActsExamples/Generators/ParametricParticleGenerator.hpp"
#include "ActsExamples/Generators/VertexGenerators.hpp"
#include "ActsExamples/Io/Csv/CsvPropagationStepsWriter.hpp"
#include "ActsExamples/Io/Csv/CsvSimHitWriter.hpp"
#include "ActsExamples/Io/Csv/CsvTrackingGeometryWriter.hpp"
#include "ActsExamples/Plugins/Obj/ObjPropagationStepsWriter.hpp"
#include "ActsExamples/Plugins/Obj/ObjSpacePointWriter.hpp"
#include "ActsExamples/Plugins/Obj/ObjTrackingGeometryWriter.hpp"
#include "ActsExamples/TelescopeDetector/BuildTelescopeDetector.hpp"
#include "ActsExamples/TrackFitting/TrackFittingAlgorithm.hpp"
#include "ActsExamples/TruthTracking/ParticleSmearing.hpp"
#include "ActsExamples/TruthTracking/TruthTrackFinder.hpp"
#include "ActsExamples/Utilities/Options.hpp"
#include "ActsFatras/EventData/Barcode.hpp"
#include "Acts/Propagator/EigenStepper.hpp"

using namespace Acts::UnitLiterals;

struct TargetSetterActor {
  using result_type = int;

  Acts::Surface* targetSurface;

  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t& state, const stepper_t&,
                  result_type&) const {
    if (!state.navigation.targetSurface)
      state.navigation.targetSurface = targetSurface;
  }
};

struct PrintPositionActor {
  using result_type = int;

  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t& state, const stepper_t& stepper,
                  result_type&) const {
    std::cout << "TRACK POSITION: "
              << stepper.position(state.stepping).transpose() << "\n";
  }
};

int main() {
  // Logger
  auto mainLogger =
      Acts::getDefaultLogger("main logger", Acts::Logging::VERBOSE);

  // no magnetic field so propagate straigt forward
  auto bField =
      std::make_shared<Acts::ConstantBField>(Acts::Vector3(0.0, 0.0, 0.0));

  // Tracking Geometry
  const typename ActsExamples::Telescope::TelescopeDetectorElement::ContextType
      detectorContext;
  std::vector<
      std::shared_ptr<ActsExamples::Telescope::TelescopeDetectorElement>>
      detectorElementStorage;
  const std::vector<double> distances = {0_mm, 100_mm, 200_mm};
  const std::array<double, 2> offsets = {0.0_mm, 0.0_mm};
  const std::array<double, 2> bounds = {100._mm, 100._mm};
  const double thickness = 10._mm;
  const auto type = ActsExamples::Telescope::TelescopeSurfaceType::Plane;
  const auto detectorDirection = Acts::BinningValue::binX;

  auto detector = std::shared_ptr(ActsExamples::Telescope::buildDetector(
      detectorContext, detectorElementStorage, distances, offsets, bounds,
      thickness, type, detectorDirection));

  // First surface
  std::shared_ptr<const Acts::Surface> first_surface;

  detector->visitSurfaces([&](auto surface) {
    if (surface->center(Acts::GeometryContext{})[0] == 0.0) {
      first_surface = surface->getSharedPtr();
    }

    std::cout << "Surface position : "
              << surface->center(Acts::GeometryContext{}).transpose()
              << ", normal : "
              << surface->normal(Acts::GeometryContext{}, Acts::Vector2{0, 0})
                     .transpose()
              << "\n";
  });

  // Target surface
  auto targetBounds = std::make_shared<Acts::RectangleBounds>(1000, 1000);
  Acts::Transform3 trafo =
      Acts::Transform3::Identity() *
      Eigen::AngleAxisd(0.5 * M_PI, Eigen::Vector3d::UnitY());
  trafo.translate(Acts::Vector3::UnitZ() * 150._mm);
  auto targetSurface =
      Acts::Surface::makeShared<Acts::PlaneSurface>(trafo, targetBounds);

  std::cout << "Target position : "
            << targetSurface->center(Acts::GeometryContext{}).transpose()
            << ", normal : "
            << targetSurface
                   ->normal(Acts::GeometryContext{}, Acts::Vector2{0, 0})
                   .transpose()
            << "\n";

  // Start pars
  Acts::BoundVector pars = Acts::BoundVector::Zero();
  pars[Acts::eBoundTheta] = 0.5 * M_PI;
  pars[Acts::eBoundQOverP] = 1.0;

  Acts::BoundTrackParameters start_params(first_surface, pars, std::nullopt);

  // Propagator
  Acts::EigenStepper<> stepper(std::move(bField));
  Acts::Navigator::Config cfg;
  cfg.trackingGeometry = detector;
  cfg.resolvePassive = false;
  cfg.resolveMaterial = true;
  cfg.resolveSensitive = true;
  Acts::Navigator navigator(cfg);
  Acts::Propagator propagator(std::move(stepper), std::move(navigator));

  using Actors = Acts::ActionList<TargetSetterActor, PrintPositionActor>;
  using Aborters =
      Acts::AbortList<Acts::EndOfWorldReached /*, Acts::SurfaceReached*/>;

  Acts::PropagatorOptions<Actors, Aborters> propOptions(
      Acts::GeometryContext{}, Acts::MagneticFieldContext{},
      Acts::LoggerWrapper{*mainLogger});

  propOptions.actionList.get<TargetSetterActor>().targetSurface =
      targetSurface.get();

  std::cout << "Start propagation\n";
  propagator.propagate(start_params, propOptions);
}
