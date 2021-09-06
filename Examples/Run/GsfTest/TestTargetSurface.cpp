
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
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

using namespace Acts::UnitLiterals;

// struct AdvancedInitializer {
//     /// The Surface sequence
//     Acts::DirectNavigator::SurfaceSequence navSurfaces = {};
//     const Acts::Surface *startSurface = nullptr;
//
//     /// Actor result / state
//     struct this_result {
//       bool initialized = false;
//     };
//     using result_type = this_result;
//
//     /// Defaulting the constructor
//     AdvancedInitializer() = default;
//
//     /// Actor operator call
//     /// @tparam statet Type of the full propagator state
//     /// @tparam stepper_t Type of the stepper
//     ///
//     /// @param state the entire propagator state
//     /// @param r the result of this Actor
//     template <typename propagator_state_t, typename stepper_t>
//     void operator()(propagator_state_t& state, const stepper_t& /*unused*/,
//                     result_type& r) const {
//       // Only act once
//       if (not r.initialized) {
//         // Initialize the surface sequence
//         state.navigation.navSurfaces = navSurfaces;
//         state.navigation.navSurfaceIter =
//         state.navigation.navSurfaces.begin(); state.navigation.startSurface =
//         startSurface; r.initialized = true;
//       }
//     }
//
//     /// Actor operator call - resultless, unused
//     template <typename propagator_state_t, typename stepper_t>
//     void operator()(propagator_state_t& /*unused*/,
//                     const stepper_t& /*unused*/) const {}
//   };

struct PrintPositionActor {
  using result_type = int;

  template <typename propagator_state_t, typename stepper_t>
  void operator()(propagator_state_t& state, const stepper_t& stepper,
                  result_type&) const {
    std::cout << "TRACK POSITION: "
              << stepper.position(state.stepping).transpose() << "\n";
    if (state.navigation.currentSurface) {
      std::cout << "SURFACE: " << state.navigation.currentSurface->geometryId()
                << "\n";
    } else {
      std::cout << "NOT ON A SURFACE\n";
    }
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
  const std::vector<double> distances = {0._mm, 100_mm, 200_mm, 300_mm};
  const std::array<double, 2> offsets = {0.0_mm, 0.0_mm};
  const std::array<double, 2> bounds = {100._mm, 100._mm};
  const double thickness = 0.1_mm;
  const auto type = ActsExamples::Telescope::TelescopeSurfaceType::Plane;
  const auto detectorDirection = Acts::BinningValue::binX;

  auto detector = std::shared_ptr(ActsExamples::Telescope::buildDetector(
      detectorContext, detectorElementStorage, distances, offsets, bounds,
      thickness, type, detectorDirection));

  // First surface
  std::shared_ptr<const Acts::Surface> first_surface;

  std::vector<const Acts::Surface*> surfaceSequence;

  detector->visitSurfaces([&](auto surface) {
    if (surface->center(Acts::GeometryContext{})[0] == 0.0) {
      first_surface = surface->getSharedPtr();
    } /*else*/
    { surfaceSequence.push_back(surface); }

    std::cout << "Surface position : "
              << surface->center(Acts::GeometryContext{})
                     .transpose()
                     .template cast<int>()
              << ", normal : "
              << surface->normal(Acts::GeometryContext{}, Acts::Vector2{0, 0})
                     .transpose()
                     .template cast<int>()
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
            << targetSurface->center(Acts::GeometryContext{})
                   .transpose()
                   .template cast<int>()
            << ", normal : "
            << targetSurface
                   ->normal(Acts::GeometryContext{}, Acts::Vector2{0, 0})
                   .transpose()
                   .template cast<int>()
            << "\n";

  // Start pars
  Acts::BoundVector pars = Acts::BoundVector::Zero();
  pars[Acts::eBoundTheta] = 90_degree;
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

  using Actors = Acts::ActionList<PrintPositionActor>;
  using Aborters = Acts::AbortList<>;

  Acts::PropagatorOptions<Actors, Aborters> propOptions(
      Acts::GeometryContext{}, Acts::MagneticFieldContext{},
      Acts::LoggerWrapper{*mainLogger});

  std::cout << "Start propagation\n";
  propagator.propagate(start_params, *targetSurface, propOptions);
}
