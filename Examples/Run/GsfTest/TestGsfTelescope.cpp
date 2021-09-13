// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/Utilities/PdgParticle.hpp"
#include "ActsExamples/Digitization/DigitizationConfig.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/Trajectories.hpp"
#include "ActsExamples/Fatras/FatrasSimulation.hpp"
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
#include "ActsExamples/Io/Performance/TrackFitterPerformanceWriter.hpp"
#include "ActsExamples/Io/Root/RootTrajectoryStatesWriter.hpp"
#include "ActsExamples/Plugins/Obj/ObjPropagationStepsWriter.hpp"
#include "ActsExamples/Plugins/Obj/ObjSpacePointWriter.hpp"
#include "ActsExamples/Plugins/Obj/ObjTrackingGeometryWriter.hpp"
#include "ActsExamples/TelescopeDetector/BuildTelescopeDetector.hpp"
#include "ActsExamples/TrackFinding/SpacePointMaker.hpp"
#include "ActsExamples/TrackFinding/TrackParamsEstimationAlgorithm.hpp"
#include "ActsExamples/TrackFitting/TrackFittingAlgorithm.hpp"
#include "ActsExamples/TruthTracking/ParticleSmearing.hpp"
#include "ActsExamples/TruthTracking/TruthTrackFinder.hpp"
#include "ActsExamples/Utilities/Options.hpp"
#include "ActsFatras/EventData/Barcode.hpp"

#include <chrono>
#include <iostream>
#include <random>

#include "AlgorithmsAndWriters/ObjHitWriter.hpp"
#include "AlgorithmsAndWriters/ParameterEstimationPerformanceWriter.hpp"
#include "AlgorithmsAndWriters/ProtoTrackLengthSelector.hpp"
#include "AlgorithmsAndWriters/SeedsFromProtoTracks.hpp"
#include "AlgorithmsAndWriters/TrackFittingPerformanceWriterCsv.hpp"
#include "GsfAlgorithmFunction.hpp"
#include "TestGsfGeneric.hpp"
#include "TestHelpers.hpp"

using namespace Acts::UnitLiterals;

int main(int argc, char **argv) {
  const std::vector<std::string> args(argv, argv + argc);

  if (std::find(begin(args), end(args), "--help") != args.end()) {
    std::cout << "Usage: " << args[0] << " <options>\n";
    std::cout << "Options:\n";
    std::cout << "\t --bf-value <val>   \t"
              << "Magnetic field value (in tesla)\n";
    std::cout << "\t -n <val>           \t"
              << "Number of simulated particles (default: 1)\n";
    std::cout << "\t -s <val>           \t"
              << "Seed for the RNG (default: std::random_device{}()\n";
    std::cout << "\t -c <val>           \t"
              << "Max number of GSF components (default: 4)\n";
    std::cout << "\t --no-gsf           \t"
              << "Disable the GSF\n";
    std::cout << "\t --no-kalman        \t"
              << "Disable the Kalman Filter\n";
    std::cout << "\t -v-gsf             \t"
              << "GSF algorithm verbose\n";
    std::cout << "\t -v                 \t"
              << "All algorithms verbose (except the GSF)\n";
    std::cout << "\t --gsf-abort-error  \t"
              << "Call std::abort on some GSF errors\n";
    std::cout << "\t --pars-from-seed   \t"
              << "Estimate the start parameters from seeds\n";
    std::cout << "\t --std-navigator    \t"
              << "Use the standard navigator instead of the DirectNavigator\n";
    std::cout << "\t --inflate-cov <val>\t"
              << "Inflate the covariance of esimated start parameters "
                 "(default: 1.0)\n";
    std::cout << "\t --help             \t"
              << "Print the help message\n";
    return EXIT_FAILURE;
  }

  GsfTestSettings settings;

  settings.globalLogLevel = [&]() {
    const bool found = std::find(begin(args), end(args), "-v") != args.end();
    return found ? Acts::Logging::VERBOSE : Acts::Logging::INFO;
  }();

  settings.gsfLogLevel = [&]() {
    const bool found =
        std::find(begin(args), end(args), "-v-gsf") != args.end();
    return found ? Acts::Logging::VERBOSE : Acts::Logging::INFO;
  }();

  settings.numParticles = [&]() {
    const auto found = std::find(begin(args), end(args), "-n");
    const bool valid = found != end(args) && std::next(found) != end(args);
    return valid ? std::stoi(*std::next(found)) : 1;
  }();

  settings.magneticField = [&]() {
    const auto found = std::find(begin(args), end(args), "--bf-value");
    if (found != args.end() && std::next(found) != args.end()) {
      return std::make_shared<Acts::ConstantBField>(
          Acts::Vector3{0, 0, std::stod(*std::next(found))});
    }
    return std::make_shared<Acts::ConstantBField>(Acts::Vector3{0, 0, 2.0_T});
  }();

  settings.seed = [&]() -> uint64_t {
    const auto found = std::find(begin(args), end(args), "-s");
    if (found != args.end() && std::next(found) != args.end()) {
      return std::stoul(*std::next(found));
    }
    return std::random_device{}();
  }();

  settings.gsfAbortOnError =
      (std::find(begin(args), end(args), "--gsf-abort-error") != args.end());

  settings.maxComponents = [&]() {
    if (auto found = std::find(begin(args), end(args), "-c");
        found != args.end() && std::next(found) != args.end()) {
      return std::stoi(*std::next(found));
    }
    return 4;
  }();

  settings.doDirectNavigation =
      std::find(begin(args), end(args), "--std-navigator") == args.end();
  settings.doGsf = std::find(begin(args), end(args), "--no-gsf") == args.end();
  settings.doKalman =
      std::find(begin(args), end(args), "--no-kalman") == args.end();

  settings.estimateParsFromSeed =
      std::find(begin(args), end(args), "--pars-from-seed") != args.end();

  settings.inflation = [&]() -> double {
    const auto found = std::find(begin(args), end(args), "--inflate-cov");
    if (found != args.end() && std::next(found) != args.end()) {
      return std::stod(*std::next(found));
    }
    return 1.0;
  }();

  // Export the seed for reproducibility
  {
    std::ofstream seedFile("seed.txt", std::ios_base::trunc);
    seedFile << settings.seed;
  }

  // Make the telescope detector
  const typename ActsExamples::Telescope::TelescopeDetectorElement::ContextType
      detectorContext;
  std::vector<
      std::shared_ptr<ActsExamples::Telescope::TelescopeDetectorElement>>
      detectorElementStorage;
  // we add a surface at the opposite site, to extend the volume etc
  const std::vector<double> distances = {-100_mm, 100_mm, 200_mm,
                                         300_mm,  400_mm, 500_mm};
  const std::array<double, 2> offsets = {0.0_mm, 0.0_mm};
  const std::array<double, 2> bounds = {100._mm, 100._mm};
  const double thickness = 1._mm;
  const auto type = ActsExamples::Telescope::TelescopeSurfaceType::Plane;
  const auto detectorDirection = Acts::BinningValue::binX;

  settings.geometry = std::shared_ptr(ActsExamples::Telescope::buildDetector(
      detectorContext, detectorElementStorage, distances, offsets, bounds,
      thickness, type, detectorDirection));

  // No need to put detector geometry writing in sequencer loop
#if 0
  export_detector_to_obj(*detector);
#endif

  // Digi config
  settings.digiConfigFactory = [&]() {
    //     std::shared_ptr<const Acts::Surface> some_surface;
    //     settings.geometry->visitSurfaces([&](auto surface) {
    //       if (surface->center(Acts::GeometryContext{})[0] == 100.0) {
    //         some_surface = surface->getSharedPtr();
    //       }
    //     });
    //
    //     throw_assert(some_surface, "no valid surface found");

    //     const auto volume_id =
    //         settings.geometry
    //             ->lowestTrackingVolume(
    //                 Acts::GeometryContext{},
    //                 some_surface->center(Acts::GeometryContext{}))
    //             ->geometryId()
    //             .volume();

    using namespace ActsExamples::Options;

    const bool doMerge = false;
    const bool mergeNsigma = false;
    const bool mergeCommonCorner = false;
    const std::vector<int> volumes = {static_cast<int>(1)};
    const std::vector<VariableIntegers> indices = {
        {std::vector<int>{0, 1}}};  // loc0, loc1
    const std::vector<VariableIntegers> types = {
        {std::vector<int>{0, 0}}};  // gauss, gauss
    const std::vector<VariableReals> parameters = {
        {std::vector<double>{10._mm, 10._mm}}};  // gaussian width

    return ActsExamples::DigitizationConfig(
        doMerge, mergeNsigma, mergeCommonCorner, volumes, indices, types,
        parameters,
        Acts::GeometryHierarchyMap<ActsExamples::DigiComponentsConfig>());
  };

  // Space point maker config
  ActsExamples::SpacePointMaker::Config spm_cfg;
  spm_cfg.geometrySelection = {
      settings.geometry->highestTrackingVolume()->geometryId()};
  settings.spmConfig = spm_cfg;

  // Set start parameters
  settings.phiMin = -5._degree;
  settings.phiMax = 5._degree;
  settings.thetaMin = 85._degree;
  settings.thetaMax = 95._degree;
  settings.pMin = 1.0_GeV;
  settings.pMax = 10.0_GeV;

  // Run test
  testGsf(settings);
}
