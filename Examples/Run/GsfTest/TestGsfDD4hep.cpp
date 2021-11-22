// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/Utilities/PdgParticle.hpp"
#include "ActsExamples/DD4hepDetector/DD4hepDetector.hpp"
#include "ActsExamples/Digitization/DigitizationConfig.hpp"
#include "ActsExamples/Digitization/DigitizationOptions.hpp"
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
#include "ActsExamples/Geometry/CommonGeometry.hpp"
#include "ActsExamples/Io/Csv/CsvPropagationStepsWriter.hpp"
#include "ActsExamples/Io/Csv/CsvSimHitWriter.hpp"
#include "ActsExamples/Io/Csv/CsvTrackingGeometryWriter.hpp"
#include "ActsExamples/Io/Json/JsonDigitizationConfig.hpp"
#include "ActsExamples/Io/Performance/TrackFitterPerformanceWriter.hpp"
#include "ActsExamples/Io/Root/RootTrajectoryStatesWriter.hpp"
#include "ActsExamples/MagneticField/MagneticFieldOptions.hpp"
#include "ActsExamples/Options/CommonOptions.hpp"
#include "ActsExamples/Plugins/Obj/ObjPropagationStepsWriter.hpp"
#include "ActsExamples/Plugins/Obj/ObjSpacePointWriter.hpp"
#include "ActsExamples/Plugins/Obj/ObjTrackingGeometryWriter.hpp"
#include "ActsExamples/TelescopeDetector/BuildTelescopeDetector.hpp"
#include "ActsExamples/TrackFinding/SpacePointMaker.hpp"
#include "ActsExamples/TrackFinding/SpacePointMakerOptions.hpp"
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
  const auto detector = std::make_shared<DD4hepDetector>();

  // Initialize the options
  boost::program_options::options_description desc;
  {
    using namespace ActsExamples;

    namespace po = boost::program_options;

    auto opt = desc.add_options();
    opt("help", "Show help message");
    opt("n", po::value<int>()->default_value(1),
        "Number of generated particles");
    opt("c", po::value<int>()->default_value(16), "Max number of components");
    opt("s", po::value<std::size_t>()->default_value(123456), "Seed for RNG");
    opt("loglevel", po::value<std::size_t>()->default_value(2),
        "LogLevel for compatibility, with almost no impact");
    opt("log-info", po::bool_switch(), "Use info as default loglevel");
    opt("pars-from-seeds", po::bool_switch(),
        "Use track parameters estimated from truth tracks");
    opt("v", po::bool_switch(), "All algorithms verbose (except the GSF)");
    opt("v-gsf", po::bool_switch(), "GSF algorithm verbose");
    opt("no-kalman", po::bool_switch(), "Disable the GSF");
    opt("no-gsf", po::bool_switch(), "Disable the Kalman Filter");
    opt("do-refit", po::bool_switch(), "Use GSF to refit Kalman result");
    opt("abort-on-error", po::bool_switch(), "Abort GSF on error");
    opt("gsf-no-mat-effects", po::bool_switch(), "Disable material effects in GSF");

    detector->addOptions(desc);
    Options::addGeometryOptions(desc);
    Options::addMaterialOptions(desc);
    Options::addInputOptions(desc);
    Options::addMagneticFieldOptions(desc);
    Options::addDigitizationOptions(desc);
    Options::addSpacePointMakerOptions(desc);
  }

  auto vm = ActsExamples::Options::parse(desc, argc, argv);
  if (vm.empty()) {
    return EXIT_FAILURE;
  }

  GsfTestSettings settings;

  // Override default error log level
  const auto default_log_level =
      vm["log-info"].as<bool>() ? Acts::Logging::INFO : Acts::Logging::ERROR;

  // Read some standard options
  settings.globalLogLevel =
      vm["v"].as<bool>() ? Acts::Logging::VERBOSE : default_log_level;
  settings.gsfLogLevel =
      vm["v-gsf"].as<bool>() ? Acts::Logging::VERBOSE : default_log_level;
  settings.doGsf = not vm["no-gsf"].as<bool>();
  settings.doKalman = not vm["no-kalman"].as<bool>();
  settings.numParticles = vm["n"].as<int>();
  settings.estimateParsFromSeed = vm["pars-from-seeds"].as<bool>();
  settings.maxComponents = vm["c"].as<int>();
  settings.inflation = 1.0;
  settings.maxSteps = 300;
  settings.gsfAbortOnError = false;
  settings.seed = vm["s"].as<std::size_t>();
  settings.gsfAbortOnError = vm["abort-on-error"].as<bool>();
  settings.doRefit = vm["do-refit"].as<bool>();
  settings.gsfLoopProtection = false;
  settings.gsfApplyMaterialEffects = !vm["gsf-no-mat-effects"].as<bool>();

  // Setup detector geometry
  const auto [geometry, decorators] =
      ActsExamples::Geometry::build(vm, *detector, Acts::Logging::INFO);
  settings.geometry = std::shared_ptr<const Acts::TrackingGeometry>(geometry);
  settings.contextDecorators = decorators;

  // Setup the magnetic field
  settings.magneticField = ActsExamples::Options::readMagneticField(vm);

  // Read digitization
  settings.digiConfigFactory = [&]() {
    return ActsExamples::DigitizationConfig(
        vm, ActsExamples::readDigiConfigFromJson(
                vm["digi-config-file"].as<std::string>()));
  };

  // Read space point config
  settings.spmConfig = ActsExamples::Options::readSpacePointMakerConfig(vm);

  // No need to put detector geometry writing in sequencer loop
  settings.objOutputDir = "obj_output";
  export_detector_to_obj(*geometry, settings.objOutputDir);

  // Set start parameters
  settings.phiMin = -180._degree;
  settings.phiMax = 180._degree;
  settings.thetaMin = 45._degree;
  settings.thetaMax = 135._degree;
  settings.pMin = 1.0_GeV;
  settings.pMax = 100.0_GeV;

  return testGsf(settings);
}
