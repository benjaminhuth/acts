// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/Utilities/PdgParticle.hpp"
#include "ActsExamples/TGeoDetector/TGeoDetector.hpp"
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
// #include "ActsExamples/Io/Csv/CsvPropagationStepsWriter.hpp"
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
#include <filesystem>

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
  const auto detector = std::make_shared<ActsExamples::TGeoDetector>();

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
    opt("gsf-no-mat-effects", po::bool_switch(),
        "Disable material effects in GSF");
    opt("stepper-max-mom", po::bool_switch(),
        "The stepper uses the max-momentum component as interface for the "
        "Navigator");
    opt("bethe-heitler-low-x0-file", po::value<std::string>()->default_value(""), "Path to low x0 bethe-heitler-description");
    opt("bethe-heitler-high-x0-file", po::value<std::string>()->default_value(""), "Path to high x0 bethe-heitler-description");

    opt("gen-vertex-xy-std-mm", po::value<double>()->default_value(0.0),
      "Transverse vertex standard deviation in mm");
    opt("gen-vertex-z-std-mm", po::value<double>()->default_value(0.0),
      "Longitudinal vertex standard deviation in mm");
    opt("gen-vertex-t-std-ns", po::value<double>()->default_value(0.0),
      "Temporal vertex standard deviation in ns");
    opt("gen-phi-degree", po::value<ActsExamples::Options::Interval>()->value_name("MIN:MAX")->default_value({-180.0, 180.0}),
      "Transverse direction angle generation range in degree");
    opt("gen-eta", po::value<ActsExamples::Options::Interval>()->value_name("MIN:MAX")->default_value({-4.0, 4.0}),
      "Pseudo-rapidity generation range");
    opt("gen-eta-uniform", po::bool_switch(),
      "Sample eta directly and not cos(theta).");
    opt("gen-mom-gev", po::value<ActsExamples::Options::Interval>()->value_name("MIN:MAX")->default_value({1.0, 10.0}),
      "Absolute (or transverse) momentum generation range in GeV");
    opt("gen-mom-transverse", po::bool_switch(),
      "Momentum refers to transverse momentum");
    opt("gen-pdg", po::value<int32_t>()->default_value(Acts::PdgParticle::eElectron),
      "PDG number of the particle, will be adjusted for charge flip.");

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
  settings.maxSteps = 1000;
  settings.gsfAbortOnError = false;
  settings.seed = vm["s"].as<std::size_t>();
  settings.gsfAbortOnError = vm["abort-on-error"].as<bool>();
  settings.doRefit = vm["do-refit"].as<bool>();
  settings.doDirectNavigation = false;
  settings.gsfLoopProtection = false;
  settings.gsfApplyMaterialEffects = !vm["gsf-no-mat-effects"].as<bool>();
  settings.stepperInterface = vm["stepper-max-mom"].as<bool>()
                                  ? StepperInteface::maxMomentum
                                  : StepperInteface::average;
  settings.gsfBetheHeitlerHighX0Path = vm["bethe-heitler-high-x0-file"].as<std::string>();
  settings.gsfBetheHeitlerLowX0Path = vm["bethe-heitler-low-x0-file"].as<std::string>();

  // Setup detector geometry
  const auto [geometry, decorators] =
      ActsExamples::Geometry::build(vm, *detector/*, Acts::Logging::INFO*/);
  settings.geometry = std::shared_ptr<const Acts::TrackingGeometry>(geometry);
  settings.contextDecorators = decorators;

  // Setup the magnetic field
  settings.magneticField = ActsExamples::Options::readMagneticField(vm);

  // Read digitization
  settings.digiConfigFactory = [&]() {
    const auto filename = vm["digi-config-file"].as<std::string>();

    if( !std::filesystem::exists(filename) ) {
        throw std::runtime_error("File '" + filename + "' seems not to exist");
    }

    return ActsExamples::DigitizationConfig(
        vm, ActsExamples::readDigiConfigFromJson(filename));
  };

  // Read space point config
  settings.spmConfig = ActsExamples::Options::readSpacePointMakerConfig(vm);

  // No need to put detector geometry writing in sequencer loop
  settings.objOutputDir = "obj_output";
  export_detector_to_obj(*geometry, settings.objOutputDir);

  // Setup gaussian distribution of the beam spot (Particle Gun)
  settings.vertexXYstd = vm["gen-vertex-xy-std-mm"].as<double>();
  settings.vertexZstd = vm["gen-vertex-z-std-mm"].as<double>();
  settings.vertexTstd = vm["gen-vertex-t-std-ns"].as<double>();

  // Set start parameters
  settings.pTransverse = vm["gen-mom-transverse"].as<bool>();
  settings.pMin = vm["gen-mom-gev"].as<ActsExamples::Options::Interval>().lower.value() * 1_GeV;
  settings.pMax = vm["gen-mom-gev"].as<ActsExamples::Options::Interval>().upper.value() * 1_GeV;
  settings.phiMin = vm["gen-phi-degree"].as<ActsExamples::Options::Interval>().lower.value() * 1_degree;
  settings.phiMax = vm["gen-phi-degree"].as<ActsExamples::Options::Interval>().upper.value() * 1_degree;
  double etaMin = vm["gen-eta"].as<ActsExamples::Options::Interval>().lower.value() * 1.0;
  double etaMax = vm["gen-eta"].as<ActsExamples::Options::Interval>().upper.value() * 1.0;
  settings.thetaMin = 2 * std::atan(std::exp(-etaMin));
  settings.thetaMax = 2 * std::atan(std::exp(-etaMax));
  settings.etaUniform = vm["gen-eta-uniform"].as<bool>();
  settings.pdg =
      static_cast<Acts::PdgParticle>(vm["gen-pdg"].as<int32_t>());


  return testGsf(settings);
}
