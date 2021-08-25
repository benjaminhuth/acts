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
#include "ActsExamples/Fatras/FatrasAlgorithm.hpp"
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
// #include "TestHelpers.hpp"

using namespace Acts::UnitLiterals;

const char *kGeneratedParticles = "particles";
const char *kSimulatedParticlesInitial = "sim-particles-initial";
const char *kSimulatedParticlesFinal = "sim-particles-final";
const char *kSimulatedHits = "sim-hits";
const char *kMeasurements = "measurements";
const char *kMeasurementParticleMap = "measurement-particle-map";
const char *kMeasurementSimHitMap = "measurement-simhit-map";
const char *kSourceLinks = "source-links";
const char *kMultiSteppingLogAverageLoop = "average-track-loop-stepper";
const char *kMultiSteppingLogComponentsLoop = "component-tracks-loop-stepper";
const char *kProtoTracks = "proto-tracks";
const char *kProtoTrackParameters = "proto-track-parameters";
const char *kKalmanOutputTrajectories = "kalman-output";
const char *kGsfOutputTrajectories = "gsf-output";

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
    opt("loglevel", po::value<std::size_t>()->default_value(2),
        "LogLevel for compatibility, with almost no impact");
    opt("pars-from-seeds", po::bool_switch(),
        "Use track parameters estimated from truth tracks");
    opt("v", po::bool_switch(), "All algorithms verbose (except the GSF)");
    opt("v-gsf", po::bool_switch(), "GSF algorithm verbose");
    opt("no-kalman", po::bool_switch(), "Disable the GSF");
    opt("no-gsf", po::bool_switch(), "Disable the Kalman Filter");

    detector->addOptions(desc);
    Options::addRandomNumbersOptions(desc);
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

  ActsExamples::Sequencer::Config seqCfg;
  seqCfg.events = 1;
  seqCfg.numThreads = 1;
  ActsExamples::Sequencer sequencer(seqCfg);

  // Read some standard options
  const auto globalLogLevel =
      vm["v"].as<bool>() ? Acts::Logging::VERBOSE : Acts::Logging::INFO;
  const auto gsfLogLevel =
      vm["v-gsf"].as<bool>() ? Acts::Logging::VERBOSE : Acts::Logging::INFO;
  const auto doGsf = not vm["no-gsf"].as<bool>();
  const auto doKalman = not vm["no-kalman"].as<bool>();
  const auto numParticles = vm["n"].as<int>();
  const auto estimateParsFromSeed = vm["pars-from-seeds"].as<bool>();

  const double inflation = 1.0;

  auto rnd = std::make_shared<ActsExamples::RandomNumbers>(
      ActsExamples::Options::readRandomNumbersConfig(vm));

  // Logger
  auto mainLogger = Acts::getDefaultLogger("main logger", globalLogLevel);
  auto multiStepperLogger = Acts::getDefaultLogger("MultiStepper", gsfLogLevel);
  ACTS_LOCAL_LOGGER(std::move(mainLogger));

  // Setup detector geometry
  auto geometry = ActsExamples::Geometry::build(vm, *detector);
  const auto &trackingGeometry = geometry.first;

  // Add context decorators
  for (auto cdr : geometry.second) {
    sequencer.addContextDecorator(cdr);
  }

  // Setup the magnetic field
  auto magneticField = ActsExamples::Options::readMagneticField(vm);

  // No need to put detector geometry writing in sequencer loop
#if 0
  export_detector_to_obj(*trackingGeometry);
#endif

  /////////////////////
  // Particle gun
  /////////////////////
  {
    Acts::Vector4 vertex = Acts::Vector4::Zero();

    ActsExamples::FixedVertexGenerator vertexGen{vertex};

    ActsExamples::ParametricParticleGenerator::Config pgCfg;
    pgCfg.phiMin = -5._degree;
    pgCfg.phiMax = 5._degree;
    pgCfg.thetaMin = 85._degree;
    pgCfg.thetaMax = 95._degree;
    pgCfg.pMin = 1.0_GeV;
    pgCfg.pMax = 10.0_GeV;
    pgCfg.pdg = Acts::PdgParticle::eElectron;
    pgCfg.numParticles = numParticles;

    ActsExamples::EventGenerator::Config cfg;
    cfg.generators = {{ActsExamples::FixedMultiplicityGenerator{1},
                       std::move(vertexGen),
                       ActsExamples::ParametricParticleGenerator(pgCfg)}};

    cfg.outputParticles = kGeneratedParticles;
    cfg.randomNumbers = rnd;

    sequencer.addReader(
        std::make_shared<ActsExamples::EventGenerator>(cfg, globalLogLevel));
  }

  ////////////////
  // Simulation
  ////////////////
  {
    ActsExamples::FatrasAlgorithm::Config cfg;
    cfg.inputParticles = kGeneratedParticles;
    cfg.outputParticlesInitial = kSimulatedParticlesInitial;
    cfg.outputParticlesFinal = kSimulatedParticlesFinal;
    cfg.outputSimHits = kSimulatedHits;
    cfg.randomNumbers = rnd;
    cfg.trackingGeometry = trackingGeometry;
    cfg.magneticField = magneticField;

    cfg.emScattering = true;
    cfg.emEnergyLossIonisation = true;
    cfg.emEnergyLossRadiation = true;
    cfg.emPhotonConversion = false;
    cfg.generateHitsOnSensitive = true;
    cfg.generateHitsOnMaterial = true;
    cfg.generateHitsOnPassive = false;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::FatrasAlgorithm>(
        std::move(cfg), globalLogLevel));
  }

  ///////////////////
  // Digitization
  ///////////////////
  {
    auto cfg = ActsExamples::DigitizationConfig(
        vm, ActsExamples::readDigiConfigFromJson(
                  vm["digi-config-file"].as<std::string>()));

    cfg.inputSimHits = kSimulatedHits;
    cfg.outputSourceLinks = kSourceLinks;
    cfg.outputMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.outputMeasurements = kMeasurements;
    cfg.outputMeasurementSimHitsMap = kMeasurementSimHitMap;

    cfg.randomNumbers = rnd;
    cfg.trackingGeometry = trackingGeometry;

    sequencer.addAlgorithm(createDigitizationAlgorithm(cfg, globalLogLevel));
  }

  ///////////////////////////
  // Prepare Track fitting //
  ///////////////////////////
  if (!estimateParsFromSeed) {
    ActsExamples::TruthTrackFinder::Config ttf_cfg;
    ttf_cfg.inputParticles = kGeneratedParticles;
    ttf_cfg.inputMeasurementParticlesMap = kMeasurementParticleMap;
    ttf_cfg.outputProtoTracks = kProtoTracks;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::TruthTrackFinder>(
        ttf_cfg, globalLogLevel));

    ActsExamples::ParticleSmearing::Config ps_cfg;
    ps_cfg.inputParticles = kGeneratedParticles;
    ps_cfg.outputTrackParameters = kProtoTrackParameters;
    ps_cfg.randomNumbers = rnd;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::ParticleSmearing>(
        ps_cfg, globalLogLevel));
  }

  /////////////////////////////
  // Seed - Estimation chain //
  /////////////////////////////
  else {
    const char *kSpacePoints = "space-points";
    const char *kIntermediateProtoTracks = "intermediate-proto-tracks";
    const char *kSeeds = "seeds";

    ActsExamples::TruthTrackFinder::Config ttf_cfg;
    ttf_cfg.inputParticles = kGeneratedParticles;
    ttf_cfg.inputMeasurementParticlesMap = kMeasurementParticleMap;
    ttf_cfg.outputProtoTracks = kIntermediateProtoTracks;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::TruthTrackFinder>(
        ttf_cfg, globalLogLevel));

    auto spm_cfg = ActsExamples::Options::readSpacePointMakerConfig(vm);
    spm_cfg.inputMeasurements = kMeasurements;
    spm_cfg.inputSourceLinks = kSourceLinks;
    spm_cfg.outputSpacePoints = kSpacePoints;
    spm_cfg.trackingGeometry = trackingGeometry;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::SpacePointMaker>(
        spm_cfg, globalLogLevel));

    ActsExamples::SeedsFromProtoTracks::Config sfp_cfg;
    sfp_cfg.inProtoTracks = kIntermediateProtoTracks;
    sfp_cfg.inSpacePoints = kSpacePoints;
    sfp_cfg.outProtoTracks = kProtoTracks;
    sfp_cfg.outSeedCollection = kSeeds;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::SeedsFromProtoTracks>(
        sfp_cfg, globalLogLevel));

    ActsExamples::TrackParamsEstimationAlgorithm::Config tpe_cfg;
    tpe_cfg.inputSeeds = kSeeds;
    tpe_cfg.inputSourceLinks = kSourceLinks;
    tpe_cfg.outputProtoTracks = "not-needed";
    tpe_cfg.outputTrackParameters = kProtoTrackParameters;
    tpe_cfg.trackingGeometry = trackingGeometry;
    tpe_cfg.magneticField = magneticField;
    tpe_cfg.initialVarInflation = {inflation, inflation, inflation,
                                   inflation, inflation, inflation};

    sequencer.addAlgorithm(
        std::make_shared<ActsExamples::TrackParamsEstimationAlgorithm>(
            tpe_cfg, globalLogLevel));
  }

#if 0
  /////////////////
  // Select Tracks
  /////////////////
  {
    ActsExamples::ProtoTrackLengthSelector::Config cfg;
    cfg.inProtoTracks = kProtoTracks;
    cfg.outProtoTracks = kSelectedProtoTracks;
    cfg.minLength = 3;

    sequencer.addAlgorithm(
        std::make_shared<ActsExamples::ProtoTrackLengthSelector>(
            cfg, globalLogLevel));
  }
#endif

  //   sequencer.addAlgorithm(std::make_shared<PrintSomeStuff>());

  ////////////////////////
  // Gaussian Sum Filter
  ////////////////////////
  if (doGsf) {
    ActsExamples::TrackFittingAlgorithm::Config cfg;

    cfg.inputMeasurements = kMeasurements;
    cfg.inputSourceLinks = kSourceLinks;
    cfg.inputProtoTracks = kProtoTracks;
    cfg.inputInitialTrackParameters = kProtoTrackParameters;
    cfg.outputTrajectories = kGsfOutputTrajectories;
    cfg.directNavigation = true;
    cfg.trackingGeometry = trackingGeometry;
    cfg.dFit = makeGsfFitterFunction(trackingGeometry, magneticField,
                                     Acts::LoggerWrapper(*multiStepperLogger));
    cfg.fitterType = "GSF";

    sequencer.addAlgorithm(
        std::make_shared<ActsExamples::TrackFittingAlgorithm>(cfg,
                                                              gsfLogLevel));
  }

  //////////////////////////////////
  // Kalman Fitter for comparison //
  //////////////////////////////////
  if (doKalman) {
    ActsExamples::TrackFittingAlgorithm::Config cfg;

    cfg.inputMeasurements = kMeasurements;
    cfg.inputSourceLinks = kSourceLinks;
    cfg.inputProtoTracks = kProtoTracks;
    cfg.inputInitialTrackParameters = kProtoTrackParameters;
    cfg.outputTrajectories = kKalmanOutputTrajectories;
    cfg.directNavigation = true;
    cfg.trackingGeometry = trackingGeometry;
    cfg.dFit = ActsExamples::TrackFittingAlgorithm::makeTrackFitterFunction(
        magneticField);
    cfg.fitterType = "Kalman";

    sequencer.addAlgorithm(
        std::make_shared<ActsExamples::TrackFittingAlgorithm>(cfg,
                                                              globalLogLevel));
  }

  ////////////////////////////////////////////
  // Track Fitter Performance Writer for GSF
  ////////////////////////////////////////////
  if (doGsf) {
    ActsExamples::TrackFitterPerformanceWriter::Config cfg;
    cfg.inputMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.inputParticles = kGeneratedParticles;
    cfg.inputTrajectories = kGsfOutputTrajectories;
    cfg.filePath = "gsf_performance.root";

    sequencer.addWriter(
        std::make_shared<ActsExamples::TrackFitterPerformanceWriter>(
            cfg, globalLogLevel));
  }

  if (doGsf) {
    ActsExamples::RootTrajectoryStatesWriter::Config cfg;
    cfg.inputMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.inputMeasurementSimHitsMap = kMeasurementSimHitMap;
    cfg.inputParticles = kGeneratedParticles;
    cfg.inputSimHits = kSimulatedHits;
    cfg.inputTrajectories = kGsfOutputTrajectories;
    cfg.filePath = "gsf_trackstates.root";
    cfg.treeName = "tree";

    sequencer.addWriter(
        std::make_shared<ActsExamples::RootTrajectoryStatesWriter>(
            cfg, globalLogLevel));
  }

  if (doGsf) {
    ActsExamples::TrackFittingPerformanceWriterCsv::Config cfg;
    cfg.inTrajectories = kGsfOutputTrajectories;
    cfg.inParticles = kGeneratedParticles;
    cfg.inMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.outputStem += "-gsf";

    sequencer.addWriter(
        std::make_shared<ActsExamples::TrackFittingPerformanceWriterCsv>(
            cfg, globalLogLevel));
  }

  //////////////////////////////////////////////////////
  // Track Fitter Performance Writer for Kalman Fitter
  //////////////////////////////////////////////////////
  if (doKalman) {
    ActsExamples::TrackFitterPerformanceWriter::Config cfg;
    cfg.inputMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.inputParticles = kGeneratedParticles;
    cfg.inputTrajectories = kKalmanOutputTrajectories;
    cfg.filePath = "kalman_performance.root";

    sequencer.addWriter(
        std::make_shared<ActsExamples::TrackFitterPerformanceWriter>(
            cfg, globalLogLevel));
  }

  if (doKalman) {
    ActsExamples::RootTrajectoryStatesWriter::Config cfg;
    cfg.inputMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.inputMeasurementSimHitsMap = kMeasurementSimHitMap;
    cfg.inputParticles = kGeneratedParticles;
    cfg.inputSimHits = kSimulatedHits;
    cfg.inputTrajectories = kKalmanOutputTrajectories;
    cfg.filePath = "kalman_trackstates.root";
    cfg.treeName = "tree";

    sequencer.addWriter(
        std::make_shared<ActsExamples::RootTrajectoryStatesWriter>(
            cfg, globalLogLevel));
  }

  if (doKalman) {
    ActsExamples::TrackFittingPerformanceWriterCsv::Config cfg;
    cfg.inTrajectories = kKalmanOutputTrajectories;
    cfg.inParticles = kGeneratedParticles;
    cfg.inMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.outputStem += "-kalman";

    sequencer.addWriter(
        std::make_shared<ActsExamples::TrackFittingPerformanceWriterCsv>(
            cfg, globalLogLevel));
  }

  //////////////////////////////////////////
  // Start parameter estimation residuals
  //////////////////////////////////////////
  {
    ActsExamples::ParameterEstimationPerformanceWriter::Config cfg;
    cfg.inMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.inParticles = kGeneratedParticles;
    cfg.inProtoTrackParameters = kProtoTrackParameters;
    cfg.inProtoTracks = kProtoTracks;

    sequencer.addWriter(
        std::make_shared<ActsExamples::ParameterEstimationPerformanceWriter>(
            cfg, globalLogLevel));
  }

  return sequencer.run();
}
