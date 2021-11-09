// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "TestGsfGeneric.hpp"

#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/Utilities/PdgParticle.hpp"
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
#include "TestHelpers.hpp"

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
const char *kSelectedProtoTracks = "selected-proto-tracks";
const char *kProtoTrackParameters = "proto-track-parameters";
const char *kTrackParametersFromKalman = "track-parameters-from-kalman";
const char *kProtoTracksFromKalman = "proto-tracks-from-kalman";
const char *kKalmanOutputTrajectories = "kalman-output";
const char *kGsfOutputTrajectories = "gsf-output";

struct AbortIfAllTracksEmpty : ActsExamples::BareAlgorithm {
  struct Config {
    std::string inProtoTracks;
  } m_cfg;

  AbortIfAllTracksEmpty(const Config &cfg, Acts::Logging::Level lvl)
      : ActsExamples::BareAlgorithm("AbortIfAllTracksEmpty", lvl), m_cfg(cfg) {}

  ActsExamples::ProcessCode execute(
      const ActsExamples::AlgorithmContext &ctx) const override {
    const auto &tracks = ctx.eventStore.get<ActsExamples::ProtoTrackContainer>(
        m_cfg.inProtoTracks);

    bool doAbort = true;
    int emptyTracks = 0;

    for (const auto &track : tracks) {
      if (track.empty()) {
        emptyTracks++;
      } else {
        doAbort = false;
      }
    }

    ACTS_INFO("Empty Tracks: " << emptyTracks << " of " << tracks.size());

    if (doAbort) {
      return ActsExamples::ProcessCode::ABORT;
    }

    return ActsExamples::ProcessCode::SUCCESS;
  }
};

struct AbortIfAllTrajectoriesEmpty : ActsExamples::BareAlgorithm {
  struct Config {
    std::string inTrajectories;
  } m_cfg;

  AbortIfAllTrajectoriesEmpty(const Config &cfg, Acts::Logging::Level lvl)
      : ActsExamples::BareAlgorithm("AbortIfAllTrajectoriesEmpty", lvl),
        m_cfg(cfg) {}

  ActsExamples::ProcessCode execute(
      const ActsExamples::AlgorithmContext &ctx) const override {
    const auto &trajectories =
        ctx.eventStore.get<ActsExamples::TrajectoriesContainer>(
            m_cfg.inTrajectories);

    bool doAbort = true;
    int empty = 0;

    for (const auto &trajectory : trajectories) {
      if (trajectory.empty()) {
        empty++;
      } else {
        doAbort = false;
      }
    }

    ACTS_INFO("Empty Trajectories: " << empty << " of " << trajectories.size());

    if (doAbort) {
      return ActsExamples::ProcessCode::ABORT;
    }

    return ActsExamples::ProcessCode::SUCCESS;
  }
};

struct ExtractKalmanResultForGsf : ActsExamples::BareAlgorithm {
  struct Config {
    std::string inTrajectories;
    std::string inProtoTracks;
    std::string outTrackParameters;
    std::string outProtoTracks;
  } m_cfg;

  ExtractKalmanResultForGsf(const Config &cfg, Acts::Logging::Level lvl)
      : ActsExamples::BareAlgorithm("ExtractKalmanResultForGsf", lvl),
        m_cfg(cfg) {}

  ActsExamples::ProcessCode execute(
      const ActsExamples::AlgorithmContext &ctx) const override {
    const auto &inTrajectories =
        ctx.eventStore.get<ActsExamples::TrajectoriesContainer>(
            m_cfg.inTrajectories);
    const auto &inProtoTracks =
        ctx.eventStore.get<ActsExamples::ProtoTrackContainer>(
            m_cfg.inProtoTracks);

    ActsExamples::TrackParametersContainer outParameters;
    ActsExamples::ProtoTrackContainer outProtoTracks;

    int invalid = 0;

    throw_assert(inTrajectories.size() == inProtoTracks.size(),
                 "size mismatch");

    for (auto i = 0ul; i < inTrajectories.size(); ++i) {
      const auto &trajectory = inTrajectories[i];

      const bool trajectoryValid =
          !trajectory.empty() &&
          trajectory.hasTrackParameters(trajectory.tips().at(0));

      if (trajectoryValid && !inProtoTracks[i].empty()) {
        outParameters.push_back(
            trajectory.trackParameters(trajectory.tips().front()));
        outProtoTracks.push_back(inProtoTracks[i]);
      } else {
        outParameters.push_back(ActsExamples::TrackParameters({}, {}));
        outProtoTracks.push_back(ActsExamples::ProtoTrack{});
        invalid++;
      }
    }

    throw_assert(outParameters.size() - invalid > 0, "no valid track left");

    ctx.eventStore.add<decltype(outParameters)>(m_cfg.outTrackParameters,
                                                std::move(outParameters));
    ctx.eventStore.add<decltype(outProtoTracks)>(m_cfg.outProtoTracks,
                                                 std::move(outProtoTracks));

    return ActsExamples::ProcessCode::SUCCESS;
  }
};

int testGsf(const GsfTestSettings &settings) {
  // Logger
  auto mainLogger =
      Acts::getDefaultLogger("main logger", settings.globalLogLevel);
  auto multiStepperLogger =
      Acts::getDefaultLogger("MultiStepper", settings.gsfLogLevel);
  ACTS_LOCAL_LOGGER(std::move(mainLogger));

  // Some checks
  //   if (settings.doRefit && !(settings.doKalman && settings.doGsf)) {
  //     throw std::invalid_argument(
  //         "if 'doRefit' is enabled, both Kalman Fitter and Gsf must be
  //         enabled");
  //   }

  // Summary
  ACTS_INFO("Parameters: numParticles = " << settings.numParticles);
  ACTS_INFO("Parameters: B-Field strength at origin = " << [&]() {
    auto cache =
        settings.magneticField->makeCache(Acts::MagneticFieldContext{});
    return (*settings.magneticField->getField(Acts::Vector3::Zero(), cache))
               .norm() /
           Acts::UnitConstants::T;
  }());
  ACTS_INFO("Parameters: doRefit = " << std::boolalpha << settings.doRefit);
  ACTS_INFO("Parameters: RNG seed = " << settings.seed);
  ACTS_INFO("Parameters: " << (settings.doGsf ? "do GSF" : "no Gsf") << ", "
                           << (settings.doKalman ? "do Kalman" : "no Kalman"));
  ACTS_INFO("Parameters: Abort on error: " << std::boolalpha
                                           << getGsfAbortOnError());
  ACTS_INFO("Parameters: GSF max components: " << getGsfMaxComponents());
  ACTS_INFO("Parameters: Estimate start parameters from seeds: "
            << std::boolalpha << settings.estimateParsFromSeed);
  ACTS_INFO("Parameters: Covariance inflation factor: " << settings.inflation);

  // Init Sequencer
  ActsExamples::Sequencer::Config seqCfg;
  seqCfg.events = 1;
  seqCfg.numThreads = 1;
  ActsExamples::Sequencer sequencer(seqCfg);

  // Context decorators
  for (auto cdr : settings.contextDecorators) {
    sequencer.addContextDecorator(cdr);
  }

  // Make RNG
  ActsExamples::RandomNumbers::Config rndCfg{settings.seed};
  auto rnd = std::make_shared<ActsExamples::RandomNumbers>(rndCfg);

  // Export the seed for reproducibility
  {
    std::ofstream seedFile("seed.txt", std::ios_base::trunc);
    seedFile << settings.seed << "\n";
  }

  // Gsf settings
  setGsfMaxComponents(settings.maxComponents);
  setGsfAbortOnError(settings.gsfAbortOnError);
  setGsfMaxSteps(settings.maxSteps);
  setGsfLoopProtection(settings.gsfLoopProtection);

  /////////////////////
  // Particle gun
  /////////////////////
  {
    Acts::Vector4 vertex = Acts::Vector4::Zero();

    auto vertexGen = std::make_shared<ActsExamples::FixedVertexGenerator>();
    vertexGen->fixed = vertex;

    ActsExamples::ParametricParticleGenerator::Config pgCfg;
    pgCfg.phiMin = settings.phiMin;
    pgCfg.phiMax = settings.phiMax;
    pgCfg.thetaMin = settings.thetaMin;
    pgCfg.thetaMax = settings.thetaMax;
    pgCfg.pMin = settings.pMin;
    pgCfg.pMax = settings.pMax;
    pgCfg.pdg = Acts::PdgParticle::eElectron;
    pgCfg.numParticles = settings.numParticles;

    ActsExamples::EventGenerator::Config cfg;
    cfg.generators = {
        {std::make_shared<ActsExamples::FixedMultiplicityGenerator>(1),
         vertexGen,
         std::make_shared<ActsExamples::ParametricParticleGenerator>(pgCfg)}};

    cfg.outputParticles = kGeneratedParticles;
    cfg.randomNumbers = rnd;

    sequencer.addReader(std::make_shared<ActsExamples::EventGenerator>(
        cfg, settings.globalLogLevel));
  }

  ////////////////
  // Simulation
  ////////////////
  {
    ActsExamples::FatrasSimulation::Config cfg;
    cfg.inputParticles = kGeneratedParticles;
    cfg.outputParticlesInitial = kSimulatedParticlesInitial;
    cfg.outputParticlesFinal = kSimulatedParticlesFinal;
    cfg.outputSimHits = kSimulatedHits;
    cfg.randomNumbers = rnd;
    cfg.trackingGeometry = settings.geometry;
    cfg.magneticField = settings.magneticField;

    cfg.emScattering = true;
    cfg.emEnergyLossIonisation = true;
    cfg.emEnergyLossRadiation = true;
    cfg.emPhotonConversion = false;
    cfg.generateHitsOnSensitive = true;
    cfg.generateHitsOnMaterial = false;
    cfg.generateHitsOnPassive = false;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::FatrasSimulation>(
        std::move(cfg), settings.globalLogLevel));
  }

  ///////////////////
  // Digitization
  ///////////////////
  {
    auto cfg = settings.digiConfigFactory();
    cfg.inputSimHits = kSimulatedHits;
    cfg.outputSourceLinks = kSourceLinks;
    cfg.outputMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.outputMeasurements = kMeasurements;
    cfg.outputMeasurementSimHitsMap = kMeasurementSimHitMap;

    cfg.randomNumbers = rnd;
    cfg.trackingGeometry = settings.geometry;

    sequencer.addAlgorithm(
        createDigitizationAlgorithm(cfg, settings.globalLogLevel));
  }

  ///////////////////////////
  // Prepare Track fitting //
  ///////////////////////////
  if (!settings.estimateParsFromSeed) {
    ActsExamples::TruthTrackFinder::Config ttf_cfg;
    ttf_cfg.inputParticles = kGeneratedParticles;
    ttf_cfg.inputMeasurementParticlesMap = kMeasurementParticleMap;
    ttf_cfg.outputProtoTracks = kProtoTracks;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::TruthTrackFinder>(
        ttf_cfg, settings.globalLogLevel));

    ActsExamples::ParticleSmearing::Config ps_cfg;
    ps_cfg.inputParticles = kGeneratedParticles;
    ps_cfg.outputTrackParameters = kProtoTrackParameters;
    ps_cfg.randomNumbers = rnd;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::ParticleSmearing>(
        ps_cfg, settings.globalLogLevel));
  }

  /////////////////////////////
  // Seed - Estimation chain //
  /////////////////////////////
  else {
    const char *kIntermediateProtoTracks = "intermediate-proto-tracks";
    const char *kSeeds = "seeds";
    const char *kSpacePoints = "space-points";

    ActsExamples::TruthTrackFinder::Config ttf_cfg;
    ttf_cfg.inputParticles = kGeneratedParticles;
    ttf_cfg.inputMeasurementParticlesMap = kMeasurementParticleMap;
    ttf_cfg.outputProtoTracks = kIntermediateProtoTracks;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::TruthTrackFinder>(
        ttf_cfg, settings.globalLogLevel));

    auto spm_cfg = settings.spmConfig;
    spm_cfg.inputMeasurements = kMeasurements;
    spm_cfg.inputSourceLinks = kSourceLinks;
    spm_cfg.outputSpacePoints = kSpacePoints;
    spm_cfg.trackingGeometry = settings.geometry;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::SpacePointMaker>(
        spm_cfg, settings.globalLogLevel));

    ActsExamples::SeedsFromProtoTracks::Config sfp_cfg;
    sfp_cfg.inProtoTracks = kIntermediateProtoTracks;
    sfp_cfg.inSpacePoints = kSpacePoints;
    sfp_cfg.outProtoTracks = kProtoTracks;
    sfp_cfg.outSeedCollection = kSeeds;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::SeedsFromProtoTracks>(
        sfp_cfg, settings.globalLogLevel));

    ActsExamples::TrackParamsEstimationAlgorithm::Config tpe_cfg;
    tpe_cfg.inputSeeds = kSeeds;
    tpe_cfg.inputSourceLinks = kSourceLinks;
    tpe_cfg.outputProtoTracks = "not-needed";
    tpe_cfg.outputTrackParameters = kProtoTrackParameters;
    tpe_cfg.trackingGeometry = settings.geometry;
    tpe_cfg.magneticField = settings.magneticField;
    tpe_cfg.initialVarInflation = {settings.inflation, settings.inflation,
                                   settings.inflation, settings.inflation,
                                   settings.inflation, settings.inflation};

    sequencer.addAlgorithm(
        std::make_shared<ActsExamples::TrackParamsEstimationAlgorithm>(
            tpe_cfg, settings.globalLogLevel));
  }

#if 1
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
            cfg, settings.globalLogLevel));
  }
#endif

#if 1
  /////////////////
  // Some check
  /////////////////
  {
    AbortIfAllTracksEmpty::Config cfg;
    cfg.inProtoTracks = kSelectedProtoTracks;

    sequencer.addAlgorithm(
        std::make_shared<AbortIfAllTracksEmpty>(cfg, settings.globalLogLevel));
  }
#endif

  ///////////////////
  // Kalman Fitter //
  ///////////////////
  if (settings.doKalman) {
    ActsExamples::TrackFittingAlgorithm::Config cfg;

    cfg.inputMeasurements = kMeasurements;
    cfg.inputSourceLinks = kSourceLinks;
    cfg.inputProtoTracks = kSelectedProtoTracks;
    cfg.inputInitialTrackParameters = kProtoTrackParameters;
    cfg.outputTrajectories = kKalmanOutputTrajectories;
    cfg.trackingGeometry = settings.geometry;
    cfg.directNavigation = settings.doDirectNavigation;
    if (settings.doDirectNavigation) {
      cfg.dFit = ActsExamples::TrackFittingAlgorithm::makeTrackFitterFunction(
          settings.magneticField);
    } else {
      cfg.fit = ActsExamples::TrackFittingAlgorithm::makeTrackFitterFunction(
          settings.geometry, settings.magneticField);
    }
    cfg.fitterType = "Kalman";

    sequencer.addAlgorithm(
        std::make_shared<ActsExamples::TrackFittingAlgorithm>(
            cfg, settings.globalLogLevel));
  }

  //////////////////////////////////////////////////////////////////////
  // Extract start parameters from Kalman-Fitter in case of GSF-refit //
  //////////////////////////////////////////////////////////////////////
  if (settings.doRefit) {
    ExtractKalmanResultForGsf::Config cfg;
    cfg.inTrajectories = kKalmanOutputTrajectories;
    cfg.inProtoTracks = kSelectedProtoTracks;
    cfg.outTrackParameters = kTrackParametersFromKalman;
    cfg.outProtoTracks = kProtoTracksFromKalman;

    sequencer.addAlgorithm(std::make_shared<ExtractKalmanResultForGsf>(
        cfg, settings.globalLogLevel));
  }

  ////////////////////////
  // Gaussian Sum Filter
  ////////////////////////
  if (settings.doGsf) {
    ActsExamples::TrackFittingAlgorithm::Config cfg;

    cfg.inputMeasurements = kMeasurements;
    cfg.inputSourceLinks = kSourceLinks;
    if (settings.doRefit) {
      cfg.inputInitialTrackParameters = kTrackParametersFromKalman;
      cfg.inputProtoTracks = kProtoTracksFromKalman;
    } else {
      cfg.inputInitialTrackParameters = kProtoTrackParameters;
      cfg.inputProtoTracks = kSelectedProtoTracks;
    }
    cfg.outputTrajectories = kGsfOutputTrajectories;
    cfg.trackingGeometry = settings.geometry;
    cfg.directNavigation = settings.doDirectNavigation;
    if (settings.doDirectNavigation) {
      cfg.dFit =
          makeGsfDirectFitterFunction(settings.geometry, settings.magneticField,
                                      Acts::LoggerWrapper(*multiStepperLogger));
    } else {
      cfg.fit = makeGsfStandardFitterFunction(
          settings.geometry, settings.magneticField,
          Acts::LoggerWrapper(*multiStepperLogger));
    }
    cfg.fitterType = "GSF";

    sequencer.addAlgorithm(
        std::make_shared<ActsExamples::TrackFittingAlgorithm>(
            cfg, settings.gsfLogLevel));

    sequencer.addAlgorithm(std::make_shared<AbortIfAllTrajectoriesEmpty>(
        AbortIfAllTrajectoriesEmpty::Config{kGsfOutputTrajectories},
        settings.globalLogLevel));
  }

  ////////////////////////////////////////////
  // Track Fitter Performance Writer for GSF
  ////////////////////////////////////////////
  if (settings.doGsf) {
    ActsExamples::TrackFitterPerformanceWriter::Config cfg;
    cfg.inputMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.inputParticles = kGeneratedParticles;
    cfg.inputTrajectories = kGsfOutputTrajectories;
    cfg.filePath = "gsf_performance.root";

    sequencer.addWriter(
        std::make_shared<ActsExamples::TrackFitterPerformanceWriter>(
            cfg, settings.globalLogLevel));
  }

  if (settings.doGsf) {
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
            cfg, settings.globalLogLevel));
  }

  if (settings.doGsf) {
    ActsExamples::TrackFittingPerformanceWriterCsv::Config cfg;
    cfg.inTrajectories = kGsfOutputTrajectories;
    cfg.inParticles = kGeneratedParticles;
    cfg.inMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.outputStem += "-gsf";

    sequencer.addWriter(
        std::make_shared<ActsExamples::TrackFittingPerformanceWriterCsv>(
            cfg, settings.globalLogLevel));
  }

  //////////////////////////////////////////////////////
  // Track Fitter Performance Writer for Kalman Fitter
  //////////////////////////////////////////////////////
  if (settings.doKalman) {
    ActsExamples::TrackFitterPerformanceWriter::Config cfg;
    cfg.inputMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.inputParticles = kGeneratedParticles;
    cfg.inputTrajectories = kKalmanOutputTrajectories;
    cfg.filePath = "kalman_performance.root";

    sequencer.addWriter(
        std::make_shared<ActsExamples::TrackFitterPerformanceWriter>(
            cfg, settings.globalLogLevel));
  }

  if (settings.doKalman) {
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
            cfg, settings.globalLogLevel));
  }

  if (settings.doKalman) {
    ActsExamples::TrackFittingPerformanceWriterCsv::Config cfg;
    cfg.inTrajectories = kKalmanOutputTrajectories;
    cfg.inParticles = kGeneratedParticles;
    cfg.inMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.outputStem += "-kalman";

    sequencer.addWriter(
        std::make_shared<ActsExamples::TrackFittingPerformanceWriterCsv>(
            cfg, settings.globalLogLevel));
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
            cfg, settings.globalLogLevel));
  }

  //////////////////////////////////////////
  // Write spacepoints to obj
  //////////////////////////////////////////
  {
    ActsExamples::ObjHitWriter::Config cfg;
    cfg.collection = kSimulatedHits;
    cfg.outputDir = settings.objOutputDir;

    sequencer.addWriter(std::make_shared<ActsExamples::ObjHitWriter>(
        cfg, settings.globalLogLevel));
  }

#if 0
  /////////////////////////
  // Write Obj
  /////////////////////////
  using ObjPropStepWriter =
      ActsExamples::ObjPropagationStepsWriter<Acts::detail::Step>;

  {
    ObjPropStepWriter::Config cfg;
    cfg.outputDir = "";
    cfg.collection = kMultiSteppingLogAverageLoop;
    cfg.outputStem = "averaged-steps";

    sequencer.addWriter(
        std::make_shared<ObjPropStepWriter>(cfg, globalLogLevel));
  }

  {
    ObjPropStepWriter::Config cfg;
    cfg.outputDir = "";
    cfg.collection = kMultiSteppingLogComponentsLoop;
    cfg.outputStem = "components-steps";

    sequencer.addWriter(
        std::make_shared<ObjPropStepWriter>(cfg, globalLogLevel));
  }

  {
    ActsExamples::ObjHitWriter::Config cfg;
    cfg.outputDir = "";
    cfg.collection = kSimulatedHits;

    sequencer.addWriter(
        std::make_shared<ActsExamples::ObjHitWriter>(cfg, globalLogLevel));
  }

  //////////////////
  // Write CSV
  ///////////////////
  {
    ActsExamples::CsvTrackingGeometryWriter::Config cfg;
    cfg.trackingGeometry = detector;
    cfg.outputDir = "";
    cfg.writePerEvent = true;
    sequencer.addWriter(
        std::make_shared<ActsExamples::CsvTrackingGeometryWriter>(
            cfg, globalLogLevel));
  }

  {
    ActsExamples::CsvSimHitWriter::Config cfg;
    cfg.inputSimHits = kSimulatedHits;
    cfg.outputDir = "";
    cfg.outputStem = "simulated-hits";

    sequencer.addWriter(
        std::make_shared<ActsExamples::CsvSimHitWriter>(cfg, globalLogLevel));
  }

  {
    ActsExamples::CsvPropagationStepsWriter::Config cfg;
    cfg.collection = kMultiSteppingLogAverageLoop;
    cfg.outputDir = "";

    sequencer.addWriter(
        std::make_shared<ActsExamples::CsvPropagationStepsWriter>(
            cfg, globalLogLevel));
  }
#endif

  return sequencer.run();
}
