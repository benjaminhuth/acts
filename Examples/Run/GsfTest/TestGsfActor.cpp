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

#include "GsfAlgorithmFunction.hpp"
#include "TestHelpers.hpp"
#include "AlgorithmsAndWriters/ObjHitWriter.hpp"
#include "AlgorithmsAndWriters/ParameterEstimationPerformanceWriter.hpp"
#include "AlgorithmsAndWriters/ProtoTrackLengthSelector.hpp"
#include "AlgorithmsAndWriters/TrackFittingPerformanceWriterCsv.hpp"
#include "AlgorithmsAndWriters/SeedsFromProtoTracks.hpp"

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

struct PrintSomeStuff : ActsExamples::BareAlgorithm {
  PrintSomeStuff()
      : ActsExamples::BareAlgorithm("PrintTrackParameters",
                                    Acts::Logging::INFO) {}

  ActsExamples::ProcessCode execute(
      const ActsExamples::AlgorithmContext &ctx) const override {
    const auto &trckPars =
        ctx.eventStore.get<ActsExamples::TrackParametersContainer>(
            kProtoTrackParameters);
    const auto &protoTracks =
        ctx.eventStore.get<ActsExamples::ProtoTrackContainer>(kProtoTracks);

    std::vector<int> layerCounts(5);

    std::cout << "# Proto Tracks: " << protoTracks.size() << "\n";
    for (std::size_t i = 0; i <= 5; ++i) {
      std::cout << "# Proto Tracks with size " << i << ": "
                << std::count_if(protoTracks.begin(), protoTracks.end(),
                                 [&](const auto &t) { return t.size() == i; })
                << "\n";
    }

    std::cout << "# parameters:   " << trckPars.size() << "\n";

    for (const auto &par : trckPars) {
      Acts::FreeVector free;
      free << par.fourPosition(ctx.geoContext), par.unitDirection(),
          par.charge() / par.absoluteMomentum();

      //             std::cout << "-- " << free.transpose() << "\t|\t" << par.referenceSurface().geometryId() << "\n";

      layerCounts[par.referenceSurface().geometryId().layer() / 2 - 1]++;
    }

    for (int i = 0; i < 5; ++i)
      std::cout << "layer " << i * 2 + 2 << ": " << layerCounts[i] << "\n";

    std::exit(1);
  }
};



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
    std::cout << "\t --inflate-cov <val>\t"
              << "Inflate the covariance of esimated start parameters "
                 "(default: 1.0)\n";
    std::cout << "\t --help             \t"
              << "Print the help message\n";
    return EXIT_SUCCESS;
  }

  const auto globalLogLevel = [&]() {
    const bool found = std::find(begin(args), end(args), "-v") != args.end();
    return found ? Acts::Logging::VERBOSE : Acts::Logging::INFO;
  }();

  const auto gsfLogLevel = [&]() {
    const bool found =
        std::find(begin(args), end(args), "-v-gsf") != args.end();
    return found ? Acts::Logging::VERBOSE : Acts::Logging::INFO;
  }();

  const int numParticles = [&]() {
    const auto found = std::find(begin(args), end(args), "-n");
    const bool valid = found != end(args) && std::next(found) != end(args);
    return valid ? std::stoi(*std::next(found)) : 1;
  }();

  const double bfValue = [&]() {
    const auto found = std::find(begin(args), end(args), "--bf-value");
    if (found != args.end() && std::next(found) != args.end()) {
      return std::stod(*std::next(found)) * Acts::UnitConstants::T;
    }
    return 2.0_T;
  }();

  const auto seed = [&]() -> uint64_t {
    const auto found = std::find(begin(args), end(args), "-s");
    if (found != args.end() && std::next(found) != args.end()) {
      return std::stoul(*std::next(found));
    }
    return std::random_device{}();
  }();

  if (std::find(begin(args), end(args), "--gsf-abort-error") != args.end()) {
    setGsfAbortOnError(true);
  }

  if (auto found = std::find(begin(args), end(args), "-c");
      found != args.end() && std::next(found) != args.end()) {
    setGsfMaxComponents(std::stoi(*std::next(found)));
  }

  const bool doGsf =
      std::find(begin(args), end(args), "--no-gsf") == args.end();
  const bool doKalman =
      std::find(begin(args), end(args), "--no-kalman") == args.end();

  const bool estimateParsFromSeed =
      std::find(begin(args), end(args), "--pars-from-seed") != args.end();

  const auto inflation = [&]() -> double {
    const auto found = std::find(begin(args), end(args), "--inflate-cov");
    if (found != args.end() && std::next(found) != args.end()) {
      return std::stod(*std::next(found));
    }
    return 1.0;
  }();

  // Export the seed for reproducibility
  {
    std::ofstream seedFile("seed.txt", std::ios_base::trunc);
    seedFile << seed;
  }

  // Logger
  auto mainLogger = Acts::getDefaultLogger("main logger", globalLogLevel);
  auto multiStepperLogger = Acts::getDefaultLogger("MultiStepper", gsfLogLevel);
  ACTS_LOCAL_LOGGER(std::move(mainLogger));

  ACTS_INFO("Parameters: numParticles = " << numParticles);
  ACTS_INFO("Parameters: B-Field = " << bfValue * Acts::UnitConstants::T
                                     << "T");
  ACTS_INFO("Parameters: RNG seed = " << seed);
  ACTS_INFO("Parameters: " << (doGsf ? "do GSF" : "no Gsf") << ", "
                           << (doKalman ? "do Kalman" : "no Kalman"));
  ACTS_INFO("Parameters: Abort on error: " << std::boolalpha
                                           << getGsfAbortOnError());
  ACTS_INFO("Parameters: GSF max components: " << getGsfMaxComponents());
  ACTS_INFO("Parameters: Estimate start parameters from seeds: "
            << std::boolalpha << estimateParsFromSeed);
  ACTS_INFO("Parameters: Covariance inflation factor: " << inflation);

  // Setup the sequencer
  ActsExamples::Sequencer::Config seqCfg;
  seqCfg.events = 1;
  seqCfg.numThreads = 1;
  ActsExamples::Sequencer sequencer(seqCfg);

  // RNG
  ActsExamples::RandomNumbers::Config rndCfg{seed};
  auto rnd = std::make_shared<ActsExamples::RandomNumbers>(rndCfg);

  // MagneticField
  auto magField =
      std::make_shared<MagneticField>(Acts::Vector3(0.0, 0.0, bfValue));

  // Make the telescope detector
  const typename ActsExamples::Telescope::TelescopeDetectorElement::ContextType
      detectorContext;
  std::vector<
      std::shared_ptr<ActsExamples::Telescope::TelescopeDetectorElement>>
      detectorElementStorage;
  const std::vector<double> distances = {100_mm, 200_mm, 300_mm, 400_mm,
                                         500_mm};
  const std::array<double, 2> offsets = {0.0_mm, 0.0_mm};
  const std::array<double, 2> bounds = {100._mm, 100._mm};
  const double thickness = 1._mm;
  const auto type = ActsExamples::Telescope::TelescopeSurfaceType::Plane;
  const auto detectorDirection = Acts::BinningValue::binX;

  auto detector = std::shared_ptr(ActsExamples::Telescope::buildDetector(
      detectorContext, detectorElementStorage, distances, offsets, bounds,
      thickness, type, detectorDirection));

  // No need to put detector geometry writing in sequencer loop
#if 0
  export_detector_to_obj(*detector);
#endif

  // Find a surface in the tgeo for some use later
  std::shared_ptr<const Acts::Surface> some_surface;

  detector->visitSurfaces([&](auto surface) {
    if (surface->center(Acts::GeometryContext{})[0] == 100.0) {
      some_surface = surface->getSharedPtr();
    }
  });

  throw_assert(some_surface, "no valid surface found");
  ACTS_VERBOSE(
      "some surface normal: "
      << some_surface->normal(geoCtx, Acts::Vector2{0, 0}).transpose());

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
    cfg.trackingGeometry = detector;
    cfg.magneticField = magField;

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
    const auto volume_id =
        detector
            ->lowestTrackingVolume(
                Acts::GeometryContext{},
                some_surface->center(Acts::GeometryContext{}))
            ->geometryId()
            .volume();

    ACTS_VERBOSE("Volume id = " << volume_id);

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

    auto cfg = ActsExamples::DigitizationConfig(
        doMerge, mergeNsigma, mergeCommonCorner, volumes, indices, types,
        parameters,
        Acts::GeometryHierarchyMap<ActsExamples::DigiComponentsConfig>());

    cfg.inputSimHits = kSimulatedHits;
    cfg.outputSourceLinks = kSourceLinks;
    cfg.outputMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.outputMeasurements = kMeasurements;
    cfg.outputMeasurementSimHitsMap = kMeasurementSimHitMap;

    cfg.randomNumbers = rnd;
    cfg.trackingGeometry = detector;

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

    ActsExamples::SpacePointMaker::Config spm_cfg;
    spm_cfg.inputMeasurements = kMeasurements;
    spm_cfg.inputSourceLinks = kSourceLinks;
    spm_cfg.outputSpacePoints = kSpacePoints;
    spm_cfg.trackingGeometry = detector;
    spm_cfg.geometrySelection = {
        detector->highestTrackingVolume()->geometryId()};

    sequencer.addAlgorithm(std::make_shared<ActsExamples::SpacePointMaker>(
        spm_cfg, globalLogLevel));
    
    ActsExamples::SeedsFromProtoTracks::Config sfp_cfg;
    sfp_cfg.inProtoTracks = kIntermediateProtoTracks;
    sfp_cfg.inSpacePoints = kSpacePoints;
    sfp_cfg.outProtoTracks = kProtoTracks;
    sfp_cfg.outSeedCollection = kSeeds;
    
    sequencer.addAlgorithm(std::make_shared<ActsExamples::SeedsFromProtoTracks>(sfp_cfg, globalLogLevel));

    ActsExamples::TrackParamsEstimationAlgorithm::Config tpe_cfg;
    tpe_cfg.inputSeeds = kSeeds;
    tpe_cfg.inputSourceLinks = kSourceLinks;
    tpe_cfg.outputProtoTracks = "not-needed";
    tpe_cfg.outputTrackParameters = kProtoTrackParameters;
    tpe_cfg.trackingGeometry = detector;
    tpe_cfg.magneticField = magField;
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
    cfg.trackingGeometry = detector;
    //     cfg.targetSurface = startSurface;
    cfg.dFit = makeGsfFitterFunction(detector, magField,
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
    cfg.trackingGeometry = detector;
    //     cfg.targetSurface = startSurface;
    cfg.dFit =
        ActsExamples::TrackFittingAlgorithm::makeTrackFitterFunction(magField);
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
    cfg.outputFilename = "gsf_performance.root";

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
    cfg.outputFilename = "gsf_trackstates.root";
    cfg.outputTreename = "tree";

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
    cfg.outputFilename = "kalman_performance.root";

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
    cfg.outputFilename = "kalman_trackstates.root";
    cfg.outputTreename = "tree";

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
