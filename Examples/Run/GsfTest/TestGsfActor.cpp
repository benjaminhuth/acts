// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Utilities/PdgParticle.hpp"
#include "ActsExamples/Digitization/DigitizationConfig.hpp"
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

#include <chrono>
#include <iostream>
#include <random>

#include "GSFAlgorithmFunction.hpp"
#include "ObjHitWriter.hpp"
#include "TestHelpers.hpp"

using namespace Acts::UnitLiterals;

const char *kGeneratedParticles = "particles";
const char *kSimulatedParticlesInitial = "sim-particles-initial";
const char *kSimulatedParticlesFinal = "sim-particles-final";
const char *kSimulatedHits = "sim-hits";
const char *kMeasurements = "measurements";
const char *kMeasurementParticleMap = "measurement-particle-map";
const char *kSourceLinks = "source-links";
const char *kMultiSteppingLogAverageLoop = "average-track-loop-stepper";
const char *kMultiSteppingLogComponentsLoop = "component-tracks-loop-stepper";
const char *kProtoTracks = "proto-tracks";
const char *kProtoTrackParameters = "proto-track-parameters";
const char *kKalmanOutputTrajectories = "kalman-output";
const char *kGsfOutputTrajectories = "gsf-output";

int main(int argc, char **argv) {
  const std::vector<std::string> args(argv, argv + argc);

  if (std::find(begin(args), end(args), "--help") != args.end()) {
    std::cout << "Usage: " << args[0] << " <options>\n";
    std::cout << "Options:\n";
    std::cout << "\t --bf-value <val-in-tesla>\n";
    return EXIT_SUCCESS;
  }

  const double bfValue = [&]() {
    const auto found = std::find(begin(args), end(args), "--bf-value");
    if (found != args.end() && std::next(found) != args.end()) {
      return std::stod(*std::next(found)) * Acts::UnitConstants::T;
    }
    return 2.0_T;
  }();

  // Logger
  auto mainLogger = Acts::getDefaultLogger("main logger", Acts::Logging::INFO);
  ACTS_LOCAL_LOGGER(std::move(mainLogger));

  // Setup the sequencer
  ActsExamples::Sequencer::Config seqCfg;
  seqCfg.events = 1;
  seqCfg.numThreads = 1;
  ActsExamples::Sequencer sequencer(seqCfg);

  // RNG
  ActsExamples::RandomNumbers::Config rndCfg{std::random_device{}()};
  auto rnd = std::make_shared<ActsExamples::RandomNumbers>(rndCfg);

  // MagneticField
  auto magField =
      std::make_shared<MagneticField>(Acts::Vector3(0.0, 0.0, bfValue));

  // Tracking Geometry
  //   auto [start_surface, detector] = build_tracking_geometry();

  const typename ActsExamples::Telescope::TelescopeDetectorElement::ContextType
      detectorContext;
  std::vector<
      std::shared_ptr<ActsExamples::Telescope::TelescopeDetectorElement>>
      detectorElementStorage;
  const std::vector<double> distances = {0_mm, 200_mm, 300_mm, 400_mm, 500_mm};
  const std::array<double, 2> offsets = {0.0_mm, 0.0_mm};
  const std::array<double, 2> bounds = {100._mm, 100._mm};
  const double thickness = 10._mm;
  const auto type = ActsExamples::Telescope::TelescopeSurfaceType::Plane;
  const auto detectorDirection = Acts::BinningValue::binX;

  auto detector = std::shared_ptr(ActsExamples::Telescope::buildDetector(
      detectorContext, detectorElementStorage, distances, offsets, bounds,
      thickness, type, detectorDirection));

  // No need to put detector geometry writing in sequencer loop
  export_detector_to_obj(*detector);

  // Find start surface
  std::shared_ptr<const Acts::Surface> start_surface;

  detector->visitSurfaces([&](auto surface) {
    if (surface->center(Acts::GeometryContext{})[0] == 0.0) {
      start_surface = surface->getSharedPtr();
    }
  });

  throw_assert(start_surface, "no valid start surface found");

  /////////////////////
  // Particle gun
  /////////////////////
  {
    Acts::Vector4 vertex;
    vertex << start_surface->center(Acts::GeometryContext{}), 0.0;

    ActsExamples::FixedVertexGenerator vertexGen{vertex};

    ActsExamples::ParametricParticleGenerator::Config pgCfg;
    pgCfg.phiMin = 0.0_degree;
    pgCfg.phiMax = 1.0_degree;
    pgCfg.thetaMin = 85.0_degree;
    pgCfg.thetaMax = 95.0_degree;
    pgCfg.pMin = 1.0_GeV;
    pgCfg.pMax = 10.0_GeV;
    pgCfg.pdg = Acts::PdgParticle::eElectron;
    pgCfg.numParticles = 1;

    ActsExamples::EventGenerator::Config cfg;
    cfg.generators = {{ActsExamples::FixedMultiplicityGenerator{1},
                       std::move(vertexGen),
                       ActsExamples::ParametricParticleGenerator(pgCfg)}};

    cfg.outputParticles = "particles";
    cfg.randomNumbers = rnd;

    sequencer.addReader(std::make_shared<ActsExamples::EventGenerator>(
        cfg, Acts::Logging::INFO));
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
        std::move(cfg), Acts::Logging::INFO));
  }

  ///////////////////
  // Digitization
  ///////////////////
  {
    const auto volume_id =
        detector
            ->lowestTrackingVolume(
                Acts::GeometryContext{},
                start_surface->center(Acts::GeometryContext{}))
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

    cfg.randomNumbers = rnd;
    cfg.trackingGeometry = detector;

    sequencer.addAlgorithm(
        createDigitizationAlgorithm(cfg, Acts::Logging::INFO));
  }

  ////////////////////
  // Truth Tracking //
  ////////////////////
  {
    ActsExamples::TruthTrackFinder::Config cfg;
    cfg.inputParticles = kGeneratedParticles;
    cfg.inputMeasurementParticlesMap = kMeasurementParticleMap;
    cfg.outputProtoTracks = kProtoTracks;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::TruthTrackFinder>(
        cfg, Acts::Logging::INFO));
  }

  {
    ActsExamples::ParticleSmearing::Config cfg;
    cfg.inputParticles = kGeneratedParticles;
    cfg.outputTrackParameters = kProtoTrackParameters;
    cfg.randomNumbers = rnd;

    sequencer.addAlgorithm(std::make_shared<ActsExamples::ParticleSmearing>(
        cfg, Acts::Logging::INFO));
  }

  ////////////////////////
  // Gaussian Sum Filter
  ////////////////////////
  {
    ActsExamples::TrackFittingAlgorithm::Config cfg;

    cfg.inputMeasurements = kMeasurements;
    cfg.inputSourceLinks = kSourceLinks;
    cfg.inputProtoTracks = kProtoTracks;
    cfg.inputInitialTrackParameters = kProtoTrackParameters;
    cfg.outputTrajectories = kGsfOutputTrajectories;
    cfg.directNavigation = false;
    cfg.trackingGeometry = detector;
    cfg.fit = makeGsfFitterFunction(detector, magField);

    sequencer.addAlgorithm(
        std::make_shared<ActsExamples::TrackFittingAlgorithm>(
            cfg, Acts::Logging::VERBOSE));
  }

  //////////////////////////////////
  // Kalman Fitter for comparison //
  //////////////////////////////////
  {
    ActsExamples::TrackFittingAlgorithm::Config cfg;

    cfg.inputMeasurements = kMeasurements;
    cfg.inputSourceLinks = kSourceLinks;
    cfg.inputProtoTracks = kProtoTracks;
    cfg.inputInitialTrackParameters = kProtoTrackParameters;
    cfg.outputTrajectories = kKalmanOutputTrajectories;
    cfg.directNavigation = false;
    cfg.trackingGeometry = detector;
    cfg.fit = ActsExamples::TrackFittingAlgorithm::makeTrackFitterFunction(
        detector, magField);

    sequencer.addAlgorithm(
        std::make_shared<ActsExamples::TrackFittingAlgorithm>(
            cfg, Acts::Logging::INFO));
  }

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
        std::make_shared<ObjPropStepWriter>(cfg, Acts::Logging::INFO));
  }

  {
    ObjPropStepWriter::Config cfg;
    cfg.outputDir = "";
    cfg.collection = kMultiSteppingLogComponentsLoop;
    cfg.outputStem = "components-steps";

    sequencer.addWriter(
        std::make_shared<ObjPropStepWriter>(cfg, Acts::Logging::INFO));
  }

  {
    ActsExamples::ObjHitWriter::Config cfg;
    cfg.outputDir = "";
    cfg.collection = kSimulatedHits;

    sequencer.addWriter(
        std::make_shared<ActsExamples::ObjHitWriter>(cfg, Acts::Logging::INFO));
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
            cfg, Acts::Logging::INFO));
  }

  {
    ActsExamples::CsvSimHitWriter::Config cfg;
    cfg.inputSimHits = kSimulatedHits;
    cfg.outputDir = "";
    cfg.outputStem = "simulated-hits";

    sequencer.addWriter(std::make_shared<ActsExamples::CsvSimHitWriter>(
        cfg, Acts::Logging::INFO));
  }

  {
    ActsExamples::CsvPropagationStepsWriter::Config cfg;
    cfg.collection = kMultiSteppingLogAverageLoop;
    cfg.outputDir = "";

    sequencer.addWriter(
        std::make_shared<ActsExamples::CsvPropagationStepsWriter>(
            cfg, Acts::Logging::INFO));
  }

  return sequencer.run();
}
