// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/TrackFitting/detail/VoidKalmanComponents.hpp"
#include "Acts/Utilities/PdgParticle.hpp"
#include "ActsExamples/Digitization/DigitizationConfig.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/SimHit.hpp"
#include "ActsExamples/Fatras/FatrasAlgorithm.hpp"
#include "ActsExamples/Framework/RandomNumbers.hpp"
#include "ActsExamples/Framework/Sequencer.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/Generators/EventGenerator.hpp"
#include "ActsExamples/Generators/MultiplicityGenerators.hpp"
#include "ActsExamples/Generators/ParametricParticleGenerator.hpp"
#include "ActsExamples/Generators/VertexGenerators.hpp"
#include "ActsExamples/Utilities/Options.hpp"
#include "ActsFatras/EventData/Barcode.hpp"

#include <chrono>
#include <iostream>
#include <random>

#include "GsfActor.hpp"
#include "MultiEigenStepperLoop.hpp"
#include "MultiEigenStepperSIMD.hpp"
#include "NewGenericDefaultExtension.hpp"
#include "NewStepperExtensionList.hpp"
#include "TestHelpers.hpp"

constexpr int N = 4;
using namespace Acts::UnitLiterals;

const char *kGeneratedParticles = "particles";
const char *kSimulatedParticlesInitial = "sim-particles-initial";
const char *kSimulatedParticlesFinal = "sim-particles-final";
const char *kSimulatedHits = "sim-hits";
const char *kSourceLinks = "source-links";

const auto logging_level = Acts::Logging::INFO;

/// The gsf algorithm
class GSFAlgorithm : public ActsExamples::BareAlgorithm {
 public:
  struct Config {
    std::shared_ptr<MagneticField> bField;
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeo;
    std::shared_ptr<const Acts::Surface> startSurface;
  };

  GSFAlgorithm(const Config &cfg,
               Acts::Logging::Level level = Acts::Logging::INFO)
      : ActsExamples::BareAlgorithm("GSFAlgorithm", level), m_cfg(cfg) {}

  /// @brief Execute the GSF with the simulated particles
  ActsExamples::ProcessCode execute(
      const ActsExamples::AlgorithmContext &ctx) const final override {
    // A logger
    const auto logger =
        Acts::getDefaultLogger("GsfActorTest", Acts::Logging::VERBOSE);

    // Make the GSF options
    Acts::GSFOptions<Acts::VoidKalmanComponents, Acts::VoidOutlierFinder>
        gsfOptions{{},
                   {},
                   ctx.geoContext,
                   ctx.magFieldContext,
                   Acts::LoggerWrapper{*logger}};

    // Extract the events
    const auto hits =
        ctx.eventStore.get<ActsExamples::SimHitContainer>(kSimulatedHits);
    const auto particles =
        ctx.eventStore.get<ActsExamples::SimParticleContainer>(kSimulatedHits);
    const auto measurements =
        ctx.eventStore.get<ActsExamples::IndexSourceLinkContainer>(
            kSourceLinks);
    const std::vector measurement_vector(measurements.begin(),
                                         measurements.end());

    // Filter out the particles
    std::vector<ActsFatras::Barcode> particleIds;
    std::transform(hits.begin(), hits.end(), std::back_inserter(particleIds),
                   [](const auto &p) { return p.particleId(); });
    std::sort(particleIds.begin(), particleIds.end());
    std::unique(particleIds.begin(), particleIds.end());

    // Dummy measurements for the moment
    for (auto particleId : particleIds) {
      // Extract start parameters
      auto found = std::find_if(
          particles.begin(), particles.end(),
          [=](const auto &p) { return p.particleId() == particleId; });

      Acts::FreeVector freePars;
      freePars << found->fourPosition(), found->unitDirection(),
          found->charge() / found->absoluteMomentum();

      auto boundPars = Acts::detail::transformFreeToBoundParameters(
          freePars, *m_cfg.startSurface, ctx.geoContext);

      // Make MultiComponentTrackParameters
      Acts::MultiComponentBoundTrackParameters<Acts::SinglyCharged> multi_pars(
          m_cfg.startSurface->getSharedPtr(), *boundPars);

      //////////////////////////
      // LOOP Stepper
      //////////////////////////
      {
        using DefaultExt =
            Acts::detail::GenericDefaultExtension<Acts::ActsScalar>;
        using ExtList = Acts::StepperExtensionList<DefaultExt>;
        const auto prop = make_propagator<Acts::MultiEigenStepperLoop<ExtList>>(
            m_cfg.bField, m_cfg.trackingGeo);

        Acts::GaussianSumFitter gsf(std::move(prop));

        auto result = gsf.fit(measurement_vector, multi_pars, gsfOptions);
      }

      //////////////////////////
      // SIMD Stepper
      //////////////////////////
      {
        using SimdScalar = Acts::SimdType<N>;
        using DefaultExt = Acts::detail::NewGenericDefaultExtension<SimdScalar>;
        using ExtList = Acts::NewStepperExtensionList<DefaultExt>;

        const auto prop =
            make_propagator<Acts::MultiEigenStepperSIMD<N, ExtList>>(
                m_cfg.bField, m_cfg.trackingGeo);

        Acts::GaussianSumFitter gsf(std::move(prop));

        auto result = gsf.fit(measurement_vector, multi_pars, gsfOptions);
      }
    }

    return ActsExamples::ProcessCode::SUCCESS;
  }

 private:
  Config m_cfg;
};

int main() {
  // Setup the sequencer
  ActsExamples::Sequencer::Config seqCfg;
  seqCfg.events = 1;
  ActsExamples::Sequencer sequencer(seqCfg);

  // RNG
  ActsExamples::RandomNumbers::Config rndCfg{std::random_device{}()};
  auto rnd = std::make_shared<ActsExamples::RandomNumbers>(rndCfg);

  // MagneticField
  auto magField = std::make_shared<MagneticField>(Acts::Vector3(0.0, 0.0, 2_T));

  // Tracking Geometry
  auto [start_surface, detector] = build_tracking_geometry();

  const auto volume_id =
      detector->highestTrackingVolume()->geometryId().volume();

  /////////////////////
  // Particle gun
  /////////////////////
  {
    ActsExamples::FixedVertexGenerator vertexGen{Acts::Vector4::Zero()};

    ActsExamples::ParametricParticleGenerator::Config pgCfg;
    pgCfg.phiMin = 0.0_degree;
    pgCfg.phiMax = 360.0_degree;
    pgCfg.thetaMin = std::atan(std::exp(-4.0));
    pgCfg.thetaMax = std::atan(std::exp(4.0));
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

    sequencer.addReader(
        std::make_shared<ActsExamples::EventGenerator>(cfg, logging_level));
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

    sequencer.addAlgorithm(std::make_shared<ActsExamples::FatrasAlgorithm>(
        std::move(cfg), logging_level));
  }

  ///////////////////
  // Digitization
  ///////////////////
  {
    using namespace ActsExamples::Options;

    const bool doMerge = false;
    const bool mergeNsigma = false;
    const bool mergeCommonCorner = false;
    const std::vector<int> volumes = {static_cast<int>(volume_id)};
    const std::vector<VariableIntegers> indices = {
        {std::vector<int>{0, 1}}};  // loc0, loc1
    const std::vector<VariableIntegers> types = {
        {std::vector<int>{0, 0}}};  // gauss, gauss
    const std::vector<VariableReals> parameters = {
        {std::vector<double>{10.0, 10.0}}};  // gaussian width

    auto cfg = ActsExamples::DigitizationConfig(
        doMerge, mergeNsigma, mergeCommonCorner, volumes, indices, types,
        parameters,
        Acts::GeometryHierarchyMap<ActsExamples::DigiComponentsConfig>());

    cfg.inputSimHits = kSimulatedHits;
    cfg.outputSourceLinks = kSourceLinks;

    sequencer.addAlgorithm(createDigitizationAlgorithm(cfg, logging_level));
  }

  ////////////////////////
  // Gaussian Sum Filter
  ////////////////////////
  {
    GSFAlgorithm::Config cfg;
    cfg.bField = magField;
    cfg.trackingGeo = detector;
    cfg.startSurface = start_surface;

    sequencer.addAlgorithm(std::make_shared<GSFAlgorithm>(cfg, logging_level));
  }

  return sequencer.run();
}
