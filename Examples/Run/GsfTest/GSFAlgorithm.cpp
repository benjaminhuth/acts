#include "GSFAlgorithm.hpp"

#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/TrackFitting/detail/VoidKalmanComponents.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/SimHit.hpp"
#include "ActsExamples/EventData/SimParticle.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"

#include "GsfActor.hpp"
#include "MultiEigenStepperLoop.hpp"
#include "MultiEigenStepperSIMD.hpp"
#include "MultiSteppingLogger.hpp"
#include "NewGenericDefaultExtension.hpp"
#include "NewStepperExtensionList.hpp"
#include "TestHelpers.hpp"

// constexpr int N = 4;

ActsExamples::ProcessCode GSFAlgorithm::execute(
    const ActsExamples::AlgorithmContext &ctx) const {
  // A logger
  const Acts::LoggerWrapper logger{ActsExamples::BareAlgorithm::logger()};

  // Extract the events
  const auto hits =
      ctx.eventStore.get<ActsExamples::SimHitContainer>(m_cfg.inSimulatedHits);
  const auto finalParticles =
      ctx.eventStore.get<ActsExamples::SimParticleContainer>(
          m_cfg.inSimulatedParticlesFinal);
  const auto initialParticles =
      ctx.eventStore.get<ActsExamples::SimParticleContainer>(
          m_cfg.inSimulatedParticlesInitial);
  const auto measurements =
      ctx.eventStore.get<ActsExamples::MeasurementContainer>(
          m_cfg.inMeasurements);
  const auto sourceLinks =
      ctx.eventStore.get<ActsExamples::IndexSourceLinkContainer>(
          m_cfg.inSourceLinks);

  // Make the GSF options
  Acts::GSFOptions<ActsExamples::MeasurementCalibrator, Acts::VoidOutlierFinder>
      gsfOptions{{measurements},
                 {},
                 ctx.geoContext,
                 ctx.magFieldContext,
                 Acts::LoggerWrapper{ActsExamples::BareAlgorithm::logger()}};

  // Find intersection of final and initial particles
  std::vector<ActsFatras::Particle> particles;
  std::set_intersection(initialParticles.begin(), initialParticles.end(),
                        finalParticles.begin(), finalParticles.end(),
                        std::back_inserter(particles));

  // Filter electrons
  particles.erase(std::remove_if(particles.begin(), particles.end(),
                                 [](const auto &p) {
                                   return p.pdg() !=
                                          Acts::PdgParticle::eElectron;
                                 }),
                  particles.end());

  // Print what remains
  ACTS_VERBOSE("Found " << particles.size()
                        << " particles remaining filtering");
  ACTS_VERBOSE("Found " << hits.size() << " hits in the data");
  ACTS_VERBOSE("Found " << sourceLinks.size() << " source-links in the data");

  for (const auto &hit : hits) {
    ACTS_VERBOSE("Hit with pos " << hit.position().transpose() << " on surface "
                                 << hit.geometryId() << ", "
                                 << hit.geometryId().value());
  }

  for (const auto &srclnk : sourceLinks) {
    ACTS_VERBOSE("Source link on surface " << srclnk.geometryId() << ", "
                                           << srclnk.geometryId().value());
  }

  for (const auto &particle : particles) {
    Acts::FreeVector freePars;
    freePars << particle.fourPosition(), particle.unitDirection(),
        particle.charge() / particle.absoluteMomentum();

    auto boundPars = Acts::detail::transformFreeToBoundParameters(
        freePars, *m_cfg.startSurface, ctx.geoContext);

    if (!boundPars.ok()) {
      ACTS_ERROR("Particle " << particle << " is not on the start surface");
      return ActsExamples::ProcessCode::ABORT;
    }

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

      auto result = gsf.fit(std::vector(sourceLinks.begin(), sourceLinks.end()),
                            multi_pars, gsfOptions);

      if (!result.ok())
        return ActsExamples::ProcessCode::ABORT;

      auto stepLog = result.value().get<MultiSteppingLogger::result_type>();

      ACTS_VERBOSE("Add MultiSteppingLogger results to whiteboard");

      using StepLogContainer = decltype(stepLog.steps);
      ctx.eventStore.add(m_cfg.outMultiStepLogAverage,
                         StepLogContainer{std::move(stepLog.averaged_steps)});
      ctx.eventStore.add(m_cfg.outMultiStepLogComponents,
                         std::move(stepLog.steps));
    }

    //////////////////////////
    // SIMD Stepper
    //////////////////////////
#if 0
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
#endif
  }

  return ActsExamples::ProcessCode::SUCCESS;
}
