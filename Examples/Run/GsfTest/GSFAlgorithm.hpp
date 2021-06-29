#pragma once

#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "ActsExamples/Framework/BareAlgorithm.hpp"

/// The gsf algorithm
class GSFAlgorithm : public ActsExamples::BareAlgorithm {
 public:
  struct Config {
    std::string inSimulatedHits;
    std::string inSimulatedParticlesFinal;
    std::string inSimulatedParticlesInitial;
    std::string inSourceLinks;
    std::shared_ptr<Acts::ConstantBField> bField;
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeo;
    std::shared_ptr<const Acts::Surface> startSurface;
  };

  GSFAlgorithm(const Config &cfg,
               Acts::Logging::Level level = Acts::Logging::INFO)
      : ActsExamples::BareAlgorithm("GSFAlgorithm", level), m_cfg(cfg) {}

  /// @brief Execute the GSF with the simulated particles
  ActsExamples::ProcessCode execute(
      const ActsExamples::AlgorithmContext &ctx) const final override;

 private:
  Config m_cfg;
};
