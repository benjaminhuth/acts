// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/Track.hpp"
#include "ActsExamples/Framework/IWriter.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/Validation/TrackClassification.hpp"
#include "ActsFatras/EventData/Barcode.hpp"

#include <limits>
#include <string>

#include <dfe/dfe_io_dsv.hpp>
#include <dfe/dfe_namedtuple.hpp>

struct BoundVectorNamedTuple {
  double LOC0, LOC1, THETA, PHI, QOP, T;

  DFE_NAMEDTUPLE(BoundVectorNamedTuple, LOC0, LOC1, THETA, PHI, QOP, T);
};

namespace ActsExamples {

/// Select tracks by applying some selection cuts.
struct TrackFittingPerformanceWriterCsv final : public IWriter {
  struct Config {
    /// Input track parameters collection.
    std::string inTrajectories;
    std::string inParticles;
    std::string inMeasurementParticlesMap;

    /// Minimum track length
    std::string outputStem = "fitted-residuals";
    std::string outputDir = "";
    std::size_t outputPrecision = 5;
  } m_cfg;

  std::unique_ptr<const Acts::Logger> m_logger;

  const Acts::Logger& logger() const { return *m_logger; }

  std::string name() const final { return "TrackFittingPerformanceWriterCsv"; }

  TrackFittingPerformanceWriterCsv(const Config& cfg, Acts::Logging::Level lvl)
      : m_cfg(cfg), m_logger(Acts::getDefaultLogger(name(), lvl)) {
    if (m_cfg.inTrajectories.empty()) {
      throw std::invalid_argument("Missing inTrajectories");
    }
    if (m_cfg.inParticles.empty()) {
      throw std::invalid_argument("Missing inParticles");
    }
    if (m_cfg.inMeasurementParticlesMap.empty()) {
      throw std::invalid_argument("Missing inMeasurementParticlesMap");
    }
  }

  ProcessCode write(const AlgorithmContext& ctx) final {
    // Setup writing
    auto pathParticles = perEventFilepath(
        m_cfg.outputDir, m_cfg.outputStem + ".csv", ctx.eventNumber);
    dfe::NamedTupleCsvWriter<BoundVectorNamedTuple> writer(
        pathParticles, m_cfg.outputPrecision);

    // Extract data
    const auto& trajectories =
        ctx.eventStore.get<TrajectoriesContainer>(m_cfg.inTrajectories);
    const auto& particles =
        ctx.eventStore.get<SimParticleContainer>(m_cfg.inParticles);
    using HitParticlesMap = ActsExamples::IndexMultimap<ActsFatras::Barcode>;
    const auto& hitParticlesMap =
        ctx.eventStore.get<HitParticlesMap>(m_cfg.inMeasurementParticlesMap);

    // Loop over all trajectories
    for (size_t itraj = 0; itraj < trajectories.size(); ++itraj) {
      const auto& traj = trajectories[itraj];

      // Check trajectory
      if (traj.empty()) {
        ACTS_WARNING("Empty trajectories object " << itraj);
        continue;
      }

      if (traj.tips().size() > 1) {
        ACTS_ERROR("Track fitting should not result in multiple trajectories.");
        return ProcessCode::ABORT;
      }

      const auto tip = traj.tips().front();

      if (not traj.hasTrackParameters(tip)) {
        ACTS_WARNING("No fitted track parameters.");
        continue;
      }

      // Get true parameters
      const auto& particle = [&]() {
        std::vector<ParticleHitCount> contributingParticles;
        identifyContributingParticles(hitParticlesMap, traj, tip,
                                      contributingParticles);
        throw_assert(contributingParticles.size() == 1, "Need 1 particle here");

        auto found = std::find_if(
            particles.begin(), particles.end(), [&](const auto& p) {
              return p.particleId() == contributingParticles.front().particleId;
            });
        throw_assert(found != particles.end(), "Particle not in List");

        return *found;
      }();

      // Convert to local coordinates
      const auto trueBound = [&]() {
        Acts::FreeVector free;
        free << particle.fourPosition(), particle.unitDirection(),
            particle.charge() / particle.absoluteMomentum();
        auto res = Acts::detail::transformFreeToBoundParameters(
            free, traj.trackParameters(tip).referenceSurface(), ctx.geoContext);
        throw_assert(res.ok(), "Particle must be on surface");
        return *res;
      }();

      // Get fitted parameters
      const auto& fittedBound = traj.trackParameters(tip).parameters();

      // Make residual
      BoundVectorNamedTuple residuals{
          trueBound[Acts::eBoundLoc0] - fittedBound[Acts::eBoundLoc0],
          trueBound[Acts::eBoundLoc1] - fittedBound[Acts::eBoundLoc1],
          trueBound[Acts::eBoundPhi] - fittedBound[Acts::eBoundPhi],
          trueBound[Acts::eBoundTheta] - fittedBound[Acts::eBoundTheta],
          trueBound[Acts::eBoundQOverP] - fittedBound[Acts::eBoundQOverP],
          trueBound[Acts::eBoundTime] - fittedBound[Acts::eBoundTime]};

      writer.append(residuals);
    }

    return ActsExamples::ProcessCode::SUCCESS;
  }

  ProcessCode endRun() { return ActsExamples::ProcessCode::SUCCESS; }
};

}  // namespace ActsExamples
