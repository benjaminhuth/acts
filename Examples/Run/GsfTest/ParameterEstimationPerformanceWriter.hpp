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
#include "ActsFatras/EventData/Barcode.hpp"

#include <limits>
#include <string>

#include <dfe/dfe_io_dsv.hpp>
#include <dfe/dfe_namedtuple.hpp>

struct FreeVectorNamedTuple {
  double X, Y, Z, T, DX, DY, DZ, QOP;

  DFE_NAMEDTUPLE(FreeVectorNamedTuple, X, Y, Z, T, DX, DY, DZ, QOP);
};

namespace ActsExamples {

/// Select tracks by applying some selection cuts.
struct ParameterEstimationPerformanceWriter final : public IWriter {
  struct Config {
    /// Input track parameters collection.
    std::string inProtoTrackParameters;
    std::string inMeasurementParticlesMap;
    std::string inProtoTracks;
    std::string inParticles;

    /// Minimum track length
    std::string outputStem = "start-bound-residuals";
    std::string outputDir = "";
    std::size_t outputPrecision = 5;
  } m_cfg;

  std::unique_ptr<const Acts::Logger> m_logger;

  const Acts::Logger& logger() const { return *m_logger; }

  std::string name() const final {
    return "ParameterEstimationPerformanceWriter";
  }

  ParameterEstimationPerformanceWriter(const Config& cfg, Acts::Logging::Level lvl)
      : m_cfg(cfg), m_logger(Acts::getDefaultLogger(name(), lvl)) {
    if (m_cfg.inProtoTracks.empty()) {
      throw std::invalid_argument("Missing inProtoTrack");
    }
    if (m_cfg.inProtoTrackParameters.empty()) {
      throw std::invalid_argument("Missing inProtoTrackParameters");
    }
    if (m_cfg.inParticles.empty()) {
      throw std::invalid_argument("Missing inParticles");
    }
    if (m_cfg.inParticles.empty()) {
      throw std::invalid_argument("Missing inMeasurementParticlesMap");
    }
  }

  ProcessCode write(const AlgorithmContext& ctx) final {
    // Setup writing
    auto pathParticles = perEventFilepath(
        m_cfg.outputDir, m_cfg.outputStem + ".csv", ctx.eventNumber);
    dfe::NamedTupleCsvWriter<FreeVectorNamedTuple> writer(
        pathParticles, m_cfg.outputPrecision);

    using HitParticlesMap = ActsExamples::IndexMultimap<ActsFatras::Barcode>;
    const auto& protoTracks =
        ctx.eventStore.get<ProtoTrackContainer>(m_cfg.inProtoTracks);
    const auto& protoTrackParameters =
        ctx.eventStore.get<TrackParametersContainer>(
            m_cfg.inProtoTrackParameters);
    const auto& particles =
        ctx.eventStore.get<SimParticleContainer>(m_cfg.inParticles);
    const auto& hitParticlesMap =
        ctx.eventStore.get<HitParticlesMap>(m_cfg.inMeasurementParticlesMap);

    for (std::size_t itrack = 0; itrack < protoTracks.size(); ++itrack) {
      const auto& protoTrack = protoTracks[itrack];

      if (protoTrack.empty()) {
        continue;
      }

      // Get true params
      const auto itPair = hitParticlesMap.equal_range(protoTrack.front());
      throw_assert(std::distance(itPair.first, itPair.second) == 1,
                   "there should only be one contributing particle");

      const auto& particle =
          *std::find_if(particles.begin(), particles.end(), [&](const auto& p) {
            return p.particleId() == itPair.first->second;
          });

      // Convert to local coordinates
      const auto trueFree = [&]() {
        Acts::FreeVector free;
        free << particle.fourPosition(), particle.unitDirection(),
            particle.charge() / particle.absoluteMomentum();
        return free;
      }();

      // Get estimated params
      const auto& bound = protoTrackParameters[itrack];
      const auto estimatedFree = Acts::detail::transformBoundToFreeParameters(bound.referenceSurface(), ctx.geoContext, bound.parameters());

      // Make residual
      FreeVectorNamedTuple residuals{
          trueFree[Acts::eFreePos0] - estimatedFree[Acts::eFreePos0],
          trueFree[Acts::eFreePos1] - estimatedFree[Acts::eFreePos1],
          trueFree[Acts::eFreePos2] - estimatedFree[Acts::eFreePos2],
          trueFree[Acts::eFreeTime] - estimatedFree[Acts::eFreeTime],
          trueFree[Acts::eFreeDir0] - estimatedFree[Acts::eFreeDir0],
          trueFree[Acts::eFreeDir1] - estimatedFree[Acts::eFreeDir1],
          trueFree[Acts::eFreeDir2] - estimatedFree[Acts::eFreeDir2],
          trueFree[Acts::eFreeQOverP] - estimatedFree[Acts::eFreeQOverP]
      };

      writer.append(residuals);
    }
    return ActsExamples::ProcessCode::SUCCESS;
  }

  ProcessCode endRun() { return ActsExamples::ProcessCode::SUCCESS; }
};

}  // namespace ActsExamples
