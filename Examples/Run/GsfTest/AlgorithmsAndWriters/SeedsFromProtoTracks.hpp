// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/BareAlgorithm.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"

#include <limits>
#include <string>

namespace ActsExamples {

/// Select tracks by applying some selection cuts.
struct SeedsFromProtoTracks final : public BareAlgorithm {
  struct Config {
    /// Input track parameters collection.
    std::string inProtoTracks;
    std::string inSpacePoints;
    std::string outProtoTracks;
    std::string outSeedCollection;

    /// Minimum track length
    std::size_t minLength = 3;
  } m_cfg;

  SeedsFromProtoTracks(const Config& cfg, Acts::Logging::Level lvl)
      : BareAlgorithm("SeedsFromProtoTracks", lvl), m_cfg(cfg) {
    if (m_cfg.inProtoTracks.empty()) {
      throw std::invalid_argument("Missing inProtoTracks");
    }
    if (m_cfg.inProtoTracks.empty()) {
      throw std::invalid_argument("Missing inSpacePoints");
    }
    if (m_cfg.outProtoTracks.empty()) {
      throw std::invalid_argument("Missing outProtoTracks");
    }
    if (m_cfg.outProtoTracks.empty()) {
      throw std::invalid_argument("Missing outSeedCollection");
    }
  }

  ProcessCode execute(const AlgorithmContext& ctx) const final {
    const auto& protoTracks =
        ctx.eventStore.get<ProtoTrackContainer>(m_cfg.inProtoTracks);
    const auto& spacePoints =
        ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inSpacePoints);

    ProtoTrackContainer outTracks;
    outTracks.reserve(protoTracks.size());

    SimSeedContainer seeds;
    seeds.reserve(protoTracks.size());

    for (std::size_t itrack = 0; itrack < protoTracks.size(); ++itrack) {
      const auto& protoTrack = protoTracks[itrack];

      if (protoTrack.size() < 3) {
        ACTS_WARNING("Proto track " << itrack << " size is less than 3.");
        continue;
      }

      std::vector<const SimSpacePoint*> spacePointsOnTrack;
      spacePointsOnTrack.reserve(protoTrack.size());

      for (const auto& hitIndex : protoTrack) {
        auto it =
            std::find_if(spacePoints.begin(), spacePoints.end(),
                         [&](const SimSpacePoint& spacePoint) {
                           return (spacePoint.measurementIndex() == hitIndex);
                         });
        throw_assert(it != spacePoints.end(),
                     "need to find a spacepoint for each hit");
        spacePointsOnTrack.push_back(&(*it));
      }

      if (spacePointsOnTrack.size() < 3) {
        ACTS_WARNING("Found less than 3 spacepoints on track" << itrack);
        continue;
      }

      std::sort(spacePointsOnTrack.begin(), spacePointsOnTrack.end(), [](const auto &a, const auto &b){ return a->r() < b->r(); });

      seeds.emplace_back(*spacePointsOnTrack[0], *spacePointsOnTrack[1],
                         *spacePointsOnTrack[2], spacePointsOnTrack[1]->z());
      outTracks.push_back(protoTrack);
    }

    ctx.eventStore.add(m_cfg.outProtoTracks, std::move(outTracks));
    ctx.eventStore.add(m_cfg.outSeedCollection, std::move(seeds));

    return ActsExamples::ProcessCode::SUCCESS;
  }
};

}  // namespace ActsExamples
