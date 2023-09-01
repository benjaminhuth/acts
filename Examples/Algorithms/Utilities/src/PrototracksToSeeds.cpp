// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Utilities/PrototracksToSeeds.hpp"

#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/Utilities/EventDataTransforms.hpp"

#include <algorithm>

namespace ActsExamples {

PrototracksToSeeds::PrototracksToSeeds(Config cfg, Acts::Logging::Level lvl)
    : IAlgorithm("PrototracksToSeeds", lvl), m_cfg(std::move(cfg)) {
  m_outputSeeds.initialize(m_cfg.outputSeeds);
  m_outputProtoTracks.initialize(m_cfg.outputProtoTracks);
  m_inputProtoTracks.initialize(m_cfg.inputProtoTracks);
  m_inputSpacePoints.initialize(m_cfg.inputSpacePoints);
}

ProcessCode PrototracksToSeeds::execute(const AlgorithmContext& ctx) const {
  const auto& sps = m_inputSpacePoints(ctx);
  const auto& prototracks = m_inputProtoTracks(ctx);

  // Make prototrack unique with respect to volume and layer
  // So we don't get a seed where we have two spacepoints on the same layer
  auto geoIdFromIndex = [&](auto index) -> Acts::GeometryIdentifier {
    return findSpacePointForIndex(index, sps)
        ->sourceLinks()
        .front()
        .geometryId();
  };

  SimSeedContainer seeds;
  seeds.reserve(prototracks.size());
  ProtoTrackContainer seededTracks;
  seededTracks.reserve(prototracks.size());

  // Here, we want to create a seed only if the prototrack with removed unique
  // layer-volume spacepoints has 3 or more hits. However, if this is the case,
  // we want to keep the whole prototrack. Therefore, we operate on a tmpTrack.
  ProtoTrack tmpTrack;
  for (const auto& track : prototracks) {
    tmpTrack.clear();
    std::unique_copy(track.begin(), track.end(), std::back_inserter(tmpTrack),
                     [&](auto a, auto b) {
                       auto ga = geoIdFromIndex(a);
                       auto gb = geoIdFromIndex(b);
                       return ga.volume() == gb.volume() &&
                              ga.layer() == gb.layer();
                     });

    if (tmpTrack.size() < 3) {
      continue;
    }

    seededTracks.push_back(track);
    seeds.push_back(prototrackToSeed(tmpTrack, sps));
  }

  ACTS_DEBUG("Seeded " << seeds.size() << " out of " << prototracks.size()
                       << " prototracks");

  m_outputSeeds(ctx, std::move(seeds));
  m_outputProtoTracks(ctx, std::move(seededTracks));

  return ProcessCode::SUCCESS;
}

}  // namespace ActsExamples
