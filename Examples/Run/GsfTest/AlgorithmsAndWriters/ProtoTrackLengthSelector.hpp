// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/Framework/BareAlgorithm.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"

#include <limits>
#include <string>

namespace ActsExamples {

/// Select tracks by applying some selection cuts.
struct ProtoTrackLengthSelector final : public BareAlgorithm {
  struct Config {
    /// Input track parameters collection.
    std::string inProtoTracks;
    std::string outProtoTracks;

    /// Minimum track length
    std::size_t minLength = 3;
  } m_cfg;

  ProtoTrackLengthSelector(const Config& cfg, Acts::Logging::Level lvl)
      : BareAlgorithm("TrackSelector", lvl), m_cfg(cfg) {
    if (m_cfg.inProtoTracks.empty()) {
      throw std::invalid_argument("Missing input track parameters collection");
    }
    if (m_cfg.outProtoTracks.empty()) {
      throw std::invalid_argument("Missing input track parameters collection");
    }
  }

  ProcessCode execute(const AlgorithmContext& ctx) const final {
    const auto& protoTracks =
        ctx.eventStore.get<ProtoTrackContainer>(m_cfg.inProtoTracks);

    ProtoTrackContainer selected;

    int nRemainingTracks = 0;

    for (const auto& track : protoTracks) {
      if (track.size() >= m_cfg.minLength) {
        selected.push_back(track);
        nRemainingTracks++;
      } else {
        selected.push_back(ProtoTrack{});
      }
    }

    ACTS_INFO("After Proto Track Length selection: "
              << nRemainingTracks << " / " << protoTracks.size() << " remaining ("
              << (100. * nRemainingTracks) / protoTracks.size()
              << "%)");

    ctx.eventStore.add(m_cfg.outProtoTracks, std::move(selected));

    return ActsExamples::ProcessCode::SUCCESS;
  }
};

}  // namespace ActsExamples
