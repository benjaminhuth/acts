// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Utilities/PrototracksToSeeds.hpp"

#include "Acts/Seeding/BinFinder.hpp"
#include "Acts/Seeding/BinnedSPGroup.hpp"
#include "Acts/Seeding/InternalSpacePoint.hpp"
#include "Acts/Seeding/SeedFilter.hpp"
#include "Acts/Seeding/SeedFinder.hpp"
#include "Acts/Seeding/SeedFinderConfig.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimSeed.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/Utilities/EventDataTransforms.hpp"

#include <algorithm>

using namespace ActsExamples;
using namespace Acts::UnitLiterals;

namespace {

std::tuple<ProtoTrackContainer, SimSeedContainer> naiveSeeding(
    const ProtoTrackContainer &prototracks, const SimSpacePointContainer &sps,
    const Acts::Logger &) {
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
  for (const auto &track : prototracks) {
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

  return {seededTracks, seeds};
}

std::tuple<ProtoTrackContainer, SimSeedContainer> realSeeding(
    const ProtoTrackContainer &prototracks, const SimSpacePointContainer &sps,
    const Acts::Logger &logger) {
  SimSeedContainer seeds;
  ProtoTrackContainer seededTracks;

  Acts::SpacePointGridConfig gridCfg;
  Acts::SpacePointGridOptions gridOptions;
  gridOptions.bFieldInZ = 2_T;

  std::vector<std::pair<int, int>> zBinNeighborsBottom;
  std::vector<std::pair<int, int>> zBinNeighborsTop;
  std::size_t numPhiNeighbors = 1;

  auto bottomBinFinder = std::make_shared<const Acts::BinFinder<SimSpacePoint>>(
      zBinNeighborsBottom, numPhiNeighbors);
  auto topBinFinder = std::make_shared<const Acts::BinFinder<SimSpacePoint>>(
      zBinNeighborsTop, numPhiNeighbors);

  Acts::SeedFinderConfig<SimSpacePoint> seedFinderConfig;
  Acts::SeedFinderOptions seedFinderOptions;

  Acts::SeedFinder<SimSpacePoint> seedFinder(seedFinderConfig);

  // construct the seeding tools
  // covariance tool, extracts covariances per spacepoint as required
  auto extractGlobalQuantities =
      [=](const SimSpacePoint &sp, float, float,
          float) -> std::pair<Acts::Vector3, Acts::Vector2> {
    Acts::Vector3 position{sp.x(), sp.y(), sp.z()};
    Acts::Vector2 covariance{sp.varianceR(), sp.varianceZ()};
    return std::make_pair(position, covariance);
  };

  // Make space point pointer vector
  std::vector<const SimSpacePoint *> spacePointPtrs;
  for (const auto &track : prototracks) {
    spacePointPtrs.clear();
    for (auto hit : track) {
      auto spPtr = findSpacePointForIndex(hit, sps);
      if (spPtr) {
        spacePointPtrs.push_back(spPtr);
      }
    }

    // extent used to store r range for middle spacepoint
    Acts::Extent rRangeSPExtent;

    auto grid = Acts::SpacePointGridCreator::createGrid<SimSpacePoint>(
        gridCfg, gridOptions);

    auto spacePointsGrouping = Acts::BinnedSPGroup<SimSpacePoint>(
        spacePointPtrs.begin(), spacePointPtrs.end(), extractGlobalQuantities,
        bottomBinFinder, topBinFinder, std::move(grid), rRangeSPExtent,
        seedFinderConfig, seedFinderOptions);

    // safely clamp double to float
    float up = Acts::clampValue<float>(
        std::floor(rRangeSPExtent.max(Acts::binR) / 2) * 2);

    /// variable middle SP radial region of interest
    const Acts::Range1D<float> rMiddleSPRange(
        std::floor(rRangeSPExtent.min(Acts::binR) / 2) * 2 +
            seedFinderConfig.deltaRMiddleMinSPRange,
        up - seedFinderConfig.deltaRMiddleMaxSPRange);

    // run the seeding
    decltype(seedFinder)::SeedingState state;
    state.spacePointData.resize(spacePointPtrs.size());

    auto seedsBefore = seeds.size();
    for (const auto [bottom, middle, top] : spacePointsGrouping) {
      seedFinder.createSeedsForGroup(
          seedFinderOptions, state, spacePointsGrouping.grid(),
          std::back_inserter(seeds), bottom, middle, top, rMiddleSPRange);
    }
    for (auto i = seedsBefore; i < seeds.size(); ++i) {
      seededTracks.push_back(track);
    }

    if (seedsBefore == seeds.size()) {
      ACTS_WARNING("No seed created für prototrack");
    } else if (seeds.size() - seedsBefore > 1) {
      ACTS_DEBUG("created " << seeds.size() - seedsBefore << " for prototrack")
    }
  }
  return {seededTracks, seeds};
}

}  // namespace

namespace ActsExamples {

PrototracksToSeeds::PrototracksToSeeds(Config cfg, Acts::Logging::Level lvl)
    : IAlgorithm("PrototracksToSeeds", lvl), m_cfg(std::move(cfg)) {
  m_outputSeeds.initialize(m_cfg.outputSeeds);
  m_outputProtoTracks.initialize(m_cfg.outputProtoTracks);
  m_inputProtoTracks.initialize(m_cfg.inputProtoTracks);
  m_inputSpacePoints.initialize(m_cfg.inputSpacePoints);
}

ProcessCode PrototracksToSeeds::execute(const AlgorithmContext &ctx) const {
  const auto &sps = m_inputSpacePoints(ctx);
  const auto &prototracks = m_inputProtoTracks(ctx);

  using SeedFunction = std::tuple<ProtoTrackContainer, SimSeedContainer> (*)(
      const ProtoTrackContainer &, const SimSpacePointContainer &,
      const Acts::Logger &);

  SeedFunction seedFunction =
      m_cfg.advancedSeeding ? realSeeding : naiveSeeding;

  auto [seededTracks, seeds] = seedFunction(prototracks, sps, logger());

  ACTS_DEBUG("Seeded " << seeds.size() << " out of " << prototracks.size()
                       << " prototracks");

  m_outputSeeds(ctx, std::move(seeds));
  m_outputProtoTracks(ctx, std::move(seededTracks));

  return ProcessCode::SUCCESS;
}

}  // namespace ActsExamples
