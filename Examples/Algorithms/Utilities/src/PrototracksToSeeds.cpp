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

class LittleDrawer {
  std::size_t m_width = 80;
  std::size_t m_height = 15;
  char m_marker = 'x';

  std::vector<std::vector<int>> m_vals;

 public:
  using SpacePointIt = ActsExamples::SimSpacePointContainer::iterator;

  LittleDrawer(SpacePointIt b, SpacePointIt e)
      : m_vals(m_height, std::vector<int>(m_width, 0)) {
    auto [minRIt, maxRIt] = std::minmax_element(
        b, e, [](const auto &a, const auto &b) { return a.r() < b.r(); });
    auto [minZIt, maxZIt] = std::minmax_element(
        b, e, [](const auto &a, const auto &b) { return a.z() < b.z(); });

    auto minR = minRIt->r();
    auto maxR = maxRIt->r();
    auto minZ = minZIt->z();
    auto maxZ = maxZIt->z();

    maxR += 0.1 * (maxR - minR);
    minR -= 0.1 * (maxR - minR);

    maxZ += 0.1 * (maxZ - minZ);
    minZ -= 0.1 * (maxZ - minZ);

    float factorZ = static_cast<float>(m_width) / (maxZ - minZ);
    float factorR = static_cast<float>(m_height) / (maxR - minR);

    for (auto it = b; it != e; ++it) {
      const ActsExamples::SimSpacePoint &sp = *it;
      float vz = (sp.z() - minZ) * factorZ;
      float vr = (sp.r() - minR) * factorR;

      auto iz = static_cast<std::size_t>(std::round(vz));
      auto ir = static_cast<std::size_t>(std::round(vr));

      assert(iz < m_width && iz >= 0);
      assert(ir < m_height && ir >= 0);

      m_vals[ir][iz]++;
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const LittleDrawer drawer) {
    os << "+" << std::string(drawer.m_width + 2, '-') << "+\n";

    std::string s;
    for (auto rowIt = drawer.m_vals.rbegin(); rowIt != drawer.m_vals.rend();
         ++rowIt) {
      s.clear();
      s.resize(rowIt->size(), ' ');

      for (auto i = 0ul; i < rowIt->size(); ++i) {
        if (rowIt->at(i) > 0 && rowIt->at(i) < 10) {
          s[i] = std::to_string(rowIt->at(i)).front();
        } else if (rowIt->at(i) >= 10) {
          s[i] = 'X';
        }
      }
      os << "| " << s << " |\n";
    }
    os << "+" << std::string(drawer.m_width + 2, '-') << "+";
    return os;
  }
};

using namespace ActsExamples;
using namespace Acts::UnitLiterals;

struct ActsExamples::PrototracksToSeeds::SeedingImpl {
  SimSpacePointContainer m_spacepoints;
  std::unique_ptr<Acts::Logger> m_logger;

  Acts::SeedFinderConfig<SimSpacePoint> m_seedFinderCfg;
  Acts::SeedFinderOptions m_seedFinderOptions;
  Acts::SeedFinder<SimSpacePoint> m_seedFinder;

  std::shared_ptr<const Acts::BinFinder<SimSpacePoint>> m_bottomBinFinder;
  std::shared_ptr<const Acts::BinFinder<SimSpacePoint>> m_topBinFinder;

  Acts::SpacePointGridConfig m_gridCfg;
  Acts::SpacePointGridOptions gridOptions;

  SeedingImpl(const Acts::Logger &logger) : m_logger(logger.clone()) {
    m_seedFinderCfg.deltaRMaxBottomSP = 100_mm;  // should go from layer 0 -> 2
    m_seedFinderCfg.deltaRMinBottomSP = 15_mm;   // should not be in same layer
    m_seedFinderCfg.deltaRMaxTopSP = 120_mm;     // should go from layer 2 -> 3
    m_seedFinderCfg.deltaRMinTopSP = 30_mm;      // should not be in same layer

    Acts::SeedFilterConfig seedFilterCfg;
    seedFilterCfg.deltaRMin = m_seedFinderCfg.deltaRMin;
    seedFilterCfg.maxSeedsPerSpM = 2;

    m_seedFinderCfg.seedFilter =
        std::make_shared<Acts::SeedFilter<SimSpacePoint>>(
            seedFilterCfg.toInternalUnits());

    m_seedFinder =
        Acts::SeedFinder<SimSpacePoint>(m_seedFinderCfg.toInternalUnits());

    m_gridCfg.deltaRMax = m_seedFinderCfg.deltaRMax;
    m_gridCfg.rMax = 200_mm;
    m_gridCfg.zMax = 1500_mm;
    m_gridCfg.zMin = -1500_mm;

    std::vector<std::pair<int, int>> zBinNeighborsBottom;
    std::vector<std::pair<int, int>> zBinNeighborsTop;
    std::size_t numPhiNeighbors = 1;

    m_bottomBinFinder = std::make_shared<const Acts::BinFinder<SimSpacePoint>>(
        zBinNeighborsBottom, numPhiNeighbors);
    m_topBinFinder = std::make_shared<const Acts::BinFinder<SimSpacePoint>>(
        zBinNeighborsTop, numPhiNeighbors);
  }

  const Acts::Logger &logger() const { return *m_logger; }

  void seedPrototrack(const ProtoTrack &track,
                      const SimSpacePointContainer &sps,
                      ProtoTrackContainer &outputTracks,
                      SimSeedContainer &outputSeeds) const {
    // construct the seeding tools
    // covariance tool, extracts covariances per spacepoint as required
    auto extractGlobalQuantities =
        [=](const SimSpacePoint &sp, float, float,
            float) -> std::pair<Acts::Vector3, Acts::Vector2> {
      Acts::Vector3 position{sp.x(), sp.y(), sp.z()};
      Acts::Vector2 covariance{sp.varianceR(), sp.varianceZ()};
      return std::make_pair(position, covariance);
    };

    std::vector<const SimSpacePoint *> spacePointPtrs;
    for (auto hit : track) {
      auto spPtr = findSpacePointForIndex(hit, sps);
      if (spPtr) {
        spacePointPtrs.push_back(spPtr);
      }
    }

    // extent used to store r range for middle spacepoint
    Acts::Extent rRangeSPExtent;

    auto grid = Acts::SpacePointGridCreator::createGrid<SimSpacePoint>(
        m_gridCfg.toInternalUnits(), gridOptions.toInternalUnits());

    auto spacePointsGrouping = Acts::BinnedSPGroup<SimSpacePoint>(
        spacePointPtrs.begin(), spacePointPtrs.end(), extractGlobalQuantities,
        m_bottomBinFinder, m_topBinFinder, std::move(grid), rRangeSPExtent,
        m_seedFinderCfg.toInternalUnits(),
        m_seedFinderOptions.toInternalUnits());

    /// variable middle SP radial region of interest
    const Acts::Range1D<float> rMiddleSPRange(50_mm, 150_mm);

    // run the seeding
    decltype(m_seedFinder)::SeedingState state;
    state.spacePointData.resize(spacePointPtrs.size());

    auto seedsBefore = outputSeeds.size();
    for (const auto [bottom, middle, top] : spacePointsGrouping) {
      m_seedFinder.createSeedsForGroup(m_seedFinderOptions.toInternalUnits(),
                                       state, spacePointsGrouping.grid(),
                                       std::back_inserter(outputSeeds), bottom,
                                       middle, top, rMiddleSPRange);
    }

    for (auto i = seedsBefore; i < outputSeeds.size(); ++i) {
      outputTracks.push_back(track);
    }

    if (outputTracks.size() != outputSeeds.size()) {
      throw std::runtime_error("size mismatch");
    }

    if (seedsBefore == outputSeeds.size()) {
      ACTS_WARNING("No seed created for prototrack");
    } else if (outputSeeds.size() - seedsBefore > 1) {
      ACTS_VERBOSE("created " << outputSeeds.size() - seedsBefore
                            << " seeds for prototrack")
    }
  }
};

namespace {

void naiveSeeding(ProtoTrack track, const SimSpacePointContainer &sps,
                  ProtoTrackContainer &outputTracks,
                  SimSeedContainer &outputSeeds, const Acts::Logger &) {
  // Make prototrack unique with respect to volume and layer
  // So we don't get a seed where we have two spacepoints on the same layer
  auto geoIdFromIndex = [&](auto index) -> Acts::GeometryIdentifier {
    return findSpacePointForIndex(index, sps)
        ->sourceLinks()
        .front()
        .geometryId();
  };

  // Here, we want to create a seed only if the prototrack with removed unique
  // layer-volume spacepoints has 3 or more hits. However, if this is the case,
  // we want to keep the whole prototrack. Therefore, we operate on a tmpTrack.
  std::sort(track.begin(), track.end(), [&](auto a, auto b) {
    auto ga = geoIdFromIndex(a);
    auto gb = geoIdFromIndex(b);
    if( ga.volume() != gb.volume() ) {
      return ga.volume() < gb.volume();
    }
    return ga.layer() < gb.layer();
  });

  ProtoTrack tmpTrack;
  std::unique_copy(track.begin(), track.end(), std::back_inserter(tmpTrack),
                   [&](auto a, auto b) {
                     auto ga = geoIdFromIndex(a);
                     auto gb = geoIdFromIndex(b);
                     return ga.volume() == gb.volume() &&
                            ga.layer() == gb.layer();
                   });

  // in this case we cannot seed properly
  if (tmpTrack.size() < 3) {
    return;
  }

  outputTracks.push_back(track);
  outputSeeds.push_back(prototrackToSeed(tmpTrack, sps));
}

}  // namespace

namespace ActsExamples {

PrototracksToSeeds::PrototracksToSeeds(Config cfg, Acts::Logging::Level lvl)
    : IAlgorithm("PrototracksToSeeds", lvl), m_cfg(std::move(cfg)) {
  m_outputSeeds.initialize(m_cfg.outputSeeds);
  m_outputProtoTracks.initialize(m_cfg.outputProtoTracks);
  m_inputProtoTracks.initialize(m_cfg.inputProtoTracks);
  m_inputSpacePoints.initialize(m_cfg.inputSpacePoints);

  m_advancedSeeding = std::make_unique<SeedingImpl>(logger());
}

PrototracksToSeeds::~PrototracksToSeeds() {}

ProcessCode PrototracksToSeeds::execute(const AlgorithmContext &ctx) const {
  const auto &sps = m_inputSpacePoints(ctx);
  auto prototracks = m_inputProtoTracks(ctx);

  ProtoTrackContainer seededTracks;
  SimSeedContainer seeds;

  auto geoIdFromIndex = [&](auto index) -> Acts::GeometryIdentifier {
    return findSpacePointForIndex(index, sps)
        ->sourceLinks()
        .front()
        .geometryId();
  };

  std::vector<Acts::GeometryIdentifier> tmpGeoIds;
  for (auto &track : prototracks) {
    ACTS_VERBOSE("Try to get seed from prototrack with " << track.size() << " hits");
    if (track.size() == 3 || !m_cfg.advancedSeeding) {
      ACTS_VERBOSE("go directly to naive seeding");
      naiveSeeding(track, sps, seededTracks, seeds, logger());
      continue;
    }
    // Make vector of geometry Ids
    tmpGeoIds.clear();
    std::transform(track.begin(), track.end(), std::back_inserter(tmpGeoIds), [&](auto i){ return geoIdFromIndex(i); });
    std::sort(tmpGeoIds.begin(), tmpGeoIds.end());

    // Go to naive seeding for tracks that are in the endcaps, because there it works quite well
    std::size_t nBarrelHits = std::count_if(tmpGeoIds.begin(), tmpGeoIds.end(), [](auto gid){ return gid.volume() == 17; });
    if( nBarrelHits < track.size() ) {
      ACTS_VERBOSE("go to naive seeding because not all hits in barrel");
      naiveSeeding(track, sps, seededTracks, seeds, logger());
      continue;
    }

    // Check if we have duplicate geoids
    auto it = std::unique(tmpGeoIds.begin(), tmpGeoIds.end());
    bool hasGeoIdDuplicates = (it != tmpGeoIds.end());

    if (hasGeoIdDuplicates) {
      ACTS_VERBOSE("Found GeoId duplicates");
      std::vector<SimSpacePoint> tsps;
      std::transform(track.begin(), track.end(), std::back_inserter(tsps),
                     [&](auto i) { return *findSpacePointForIndex(i, sps); });
      LittleDrawer drawer{tsps.begin(), tsps.end()};
      ACTS_VERBOSE("Advanced seeding for track with " << track.size()
                                                    << " hits");
      m_advancedSeeding->seedPrototrack(track, sps, seededTracks, seeds);
      ACTS_VERBOSE("Visualization:\n" << drawer);
    } else {
      ACTS_VERBOSE("Naive seeding");
      naiveSeeding(track, sps, seededTracks, seeds, logger());
    }

  }

  ACTS_DEBUG("Seeded " << seeds.size() << " out of " << prototracks.size()
                       << " prototracks");

  m_outputSeeds(ctx, std::move(seeds));
  m_outputProtoTracks(ctx, std::move(seededTracks));

  return ProcessCode::SUCCESS;
}

}  // namespace ActsExamples
