// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFindingExaTrkX/GNNPlusTracccAlgorithm.hpp"

#include "Acts/Definitions/Units.hpp"
#include "Acts/Plugins/ExaTrkX/TorchGraphStoreHook.hpp"
#include "Acts/Plugins/ExaTrkX/TorchTruthGraphMetricsHook.hpp"
#include "Acts/Utilities/Zip.hpp"
#include "ActsExamples/EventData/Index.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"

#include <numeric>

using namespace ActsExamples;
using namespace Acts::UnitLiterals;

ActsExamples::GNNPlusTracccAlgorithm::GNNPlusTracccAlgorithm(
    Config config, Acts::Logging::Level level)
    : Traccc::Common::TracccChainAlgorithmBase(config.baseConfig, level),
      m_cfg(std::move(config)),
      m_pipeline(m_cfg.graphConstructor, m_cfg.edgeClassifiers,
                 m_cfg.trackBuilder, logger().clone()) {
  if (m_cfg.inputSpacePoints.empty()) {
    throw std::invalid_argument("Missing spacepoint input collection");
  }
  m_inputProtoTracks.maybeInitialize(m_cfg.inputTruthTracks);
  m_inputSpacePoints.initialize(m_cfg.inputSpacePoints);
}

ActsExamples::ProcessCode ActsExamples::GNNPlusTracccAlgorithm::execute(
    const ActsExamples::AlgorithmContext& ctx) const {

  std::vector<ProtoTrack> trackCandidates;

  if( !m_inputProtoTracks.isInitialized() ) {
    // Read input data
    auto spacepoints = m_inputSpacePoints(ctx);

    // Convert Input data to a list of size [num_measurements x
    // measurement_features]
    const std::size_t numSpacepoints = spacepoints.size();
    const std::size_t numFeatures = 3;
    ACTS_INFO("Received " << numSpacepoints << " spacepoints");

    std::vector<float> features(numSpacepoints * numFeatures);
    std::vector<int> spacepointIDs;

    spacepointIDs.reserve(spacepoints.size());

    for (auto i = 0ul; i < numSpacepoints; ++i) {
      const auto& sp = spacepoints[i];

      // I would prefer to use a std::span or boost::span here once available
      float* featurePtr = features.data() + i * numFeatures;

      // For now just take the first index since does require one single index
      // per spacepoint
      const auto& sl = sp.sourceLinks()[0].template get<IndexSourceLink>();
      spacepointIDs.push_back(sl.index());

      featurePtr[0] = std::hypot(sp.x(), sp.y()) / m_cfg.rScale;
      featurePtr[1] = std::atan2(sp.y(), sp.x()) / m_cfg.phiScale;
      featurePtr[2] = sp.z() / m_cfg.zScale;
    }

    // Run the pipeline
    std::lock_guard<std::mutex> lock(m_mutex);

    auto res = m_pipeline.run(features, spacepointIDs);

    // Make the prototracks
    trackCandidates.reserve(res.size());

    int nShortTracks = 0;

    for (auto& x : res) {
      if (m_cfg.filterShortTracks && x.size() < 3) {
        nShortTracks++;
        continue;
      }

      ProtoTrack onetrack;
      onetrack.reserve(x.size());

      std::copy(x.begin(), x.end(), std::back_inserter(onetrack));
      trackCandidates.push_back(std::move(onetrack));
    }

    ACTS_INFO("Removed " << nShortTracks << " with less then 3 hits");
    ACTS_INFO("Created " << trackCandidates.size() << " proto tracks");
  } else {
    trackCandidates = m_inputProtoTracks(ctx);
  }

  ///
  /// Tracc part
  ///

  ACTS_INFO("Ran the finding algorithm");

  trackStates = fittingAlgorithm(detector, field, trackCandidates);

  ACTS_INFO("Ran the fitting algorithm");

  resolvedTrackStates = ambiguityResolutionAlgorithm(trackStates);

  const auto& actsMeasurements = m_inputMeasurements(ctx);

  auto result = converter.convertTracks(resolvedTrackStates, measurements,
                                        actsMeasurements);
  m_outputTracks(ctx, std::move(result));

  return ActsExamples::ProcessCode::SUCCESS;
}

ActsExamples::ProcessCode GNNPlusTracccAlgorithm::finalize() {}
