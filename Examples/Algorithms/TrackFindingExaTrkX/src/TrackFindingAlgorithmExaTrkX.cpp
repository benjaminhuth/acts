// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFindingExaTrkX/TrackFindingAlgorithmExaTrkX.hpp"

#include "Acts/Definitions/Units.hpp"
#include "Acts/Plugins/ExaTrkX/TorchTruthGraphMetricsHook.hpp"
#include "ActsExamples/EventData/Index.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"

#include <torch/torch.h>

using namespace ActsExamples;
using namespace Acts::UnitLiterals;

namespace {

class ExamplesEdmHook : public Acts::PipelineHook {
  constexpr static double m_targetPT = 0.5_GeV;
  constexpr static std::size_t m_targetSize = 3;

  std::unique_ptr<const Acts::Logger> m_logger;
  std::unique_ptr<Acts::TorchTruthGraphMetricsHook> m_truthGraphHook;
  std::unique_ptr<Acts::TorchTruthGraphMetricsHook> m_targetGraphHook;

  const Acts::Logger& logger() const { return *m_logger; }

  struct HitInfo {
    std::size_t spacePointIndex;
    int32_t hitIndex;
  };

 public:
  ExamplesEdmHook(const SimSpacePointContainer& spacepoints,
                  const IndexMultimap<Index>& measHitMap,
                  const SimHitContainer& simhits,
                  const SimParticleContainer& particles,
                  const Acts::Logger& logger)
      : m_logger(logger.clone("MetricsHook")) {
    // Associate tracks to graph, collect momentum
    std::unordered_map<ActsFatras::Barcode, std::vector<HitInfo>> tracks;

    for (auto i = 0ul; i < spacepoints.size(); ++i) {
      const auto measId = spacepoints[i]
                              .sourceLinks()[0]
                              .template get<IndexSourceLink>()
                              .index();

      auto [a, b] = measHitMap.equal_range(measId);
      for (auto it = a; it != b; ++it) {
        const auto& hit = *simhits.nth(it->second);

        tracks[hit.particleId()].push_back({i, hit.index()});
      }
    }

    // Collect edges for truth graph and target graph
    std::vector<int64_t> truthGraph;
    std::vector<int64_t> targetGraph;

    std::size_t notMatched = 0;

    for (auto& [pid, track] : tracks) {
      // Sort by hit index, so the edges are connected correctly
      std::sort(track.begin(), track.end(), [](const auto& a, const auto& b) {
        return a.hitIndex < b.hitIndex;
      });

      for (auto i = 0ul; i < track.size() - 1; ++i) {
        truthGraph.push_back(track[i].spacePointIndex);
        truthGraph.push_back(track[i + 1].spacePointIndex);
      }

      auto found = particles.find(pid);
      if (found == particles.end()) {
        ACTS_VERBOSE("Did not find " << pid << ", cannot add to target graph");
        notMatched++;
        continue;
      }

      if (found->transverseMomentum() > m_targetPT && track.size() >= m_targetSize) {
        for (auto i = 0ul; i < track.size() - 1; ++i) {
          targetGraph.push_back(track[i].spacePointIndex);
          targetGraph.push_back(track[i + 1].spacePointIndex);
        }
      }
    }

    ACTS_DEBUG("Was not able to match " << notMatched << " particles, these might be missing in target graph");

    m_truthGraphHook = std::make_unique<Acts::TorchTruthGraphMetricsHook>(
        truthGraph, logger.clone());
    m_targetGraphHook = std::make_unique<Acts::TorchTruthGraphMetricsHook>(
        targetGraph, logger.clone());
  }

  ~ExamplesEdmHook(){};

  void operator()(const std::any& nodes, const std::any& edges) const override {
    ACTS_INFO("Metrics for total graph:");
    (*m_truthGraphHook)(nodes, edges);
    ACTS_INFO("Metrics for target graph (pT > "
              << m_targetPT / Acts::UnitConstants::GeV
              << " GeV, nHits >= " << m_targetSize << "):");
    (*m_targetGraphHook)(nodes, edges);
  }
};

}  // namespace

ActsExamples::TrackFindingAlgorithmExaTrkX::TrackFindingAlgorithmExaTrkX(
    Config config, Acts::Logging::Level level)
    : ActsExamples::IAlgorithm("TrackFindingMLBasedAlgorithm", level),
      m_cfg(std::move(config)),
      m_pipeline(m_cfg.graphConstructor, m_cfg.edgeClassifiers,
                 m_cfg.trackBuilder, logger().clone()) {
  if (m_cfg.inputSpacePoints.empty()) {
    throw std::invalid_argument("Missing spacepoint input collection");
  }
  if (m_cfg.outputProtoTracks.empty()) {
    throw std::invalid_argument("Missing protoTrack output collection");
  }

  // Sanitizer run with dummy input to detect configuration issues
  // TODO This would be quite helpful I think, but currently it does not work
  // in general because the stages do not expose the number of node features.
  // However, this must be addressed anyways when we also want to allow to
  // configure this more flexible with e.g. cluster information as input. So
  // for now, we disable this.
#if 0
  if( m_cfg.sanitize ) {
  Eigen::VectorXf dummyInput = Eigen::VectorXf::Random(3 * 15);
  std::vector<float> dummyInputVec(dummyInput.data(),
                                   dummyInput.data() + dummyInput.size());
  std::vector<int> spacepointIDs;
  std::iota(spacepointIDs.begin(), spacepointIDs.end(), 0);
  
  runPipeline(dummyInputVec, spacepointIDs);
  }
#endif

  m_inputSpacePoints.initialize(m_cfg.inputSpacePoints);
  m_inputClusters.maybeInitialize(m_cfg.inputClusters);
  m_outputProtoTracks.initialize(m_cfg.outputProtoTracks);

  m_inputSimHits.maybeInitialize(m_cfg.inputSimHits);
  m_inputParticles.maybeInitialize(m_cfg.inputParticles);
  m_inputMeasurementMap.maybeInitialize(m_cfg.inputMeasurementSimhitsMap);
}

enum feat : std::size_t {
  eR,
  ePhi,
  eZ,
  eCellCount,
  eCellSum,
  eClusterX,
  eClusterY
};

ActsExamples::ProcessCode ActsExamples::TrackFindingAlgorithmExaTrkX::execute(
    const ActsExamples::AlgorithmContext& ctx) const {
  // Read input data
  auto spacepoints = m_inputSpacePoints(ctx);
  std::sort(spacepoints.begin(), spacepoints.end(),
            [](const auto& a, const auto& b) {
              return std::hypot(a.x(), a.y()) < std::hypot(b.x(), b.y());
            });

  std::optional<ClusterContainer> clusters;
  if (m_inputClusters.isInitialized()) {
    clusters = m_inputClusters(ctx);
  }

  auto hook = std::make_unique<Acts::PipelineHook>();
  if (m_inputSimHits.isInitialized() && m_inputMeasurementMap.isInitialized()) {
    hook = std::make_unique<ExamplesEdmHook>(
        spacepoints, m_inputMeasurementMap(ctx), m_inputSimHits(ctx), m_inputParticles(ctx), logger());
  }

  // Convert Input data to a list of size [num_measurements x
  // measurement_features]
  const std::size_t numSpacepoints = spacepoints.size();
  const std::size_t numFeatures = clusters ? 7 : 3;
  ACTS_INFO("Received " << numSpacepoints << " spacepoints");

  std::vector<float> features(numSpacepoints * numFeatures);
  std::vector<int> spacepointIDs;

  spacepointIDs.reserve(spacepoints.size());

  double sumCells = 0.0;
  double sumActivation = 0.0;

  for (auto i = 0ul; i < numSpacepoints; ++i) {
    const auto& sp = spacepoints[i];

    float* featurePtr = features.data() + i * numFeatures;

    // For now just take the first index since does require one single index
    // per spacepoint
    const auto& sl = sp.sourceLinks()[0].template get<IndexSourceLink>();
    spacepointIDs.push_back(sl.index());

    featurePtr[eR] = std::hypot(sp.x(), sp.y()) / m_cfg.rScale;
    featurePtr[ePhi] = std::atan2(sp.y(), sp.x()) / m_cfg.phiScale;
    featurePtr[eZ] = sp.z() / m_cfg.zScale;

    if (clusters) {
      const auto& cluster = clusters->at(sl.index());
      const auto& chnls = cluster.channels;

      featurePtr[eCellCount] = cluster.channels.size() / m_cfg.cellCountScale;
      featurePtr[eCellSum] =
          std::accumulate(chnls.begin(), chnls.end(), 0.0,
                          [](double s, const Cluster::Cell& c) {
                            return s + c.activation;
                          }) /
          m_cfg.cellSumScale;
      featurePtr[eClusterX] = cluster.sizeLoc0 / m_cfg.clusterXScale;
      featurePtr[eClusterY] = cluster.sizeLoc1 / m_cfg.clusterYScale;

      sumCells += featurePtr[eCellCount];
      sumActivation += featurePtr[eCellSum];
    }
  }

  ACTS_DEBUG("Avg cell count: " << sumCells / spacepoints.size());
  ACTS_DEBUG("Avg activation: " << sumActivation / sumCells);

  {
    std::stringstream ss;
    std::copy(features.begin(), features.begin() + numFeatures,
              std::ostream_iterator<float>(ss, " "));
    ss << "\n";
    std::copy(features.end() - numFeatures, features.end(),
              std::ostream_iterator<float>(ss, "  "));
    ss << "\n";
    ACTS_DEBUG("First & last row:\n" << ss.str());
  }

  // Run the pipeline
  const auto trackCandidates = m_pipeline.run(features, spacepointIDs, *hook);

  ACTS_DEBUG("Done with pipeline, received " << trackCandidates.size()
                                             << " candidates");

  // Make the prototracks
  std::vector<ProtoTrack> protoTracks;
  protoTracks.reserve(trackCandidates.size());

  int nShortTracks = 0;

  for (auto& x : trackCandidates) {
    if (x.size() < 3) {
      nShortTracks++;
      continue;
    }

    ProtoTrack onetrack;
    onetrack.reserve(x.size());

    std::copy(x.begin(), x.end(), std::back_inserter(onetrack));
    protoTracks.push_back(std::move(onetrack));
  }

  ACTS_INFO("Removed " << nShortTracks << " with less then 3 hits");
  ACTS_INFO("Created " << protoTracks.size() << " proto tracks");
  m_outputProtoTracks(ctx, std::move(protoTracks));

  return ActsExamples::ProcessCode::SUCCESS;
}
