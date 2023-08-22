// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/ExaTrkX/Pipeline.hpp"
#include "Acts/Plugins/ExaTrkX/Stages.hpp"
#include "ActsExamples/EventData/Cluster.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimParticle.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"

#include <string>
#include <vector>

#include <boost/multi_array.hpp>

class TruthGraph;

namespace ActsExamples {

class TrackFindingAlgorithmExaTrkX final : public IAlgorithm {
 public:
  struct Config {
    /// Input spacepoints collection.
    std::string inputSpacePoints;

    /// Input cluster information (Optional). If given, the following features
    /// are added:
    /// * cell count
    /// * sum cell activations
    /// * cluster size in local x
    /// * cluster size in local y
    std::string inputClusters;

    /// Input particles (Optional). If given together with
    /// measurementParticlesMap, the truth graph is computed and metrics can be
    /// printed.
    std::string inputParticles;

    /// Input measurement simhit map (Optional). If given together with simhits,
    /// the truth graph is computed and metrics can be printed.
    std::string inputMeasurementParticlesMap;

    /// Output protoTracks collection.
    std::string outputProtoTracks;

    std::shared_ptr<Acts::GraphConstructionBase> graphConstructor;

    std::vector<std::shared_ptr<Acts::EdgeClassificationBase>> edgeClassifiers;

    std::shared_ptr<Acts::TrackBuildingBase> trackBuilder;

    /// Scaling of the input features
    float rScale = 1.f;
    float phiScale = 1.f;
    float zScale = 1.f;
    float cellCountScale = 1.f;
    float cellSumScale = 1.f;
    float clusterXScale = 1.f;
    float clusterYScale = 1.f;
  };

  /// Constructor of the track finding algorithm
  ///
  /// @param cfg is the config struct to configure the algorithm
  /// @param level is the logging level
  TrackFindingAlgorithmExaTrkX(Config cfg, Acts::Logging::Level lvl);

  ~TrackFindingAlgorithmExaTrkX() override = default;

  /// Framework execute method of the track finding algorithm
  ///
  /// @param ctx is the algorithm context that holds event-wise information
  /// @return a process code to steer the algorithm flow
  ActsExamples::ProcessCode execute(
      const ActsExamples::AlgorithmContext& ctx) const final;

  const Config& config() const { return m_cfg; }

 private:
  // configuration
  Config m_cfg;

  Acts::Pipeline m_pipeline;

  ReadDataHandle<SimSpacePointContainer> m_inputSpacePoints{this,
                                                            "InputSpacePoints"};
  ReadDataHandle<ClusterContainer> m_inputClusters{this, "InputClusters"};

  ReadDataHandle<SimParticleContainer> m_inputParticles{this, "InputParticles"};
  ReadDataHandle<IndexMultimap<ActsFatras::Barcode>> m_inputMeasurementMap{
      this, "InputMeasurementMap"};

  WriteDataHandle<ProtoTrackContainer> m_outputProtoTracks{this,
                                                           "OutputProtoTracks"};
};

}  // namespace ActsExamples
