
// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Units.hpp"
#include "Acts/Plugins/ExaTrkX/ExaTrkXPipeline.hpp"
#include "Acts/Plugins/ExaTrkX/Stages.hpp"
#include "Acts/Plugins/ExaTrkX/TorchGraphStoreHook.hpp"
#include "ActsExamples/EventData/Cluster.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimHit.hpp"
#include "ActsExamples/EventData/SimParticle.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/DataHandle.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Traccc/Common/TracccChainAlgorithmBase.hpp"

#include <mutex>
#include <string>
#include <vector>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

namespace ActsExamples {

class GNNPlusTracccAlgorithm final
    : public Traccc::Common::TracccChainAlgorithmBase {
 public:
  struct Config {
    /// Config for base algorithm
    Traccc::Common::TracccChainAlgorithmBase::Config baseConfig;

    /// Input spacepoints collection.
    std::string inputSpacePoints;

    /// Input measurements
    std::string inputMeasurements;

    /// Input measurements
    std::string inputTruthTracks;

    std::shared_ptr<Acts::GraphConstructionBase> graphConstructor;

    std::vector<std::shared_ptr<Acts::EdgeClassificationBase>> edgeClassifiers;

    std::shared_ptr<Acts::TrackBuildingBase> trackBuilder;

    /// Scaling of the input features
    float rScale = 1.f;
    float phiScale = 1.f;
    float zScale = 1.f;

    /// Remove track candidates with 2 or less hits
    bool filterShortTracks = false;
  };

  /// Constructor of the track finding algorithm
  ///
  /// @param cfg is the config struct to configure the algorithm
  /// @param level is the logging level
  GNNPlusTracccAlgorithm(Config cfg, Acts::Logging::Level lvl);

  ~GNNPlusTracccAlgorithm() override = default;

  /// Framework execute method of the track finding algorithm
  ///
  /// @param ctx is the algorithm context that holds event-wise information
  /// @return a process code to steer the algorithm flow
  ActsExamples::ProcessCode execute(
      const ActsExamples::AlgorithmContext& ctx) const final;

  /// Finalize and print timing
  ActsExamples::ProcessCode finalize() final;

  const Config& config() const { return m_cfg; }

 private:
  Config m_cfg;

  Acts::ExaTrkXPipeline m_pipeline;
  mutable std::mutex m_mutex;

  ReadDataHandle<SimSpacePointContainer> m_inputSpacePoints{this,
                                                            "InputSpacePoints"};
  ReadDataHandle<MeasurementContainer> m_inputMeasurements{this,
                                                           "InputMeasurements"};
  ReadDataHandle<ProtoTrackContainer> m_inputProtoTracks{this,
                                                           "InputProtoTracks"};
};

}  // namespace ActsExamples
