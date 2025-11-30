// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Utilities/Logger.hpp"
#include "ActsPlugins/Gnn/Stages.hpp"

#include <memory>

namespace ActsPlugins {

class TrackLengthEdgeFilter final : public EdgeClassificationBase {
 public:
  struct Config {
    /// Radius threshold to distinguish pixel (r < threshold) from strip layers.
    /// Nodes with radius < stripRadius get weight 1 (pixel),
    /// nodes with radius >= stripRadius get weight 2 (strip).
    /// Default 0.f means all nodes are treated as strip layers (weight 2).
    float stripRadius = 0.f;

    /// Index of the radius feature in the node feature tensor.
    /// The radius value is extracted from nodeFeatures[i, radiusFeatureIdx]
    /// for each node i.
    std::size_t radiusFeatureIdx = 0;

    /// Minimum accumulated track length (sum of node weights along path)
    /// required to keep an edge. Edges belonging to tracks shorter than
    /// this threshold are filtered out.
    std::size_t minTrackLength = 7;
  };

  TrackLengthEdgeFilter(const Config& cfg,
                        std::unique_ptr<const Acts::Logger> logger);
  ~TrackLengthEdgeFilter();

  PipelineTensors operator()(PipelineTensors tensors,
                             const ExecutionContext& execContext = {}) override;

  Config config() const { return m_cfg; }

 private:
  /// Filter edges using CPU implementation
  /// @param tensors Pipeline tensors (moved, will be modified)
  /// @param execContext Execution context for device and stream info
  /// @return Modified PipelineTensors with filtered edgeIndex
  /// @throws std::runtime_error if edge features are present (not yet supported)
  PipelineTensors filterEdgesCpu(PipelineTensors&& tensors,
                                 const ExecutionContext& execContext);

  /// Filter edges using CUDA implementation
  /// @param tensors Pipeline tensors (moved, will be modified)
  /// @param execContext Execution context for device and stream info
  /// @return Modified PipelineTensors with filtered edgeIndex
  /// @throws std::runtime_error if CUDA support is not enabled or if edge
  /// features are present (not yet supported)
  PipelineTensors filterEdgesCuda(PipelineTensors&& tensors,
                                  const ExecutionContext& execContext);

  std::unique_ptr<const Acts::Logger> m_logger;
  const auto& logger() const { return *m_logger; }

  Config m_cfg;
};

}  // namespace ActsPlugins
