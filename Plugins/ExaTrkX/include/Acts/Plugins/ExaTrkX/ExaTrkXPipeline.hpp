// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/ExaTrkX/Stages.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <chrono>
#include <memory>
#include <vector>

#include <boost/container/small_vector.hpp>

namespace Acts {

namespace detail {
std::vector<std::vector<int>> unpackTrackLabels(
    const Acts::Tensor<int> &trackLabels, std::size_t numberLabels,
    const std::vector<int> &spacepointIDs);
}

struct ExaTrkXTiming {
  using Duration = std::chrono::duration<float, std::milli>;

  Duration graphBuildingTime = Duration{0.f};
  boost::container::small_vector<Duration, 3> classifierTimes;
  Duration trackBuildingTime = Duration{0.f};
};

class ExaTrkXHook {
 public:
  virtual ~ExaTrkXHook() = default;
  virtual void operator()(const PipelineTensors & /*tensors*/,
                          const ExecutionContext & /*execCtx*/) const {};
};

class ExaTrkXPipeline {
 public:
  /// Constructor for the GNN pipeline
  ///
  /// @param graphConstructor Graph construction stage
  /// @param edgeClassifiers Edge classification stages
  /// @param trackBuilder Track building stage
  /// @param logger Logger to use
  ExaTrkXPipeline(
      std::shared_ptr<GraphConstructionBase> graphConstructor,
      std::vector<std::shared_ptr<EdgeClassificationBase>> edgeClassifiers,
      std::shared_ptr<TrackBuildingBase> trackBuilder,
      std::unique_ptr<const Acts::Logger> logger);

  /// Run the GNN pipeline from tensor data
  ///
  /// @param nodeFeatures Node features in a 2D tensor
  /// @param moduleIds Module IDs of the features (used for module-map-like
  /// graph construction)
  /// @param execCtx Device and stream information
  /// @param hook Optional hook to run after each stage
  /// @param timing Optional timing object to fill with the execution times
  /// @return {tensor with the track labels, number of track candidates}
  std::pair<Acts::Tensor<int>, std::size_t> run(
      Tensor<float> nodeFeatures,
      std::optional<Tensor<std::uint64_t>> moduleIds,
      const ExecutionContext &execCtx, const ExaTrkXHook &hook = {},
      ExaTrkXTiming *timing = nullptr) const;

  /// Run the GNN pipeline from data in STL containers
  /// @note Data will be copied even if the device is CPU
  ///
  /// @param features Node features in a 1D vector
  /// @param moduleIds Module IDs of the features (used for module-map-like
  /// graph construction)
  /// @param spacepointIDs Spacepoint IDs corresponding to the features
  /// @param device Device to run the pipeline on
  /// @param hook Optional hook to run after each stage
  /// @param timing Optional timing object to fill with the execution times
  /// @return vector of track candidates
  std::vector<std::vector<int>> run(const std::vector<float> &features,
                                    const std::vector<std::uint64_t> &moduleIds,
                                    const std::vector<int> &spacepointIDs,
                                    Acts::Device device,
                                    const ExaTrkXHook &hook = {},
                                    ExaTrkXTiming *timing = nullptr) const;

 private:
  std::unique_ptr<const Acts::Logger> m_logger;

  std::shared_ptr<GraphConstructionBase> m_graphConstructor;
  std::vector<std::shared_ptr<EdgeClassificationBase>> m_edgeClassifiers;
  std::shared_ptr<TrackBuildingBase> m_trackBuilder;

  const Logger &logger() const { return *m_logger; }
};

}  // namespace Acts
