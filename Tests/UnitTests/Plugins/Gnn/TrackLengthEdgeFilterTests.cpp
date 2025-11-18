// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "Acts/Utilities/Logger.hpp"
#include "ActsPlugins/Gnn/TrackLengthEdgeFilter.hpp"

#include <algorithm>
#include <vector>

using namespace Acts;
using namespace ActsPlugins;

namespace {

// Helper function to create edge tensor from source and target vectors
Tensor<std::int64_t> createEdgeTensor(const std::vector<std::int64_t>& sources,
                                      const std::vector<std::int64_t>& targets,
                                      const ExecutionContext& execCtx) {
  auto edgeTensor =
      Tensor<std::int64_t>::Create({2, sources.size()}, execCtx);
  std::copy(sources.begin(), sources.end(), edgeTensor.data());
  std::copy(targets.begin(), targets.end(),
            edgeTensor.data() + sources.size());
  return edgeTensor;
}

// Device-agnostic test helper for TrackLengthEdgeFilter
// Creates tensors on CPU, clones to target device, applies filter, and verifies
// nodeRadii: radius value for each node (1D vector)
void testTrackLengthFilter(const std::vector<std::int64_t>& inputSources,
                           const std::vector<std::int64_t>& inputTargets,
                           const std::vector<std::int64_t>& expectedSources,
                           const std::vector<std::int64_t>& expectedTargets,
                           const std::vector<float>& nodeRadii,
                           const TrackLengthEdgeFilter::Config& cfg,
                           ExecutionContext execContext) {
  const ExecutionContext execContextCpu{Device::Cpu(), {}};

  // Create input tensors on CPU
  auto edgeTensor = createEdgeTensor(inputSources, inputTargets, execContextCpu);

  // Node features: 1D tensor containing only radius values
  auto nodeFeatures =
      Tensor<float>::Create({nodeRadii.size(), 1}, execContextCpu);
  std::copy(nodeRadii.begin(), nodeRadii.end(), nodeFeatures.data());

  // Clone to target device
  auto edgeTensorTarget = edgeTensor.clone(execContext);
  auto nodeFeaturesTarget = nodeFeatures.clone(execContext);

  PipelineTensors input{std::move(nodeFeaturesTarget),
                        std::move(edgeTensorTarget), std::nullopt, std::nullopt};

  auto logger = Acts::getDefaultLogger("TestLogger", Acts::Logging::INFO);
  TrackLengthEdgeFilter filter(cfg, std::move(logger));

  // Apply filter
  auto output = filter(std::move(input));

  // Clone results back to CPU for verification
  auto outputEdgesHost = output.edgeIndex.clone(execContextCpu);

  // Verify output shape and content
  BOOST_CHECK_EQUAL(outputEdgesHost.shape()[0], 2);
  BOOST_CHECK_EQUAL(outputEdgesHost.shape()[1], expectedSources.size());

  // Check edge indices
  BOOST_CHECK_EQUAL_COLLECTIONS(
      outputEdgesHost.data(), outputEdgesHost.data() + expectedSources.size(),
      expectedSources.begin(), expectedSources.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      outputEdgesHost.data() + expectedSources.size(),
      outputEdgesHost.data() + outputEdgesHost.size(), expectedTargets.begin(),
      expectedTargets.end());
}

}  // namespace

namespace ActsTests {

BOOST_AUTO_TEST_SUITE(GnnSuite)

BOOST_AUTO_TEST_CASE(test_track_length_filter_all_pass) {
  ExecutionContext execCtx{Device::Cpu(), {}};

  // Single long track: 0 -> 1 -> 2 -> 3 -> 4 -> 5 (6 nodes, minTrackLength=4)
  std::vector<std::int64_t> inputSources = {0, 1, 2, 3, 4};
  std::vector<std::int64_t> inputTargets = {1, 2, 3, 4, 5};

  // All edges should pass
  std::vector<std::int64_t> expectedSources = inputSources;
  std::vector<std::int64_t> expectedTargets = inputTargets;

  std::vector<float> nodeRadii(6, 0.f);  // 6 nodes, all with radius 0

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 0.f;
  cfg.radiusFeatureIdx = 0;

  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, nodeRadii, cfg, execCtx);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_all_filtered) {
  ExecutionContext execCtx{Device::Cpu(), {}};

  // Short track: 0 -> 1 -> 2 (3 nodes, less than minTrackLength=4)
  std::vector<std::int64_t> inputSources = {0, 1};
  std::vector<std::int64_t> inputTargets = {1, 2};

  // All edges should be filtered
  std::vector<std::int64_t> expectedSources, expectedTargets;

  std::vector<float> nodeRadii(3, 0.f);  // 3 nodes, all with radius 0

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 1000.f; // all weights should be one
  cfg.radiusFeatureIdx = 0;

  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, nodeRadii, cfg, execCtx);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_empty_graph) {
  ExecutionContext execCtx{Device::Cpu(), {}};

  // Empty graph
  std::vector<std::int64_t> inputSources, inputTargets;
  std::vector<std::int64_t> expectedSources, expectedTargets;
  std::vector<float> nodeRadii;

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 1000.f; // all weights should be one
  cfg.radiusFeatureIdx = 0;

  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, nodeRadii, cfg, execCtx);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_branching) {
  ExecutionContext execCtx{Device::Cpu(), {}};

  // Test branching behavior: edges on long branches kept, short branches filtered
  // Graph: 0 -> 1 -> 2 -> 3 (main path, 4 nodes)
  //           \-> 4 (short branch, only 3 nodes from 0)
  // With minTrackLength=4, keep main path edges, filter short branch

  std::vector<std::int64_t> inputSources = {0, 1, 2, 1};  // Last edge is the branch
  std::vector<std::int64_t> inputTargets = {1, 2, 3, 4};

  // Expected: keep main path (0->1, 1->2, 2->3), filter branch (1->4)
  std::vector<std::int64_t> expectedSources = {0, 1, 2};
  std::vector<std::int64_t> expectedTargets = {1, 2, 3};

  std::vector<float> nodeRadii(5, 0.f);  // 5 nodes, all with radius 0

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 1000.f; // all weights should be one
  cfg.radiusFeatureIdx = 0;

  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, nodeRadii, cfg, execCtx);
}


BOOST_AUTO_TEST_CASE(test_track_length_filter_with_weights) {
  ExecutionContext execCtx{Device::Cpu(), {}};

  // Track with mixed pixel (weight=1) and strip (weight=2) detectors
  // 0 -> 1 -> 2 -> 3
  // With stripRadius=50, nodes with radius < 50 get weight 1, others get weight 2
  // Nodes 0,1 are pixels (r=30), nodes 2,3 are strips (r=70)
  // Weighted track length = 1 + 1 + 2 + 2 = 6, passes minTrackLength=5

  std::vector<std::int64_t> inputSources = {0, 1, 2};
  std::vector<std::int64_t> inputTargets = {1, 2, 3};

  // All edges should be kept with weighted nodes
  std::vector<std::int64_t> expectedSources = inputSources;
  std::vector<std::int64_t> expectedTargets = inputTargets;

  // Radius values: nodes 0,1 are pixels (r=30), nodes 2,3 are strips (r=70)
  std::vector<float> nodeRadii = {30.f, 30.f, 70.f, 70.f};

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 5;  // Require weighted length >= 5
  cfg.stripRadius = 50.f;  // Distinguish pixel vs strip
  cfg.radiusFeatureIdx = 0;

  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, nodeRadii, cfg, execCtx);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
