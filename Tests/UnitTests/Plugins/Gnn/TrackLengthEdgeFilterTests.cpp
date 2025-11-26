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
  auto edgeTensor = Tensor<std::int64_t>::Create({2, sources.size()}, execCtx);
  std::copy(sources.begin(), sources.end(), edgeTensor.data());
  std::copy(targets.begin(), targets.end(), edgeTensor.data() + sources.size());
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
  auto edgeTensor =
      createEdgeTensor(inputSources, inputTargets, execContextCpu);

  // Node features: 1D tensor containing only radius values
  auto nodeFeatures =
      Tensor<float>::Create({nodeRadii.size(), 1}, execContextCpu);
  std::copy(nodeRadii.begin(), nodeRadii.end(), nodeFeatures.data());

  // Clone to target device
  auto edgeTensorTarget = edgeTensor.clone(execContext);
  auto nodeFeaturesTarget = nodeFeatures.clone(execContext);

  PipelineTensors input{std::move(nodeFeaturesTarget),
                        std::move(edgeTensorTarget), std::nullopt,
                        std::nullopt};

  auto logger = Acts::getDefaultLogger("TestLogger", Acts::Logging::INFO);
  TrackLengthEdgeFilter filter(cfg, std::move(logger));

  // Apply filter
  auto output = filter(std::move(input), execContext);

  // Clone results back to CPU for verification
  auto outputEdgesHost = output.edgeIndex.clone(execContextCpu);

  // Verify output shape and content
  BOOST_CHECK_EQUAL(outputEdgesHost.shape()[0], 2);
  BOOST_CHECK_EQUAL(outputEdgesHost.shape()[1], expectedSources.size());

  // Check edge indices
  BOOST_CHECK_EQUAL_COLLECTIONS(outputEdgesHost.data(),
                                outputEdgesHost.data() + expectedSources.size(),
                                expectedSources.begin(), expectedSources.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(outputEdgesHost.data() + expectedSources.size(),
                                outputEdgesHost.data() + outputEdgesHost.size(),
                                expectedTargets.begin(), expectedTargets.end());
}

// Test case functions that take only ExecutionContext as parameter
void runTestAllPass(const ExecutionContext& execContext) {
  std::vector<std::int64_t> inputSources = {0, 1, 2};
  std::vector<std::int64_t> inputTargets = {1, 2, 3};
  std::vector<std::int64_t> expectedSources = inputSources;
  std::vector<std::int64_t> expectedTargets = inputTargets;
  std::vector<float> nodeRadii(4, 0.f);

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 0.f;
  cfg.radiusFeatureIdx = 0;

  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, nodeRadii, cfg, execContext);
}

void runTestAllFiltered(const ExecutionContext& execContext) {
  std::vector<std::int64_t> inputSources = {0, 1};
  std::vector<std::int64_t> inputTargets = {1, 2};
  std::vector<std::int64_t> expectedSources, expectedTargets;
  std::vector<float> nodeRadii(3, 0.f);

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 1000.f;
  cfg.radiusFeatureIdx = 0;

  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, nodeRadii, cfg, execContext);
}

void runTestEmptyGraph(const ExecutionContext& execContext) {
  std::vector<std::int64_t> inputSources, inputTargets;
  std::vector<std::int64_t> expectedSources, expectedTargets;
  std::vector<float> nodeRadii;

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 1000.f;
  cfg.radiusFeatureIdx = 0;

  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, nodeRadii, cfg, execContext);
}

void runTestBranching(const ExecutionContext& execContext) {
  std::vector<std::int64_t> inputSources = {0, 1, 2, 1};
  std::vector<std::int64_t> inputTargets = {1, 2, 3, 4};
  std::vector<std::int64_t> expectedSources = {0, 1, 2};
  std::vector<std::int64_t> expectedTargets = {1, 2, 3};
  std::vector<float> nodeRadii(5, 0.f);

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 1000.f;
  cfg.radiusFeatureIdx = 0;

  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, nodeRadii, cfg, execContext);
}

void runTestWithWeights(const ExecutionContext& execContext) {
  std::vector<std::int64_t> inputSources = {0, 1, 2};
  std::vector<std::int64_t> inputTargets = {1, 2, 3};
  std::vector<std::int64_t> expectedSources = inputSources;
  std::vector<std::int64_t> expectedTargets = inputTargets;
  std::vector<float> nodeRadii = {30.f, 30.f, 70.f, 70.f};

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 5;
  cfg.stripRadius = 50.f;
  cfg.radiusFeatureIdx = 0;

  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, nodeRadii, cfg, execContext);
}

void runTestOutgoingBranch(const ExecutionContext& execContext) {
  // Minimal outgoing branch: 0 -> 1 -> 2
  //                               \-> 3
  // Use weights to make one branch longer than the other
  std::vector<std::int64_t> inputSources = {0, 1, 1};
  std::vector<std::int64_t> inputTargets = {1, 2, 3};
  // Node radii: nodes 0,1,3 have weight=1, node 2 has weight=2
  // Path 0->1->2 has accumulated length 4, path 0->1->3 has length 3
  std::vector<std::int64_t> expectedSources = {0, 1};
  std::vector<std::int64_t> expectedTargets = {1, 2};
  std::vector<float> nodeRadii = {10.f, 10.f, 60.f, 10.f};

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 50.f;  // Nodes 0,1,3 have weight 1, node 2 has weight 2
  cfg.radiusFeatureIdx = 0;

  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, nodeRadii, cfg, execContext);
}

void runTestJunction(const ExecutionContext& execContext) {
  // Junction with two incoming and two outgoing edges:
  // 0 -> 2 -> 3
  // 1 -> 2 -> 4
  // Node 2 is the junction
  std::vector<std::int64_t> inputSources = {0, 1, 2, 2};
  std::vector<std::int64_t> inputTargets = {2, 2, 3, 4};
  // All edges have accumulated length 3 with weights=1
  // With minTrackLength=3, all edges pass
  std::vector<std::int64_t> expectedSources = inputSources;
  std::vector<std::int64_t> expectedTargets = inputTargets;
  std::vector<float> nodeRadii(5, 0.f);

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 3;
  cfg.stripRadius = 1000.f;
  cfg.radiusFeatureIdx = 0;

  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, nodeRadii, cfg, execContext);
}

}  // namespace

namespace ActsTests {

BOOST_AUTO_TEST_SUITE(GnnSuite)

const ExecutionContext execContextCpu{Device::Cpu(), {}};

BOOST_AUTO_TEST_CASE(test_track_length_filter_all_pass) {
  runTestAllPass(execContextCpu);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_all_filtered) {
  runTestAllFiltered(execContextCpu);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_empty_graph) {
  runTestEmptyGraph(execContextCpu);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_branching) {
  runTestBranching(execContextCpu);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_with_weights) {
  runTestWithWeights(execContextCpu);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_outgoing_branch) {
  runTestOutgoingBranch(execContextCpu);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_junction) {
  runTestJunction(execContextCpu);
}

#ifdef ACTS_GNN_WITH_CUDA
#include <cuda_runtime_api.h>

// Initialize CUDA runtime and create a stream
static cudaStream_t createCudaStream() {
  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  return stream;
}
static cudaStream_t cudaStream = createCudaStream();

const ExecutionContext execContextCuda{Device::Cuda(0), cudaStream};

BOOST_AUTO_TEST_CASE(test_track_length_filter_all_pass_cuda) {
  runTestAllPass(execContextCuda);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_all_filtered_cuda) {
  runTestAllFiltered(execContextCuda);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_empty_graph_cuda) {
  runTestEmptyGraph(execContextCuda);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_branching_cuda) {
  runTestBranching(execContextCuda);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_with_weights_cuda) {
  runTestWithWeights(execContextCuda);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_outgoing_branch_cuda) {
  runTestOutgoingBranch(execContextCuda);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_junction_cuda) {
  runTestJunction(execContextCuda);
}

#endif  // ACTS_GNN_WITH_CUDA

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
