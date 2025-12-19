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
#include <random>
#include <set>
#include <vector>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>

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

// Test edge feature filtering with mixed results (some edges kept, some
// filtered) Creates a branching graph where some branches are too short and get
// filtered out
void runTestEdgeFeaturesWithFiltering(const ExecutionContext& execContext) {
  //   Graph structure (numbers are node IDs, radii all < stripRadius for pixel
  //   weights):
  //        0 -> 1 -> 2 -> 3 -> 4 (long chain, keeps all edges)
  //        0 -> 5 (short branch, gets filtered)
  //
  //   With minTrackLength = 4 (4 nodes = weight 4):
  //   - Edges in long chain (0->1, 1->2, 2->3, 3->4): KEPT
  //   - Edge in short branch (0->5): FILTERED

  const ExecutionContext execContextCpu{Device::Cpu(), {}};

  std::vector<std::int64_t> inputSources = {0, 1, 2, 3, 0};
  std::vector<std::int64_t> inputTargets = {1, 2, 3, 4, 5};
  std::vector<float> nodeRadii(6, 0.f);  // All pixels

  // Create edge features: 2 features per edge, with unique values
  // Edge 0 (0->1): features [1.0, 2.0]
  // Edge 1 (1->2): features [3.0, 4.0]
  // Edge 2 (2->3): features [5.0, 6.0]
  // Edge 3 (3->4): features [7.0, 8.0]
  // Edge 4 (0->5): features [9.0, 10.0] - this edge will be filtered out
  const std::size_t numEdges = 5;
  const std::size_t numFeatures = 2;
  std::vector<float> edgeFeatureData = {
      1.0f, 2.0f,  // Edge 0 (0->1) - KEPT
      3.0f, 4.0f,  // Edge 1 (1->2) - KEPT
      5.0f, 6.0f,  // Edge 2 (2->3) - KEPT
      7.0f, 8.0f,  // Edge 3 (3->4) - KEPT
      9.0f, 10.0f  // Edge 4 (0->5) - FILTERED
  };

  // Expected results after filtering (only long chain edges)
  std::vector<std::int64_t> expectedSources = {0, 1, 2, 3};
  std::vector<std::int64_t> expectedTargets = {1, 2, 3, 4};
  std::vector<float> expectedEdgeFeatures = {
      1.0f, 2.0f,  // Edge 0 (0->1)
      3.0f, 4.0f,  // Edge 1 (1->2)
      5.0f, 6.0f,  // Edge 2 (2->3)
      7.0f, 8.0f   // Edge 3 (3->4)
  };

  // Create input tensors on CPU
  auto edgeTensor =
      createEdgeTensor(inputSources, inputTargets, execContextCpu);
  auto nodeFeatures =
      Tensor<float>::Create({nodeRadii.size(), 1}, execContextCpu);
  std::copy(nodeRadii.begin(), nodeRadii.end(), nodeFeatures.data());

  auto edgeFeatures =
      Tensor<float>::Create({numEdges, numFeatures}, execContextCpu);
  std::copy(edgeFeatureData.begin(), edgeFeatureData.end(),
            edgeFeatures.data());

  // Clone to target device
  auto edgeTensorTarget = edgeTensor.clone(execContext);
  auto nodeFeaturesTarget = nodeFeatures.clone(execContext);
  auto edgeFeaturesTarget = edgeFeatures.clone(execContext);

  PipelineTensors input{std::move(nodeFeaturesTarget),
                        std::move(edgeTensorTarget),
                        std::move(edgeFeaturesTarget), std::nullopt};

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 1.0f;  // All radii < 1.0, so all nodes are pixels
  cfg.radiusFeatureIdx = 0;

  auto logger = Acts::getDefaultLogger("TestLogger", Acts::Logging::INFO);
  TrackLengthEdgeFilter filter(cfg, std::move(logger));

  // Apply filter
  auto output = filter(std::move(input), execContext);

  // Clone results back to CPU for verification
  auto outputEdgesHost = output.edgeIndex.clone(execContextCpu);
  auto outputEdgeFeaturesHost = output.edgeFeatures->clone(execContextCpu);

  // Verify output shape
  BOOST_CHECK_EQUAL(outputEdgesHost.shape()[0], 2);
  BOOST_CHECK_EQUAL(outputEdgesHost.shape()[1], expectedSources.size());
  BOOST_CHECK_EQUAL(outputEdgeFeaturesHost.shape()[0], expectedSources.size());
  BOOST_CHECK_EQUAL(outputEdgeFeaturesHost.shape()[1], numFeatures);

  // Check edge indices
  BOOST_CHECK_EQUAL_COLLECTIONS(outputEdgesHost.data(),
                                outputEdgesHost.data() + expectedSources.size(),
                                expectedSources.begin(), expectedSources.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(outputEdgesHost.data() + expectedSources.size(),
                                outputEdgesHost.data() + outputEdgesHost.size(),
                                expectedTargets.begin(), expectedTargets.end());

  // Check edge features - verify each value matches
  BOOST_CHECK_EQUAL_COLLECTIONS(
      outputEdgeFeaturesHost.data(),
      outputEdgeFeaturesHost.data() + expectedEdgeFeatures.size(),
      expectedEdgeFeatures.begin(), expectedEdgeFeatures.end());
}

// Test edge feature filtering when all edges are filtered out
void runTestEdgeFeaturesAllFiltered(const ExecutionContext& execContext) {
  // Graph where all edges fail the minimum track length
  // Single edge 0->1 with minTrackLength = 3 (requires 3 nodes, but we only
  // have 2)

  const ExecutionContext execContextCpu{Device::Cpu(), {}};

  std::vector<std::int64_t> inputSources = {0};
  std::vector<std::int64_t> inputTargets = {1};
  std::vector<float> nodeRadii(2, 0.f);  // All pixels

  // Create edge features: 3 features per edge
  const std::size_t numEdges = 1;
  const std::size_t numFeatures = 3;
  std::vector<float> edgeFeatureData = {1.0f, 2.0f, 3.0f};  // Will be filtered

  // Create input tensors on CPU
  auto edgeTensor =
      createEdgeTensor(inputSources, inputTargets, execContextCpu);
  auto nodeFeatures =
      Tensor<float>::Create({nodeRadii.size(), 1}, execContextCpu);
  std::copy(nodeRadii.begin(), nodeRadii.end(), nodeFeatures.data());

  auto edgeFeatures =
      Tensor<float>::Create({numEdges, numFeatures}, execContextCpu);
  std::copy(edgeFeatureData.begin(), edgeFeatureData.end(),
            edgeFeatures.data());

  // Clone to target device
  auto edgeTensorTarget = edgeTensor.clone(execContext);
  auto nodeFeaturesTarget = nodeFeatures.clone(execContext);
  auto edgeFeaturesTarget = edgeFeatures.clone(execContext);

  PipelineTensors input{std::move(nodeFeaturesTarget),
                        std::move(edgeTensorTarget),
                        std::move(edgeFeaturesTarget), std::nullopt};

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 3;  // Requires at least 3 nodes
  cfg.stripRadius = 1.0f;
  cfg.radiusFeatureIdx = 0;

  auto logger = Acts::getDefaultLogger("TestLogger", Acts::Logging::INFO);
  TrackLengthEdgeFilter filter(cfg, std::move(logger));

  // Apply filter
  auto output = filter(std::move(input), execContext);

  // Clone results back to CPU for verification
  auto outputEdgesHost = output.edgeIndex.clone(execContextCpu);
  auto outputEdgeFeaturesHost = output.edgeFeatures->clone(execContextCpu);

  // Verify output is empty
  BOOST_CHECK_EQUAL(outputEdgesHost.shape()[0], 2);
  BOOST_CHECK_EQUAL(outputEdgesHost.shape()[1], 0);         // No edges kept
  BOOST_CHECK_EQUAL(outputEdgeFeaturesHost.shape()[0], 0);  // No edge features
  BOOST_CHECK_EQUAL(outputEdgeFeaturesHost.shape()[1], numFeatures);
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

BOOST_AUTO_TEST_CASE(test_edge_features_with_filtering) {
  runTestEdgeFeaturesWithFiltering(execContextCpu);
}

BOOST_AUTO_TEST_CASE(test_edge_features_all_filtered) {
  runTestEdgeFeaturesAllFiltered(execContextCpu);
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

BOOST_AUTO_TEST_CASE(test_track_length_filter_large_random_dag_cuda) {
  const int n_nodes = 64;
  const int n_edges = 128;
  const int seed = 42;

  // Generate DAG using lower triangular sampling
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> node_dist(0, n_nodes - 1);

  // Sample exactly n_edges unique edges
  std::set<std::pair<int, int>> edge_set;
  while (edge_set.size() < n_edges) {
    int i = node_dist(rng);
    int j = node_dist(rng);
    if (i < j) {
      edge_set.insert({i, j});
    } else if (i > j) {
      edge_set.insert({j, i});
    }
  }

  // Convert to vectors
  std::vector<std::int64_t> inputSources, inputTargets;
  inputSources.reserve(n_edges);
  inputTargets.reserve(n_edges);
  for (const auto& [src, tgt] : edge_set) {
    inputSources.push_back(src);
    inputTargets.push_back(tgt);
  }

  // Assign node radii: inner nodes (0-31) get 25.0, outer nodes (32-63) get
  // 75.0
  std::vector<float> node_radii;
  node_radii.reserve(n_nodes);
  for (int i = 0; i < n_nodes; ++i) {
    node_radii.push_back(i < n_nodes / 2 ? 25.0f : 75.0f);
  }

  // Verify DAG property using boost::topological_sort
  using Graph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS>;
  Graph g(n_nodes);
  for (std::size_t i = 0; i < inputSources.size(); ++i) {
    boost::add_edge(inputSources[i], inputTargets[i], g);
  }

  std::vector<Graph::vertex_descriptor> topo_order;
  BOOST_REQUIRE_NO_THROW(
      boost::topological_sort(g, std::back_inserter(topo_order)));

  // Configuration
  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 5;
  cfg.stripRadius = 50.0f;
  cfg.radiusFeatureIdx = 0;

  // First run on CPU to get expected result
  auto edgeTensor =
      createEdgeTensor(inputSources, inputTargets, execContextCpu);
  auto nodeFeatures =
      Tensor<float>::Create({node_radii.size(), 1}, execContextCpu);
  std::copy(node_radii.begin(), node_radii.end(), nodeFeatures.data());

  PipelineTensors input{nodeFeatures.clone(execContextCpu),
                        edgeTensor.clone(execContextCpu), std::nullopt,
                        std::nullopt};

  auto logger = Acts::getDefaultLogger("TestLogger", Acts::Logging::INFO);
  TrackLengthEdgeFilter filter(cfg, std::move(logger));

  auto cpuOutput = filter(std::move(input), execContextCpu);

  // Extract CPU result as expected output
  const std::size_t nEdgesKept = cpuOutput.edgeIndex.shape()[1];
  std::vector<std::int64_t> expectedSources(
      cpuOutput.edgeIndex.data(), cpuOutput.edgeIndex.data() + nEdgesKept);
  std::vector<std::int64_t> expectedTargets(
      cpuOutput.edgeIndex.data() + nEdgesKept,
      cpuOutput.edgeIndex.data() + 2 * nEdgesKept);

  // Test CUDA against CPU result
  testTrackLengthFilter(inputSources, inputTargets, expectedSources,
                        expectedTargets, node_radii, cfg, execContextCuda);
}

BOOST_AUTO_TEST_CASE(test_edge_features_with_filtering_cuda) {
  runTestEdgeFeaturesWithFiltering(execContextCuda);
}

BOOST_AUTO_TEST_CASE(test_edge_features_all_filtered_cuda) {
  runTestEdgeFeaturesAllFiltered(execContextCuda);
}

#endif  // ACTS_GNN_WITH_CUDA

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
