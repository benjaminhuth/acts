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

}  // namespace

namespace ActsTests {

BOOST_AUTO_TEST_SUITE(GnnSuite)

BOOST_AUTO_TEST_CASE(test_track_length_filter_basic) {
  ExecutionContext execCtx{Device::Cpu(), {}};

  // Create a test graph with multiple track scenarios:
  // Track 1 (long): 0 -> 1 -> 2 -> 3 (4 nodes, length=4, KEEP)
  // Track 2 (short): 4 -> 5 -> 6 (3 nodes, length=3, FILTER)
  // Track 3 (branching): 7 -> 8 -> 9 (3 nodes)
  //                           \-> 10 (making total length from 7 = 4, KEEP)

  std::vector<std::int64_t> sources = {0, 1, 2,  // Track 1
                                       4, 5,      // Track 2
                                       7, 8, 8};  // Track 3 (branching)
  std::vector<std::int64_t> targets = {1, 2, 3,  // Track 1
                                       5, 6,      // Track 2
                                       8, 9, 10}; // Track 3 (branching)

  auto edgeTensor = createEdgeTensor(sources, targets, execCtx);

  // Create node features (11 nodes with radius feature)
  // For this test, we use stripRadius=0, so all nodes have weight 1
  auto nodeFeatures = Tensor<float>::Create({11, 3}, execCtx);
  std::fill(nodeFeatures.data(), nodeFeatures.data() + nodeFeatures.size(),
            0.f);

  // Create edge features to verify they are filtered correctly
  auto edgeFeatures = Tensor<float>::Create({sources.size(), 2}, execCtx);
  for (std::size_t i = 0; i < sources.size(); ++i) {
    edgeFeatures.data()[i * 2] = static_cast<float>(i);
    edgeFeatures.data()[i * 2 + 1] = static_cast<float>(i + 1);
  }

  PipelineTensors input{std::move(nodeFeatures), std::move(edgeTensor),
                        std::move(edgeFeatures), std::nullopt};

  // Configure filter with minTrackLength=4
  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 0.f;  // All nodes have weight 1

  auto logger = getDefaultLogger("TestLogger", Logging::ERROR);
  TrackLengthEdgeFilter filter(cfg, std::move(logger));

  // Apply filter
  auto output = filter(std::move(input));

  // Expected to keep: edges from Track 1 (0->1, 1->2, 2->3)
  // and Track 3 (7->8, 8->9, 8->10)
  // Expected to filter: edges from Track 2 (4->5, 5->6)
  std::vector<std::int64_t> expectedSources = {0, 1, 2, 7, 8, 8};
  std::vector<std::int64_t> expectedTargets = {1, 2, 3, 8, 9, 10};

  BOOST_CHECK_EQUAL(output.edgeIndex.shape()[0], 2);
  BOOST_CHECK_EQUAL(output.edgeIndex.shape()[1], expectedSources.size());

  // Check edge indices
  BOOST_CHECK_EQUAL_COLLECTIONS(
      output.edgeIndex.data(),
      output.edgeIndex.data() + expectedSources.size(),
      expectedSources.begin(), expectedSources.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      output.edgeIndex.data() + expectedSources.size(),
      output.edgeIndex.data() + output.edgeIndex.size(),
      expectedTargets.begin(), expectedTargets.end());

  // Verify edge features are filtered correctly
  BOOST_REQUIRE(output.edgeFeatures.has_value());
  BOOST_CHECK_EQUAL(output.edgeFeatures->shape()[0], expectedSources.size());
  BOOST_CHECK_EQUAL(output.edgeFeatures->shape()[1], 2);

  // Check that the edge features correspond to the kept edges
  // Kept edges are at indices: 0, 1, 2 (Track 1) and 5, 6, 7 (Track 3)
  std::vector<std::size_t> keptIndices = {0, 1, 2, 5, 6, 7};
  for (std::size_t i = 0; i < keptIndices.size(); ++i) {
    std::size_t origIdx = keptIndices[i];
    BOOST_CHECK_EQUAL(output.edgeFeatures->data()[i * 2],
                      static_cast<float>(origIdx));
    BOOST_CHECK_EQUAL(output.edgeFeatures->data()[i * 2 + 1],
                      static_cast<float>(origIdx + 1));
  }
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_empty_graph) {
  ExecutionContext execCtx{Device::Cpu(), {}};

  // Empty graph
  std::vector<std::int64_t> sources, targets;
  auto edgeTensor = createEdgeTensor(sources, targets, execCtx);
  auto nodeFeatures = Tensor<float>::Create({0, 3}, execCtx);

  PipelineTensors input{std::move(nodeFeatures), std::move(edgeTensor),
                        std::nullopt, std::nullopt};

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;

  auto logger = getDefaultLogger("TestLogger", Logging::ERROR);
  TrackLengthEdgeFilter filter(cfg, std::move(logger));

  auto output = filter(std::move(input));

  // Should return empty edge tensor
  BOOST_CHECK_EQUAL(output.edgeIndex.shape()[0], 2);
  BOOST_CHECK_EQUAL(output.edgeIndex.shape()[1], 0);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_all_pass) {
  ExecutionContext execCtx{Device::Cpu(), {}};

  // Single long track: 0 -> 1 -> 2 -> 3 -> 4 -> 5 (6 nodes, minTrackLength=4)
  std::vector<std::int64_t> sources = {0, 1, 2, 3, 4};
  std::vector<std::int64_t> targets = {1, 2, 3, 4, 5};

  auto edgeTensor = createEdgeTensor(sources, targets, execCtx);
  auto nodeFeatures = Tensor<float>::Create({6, 3}, execCtx);
  std::fill(nodeFeatures.data(), nodeFeatures.data() + nodeFeatures.size(),
            0.f);

  PipelineTensors input{std::move(nodeFeatures), std::move(edgeTensor),
                        std::nullopt, std::nullopt};

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 0.f;

  auto logger = getDefaultLogger("TestLogger", Logging::ERROR);
  TrackLengthEdgeFilter filter(cfg, std::move(logger));

  auto output = filter(std::move(input));

  // All edges should pass
  BOOST_CHECK_EQUAL(output.edgeIndex.shape()[1], sources.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(output.edgeIndex.data(),
                                output.edgeIndex.data() + sources.size(),
                                sources.begin(), sources.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(
      output.edgeIndex.data() + sources.size(),
      output.edgeIndex.data() + output.edgeIndex.size(), targets.begin(),
      targets.end());
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_all_filtered) {
  ExecutionContext execCtx{Device::Cpu(), {}};

  // Short track: 0 -> 1 -> 2 (3 nodes, less than minTrackLength=4)
  std::vector<std::int64_t> sources = {0, 1};
  std::vector<std::int64_t> targets = {1, 2};

  auto edgeTensor = createEdgeTensor(sources, targets, execCtx);
  auto nodeFeatures = Tensor<float>::Create({3, 3}, execCtx);
  std::fill(nodeFeatures.data(), nodeFeatures.data() + nodeFeatures.size(),
            0.f);

  PipelineTensors input{std::move(nodeFeatures), std::move(edgeTensor),
                        std::nullopt, std::nullopt};

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 4;
  cfg.stripRadius = 0.f;

  auto logger = getDefaultLogger("TestLogger", Logging::ERROR);
  TrackLengthEdgeFilter filter(cfg, std::move(logger));

  auto output = filter(std::move(input));

  // All edges should be filtered
  BOOST_CHECK_EQUAL(output.edgeIndex.shape()[1], 0);
}

BOOST_AUTO_TEST_CASE(test_track_length_filter_with_weights) {
  ExecutionContext execCtx{Device::Cpu(), {}};

  // Track with mixed pixel (weight=1) and strip (weight=2) detectors
  // 0 -> 1 -> 2 -> 3
  // With stripRadius=50, nodes with radius < 50 get weight 1, others get weight 2
  // Let's say nodes 0,1 are pixels (r=30), nodes 2,3 are strips (r=70)
  // Track length = 1 + 1 + 2 + 2 = 6 (accumulated through edges)

  std::vector<std::int64_t> sources = {0, 1, 2};
  std::vector<std::int64_t> targets = {1, 2, 3};

  auto edgeTensor = createEdgeTensor(sources, targets, execCtx);

  // Node features with radius at index 0
  auto nodeFeatures = Tensor<float>::Create({4, 3}, execCtx);
  std::fill(nodeFeatures.data(), nodeFeatures.data() + nodeFeatures.size(),
            0.f);

  // Set radius values: nodes 0,1 are pixels (r=30), nodes 2,3 are strips (r=70)
  nodeFeatures.data()[0 * 3 + 0] = 30.f;  // Node 0
  nodeFeatures.data()[1 * 3 + 0] = 30.f;  // Node 1
  nodeFeatures.data()[2 * 3 + 0] = 70.f;  // Node 2
  nodeFeatures.data()[3 * 3 + 0] = 70.f;  // Node 3

  PipelineTensors input{std::move(nodeFeatures), std::move(edgeTensor),
                        std::nullopt, std::nullopt};

  TrackLengthEdgeFilter::Config cfg;
  cfg.minTrackLength = 5;  // Require weighted length >= 5
  cfg.stripRadius = 50.f;  // Distinguish pixel vs strip
  cfg.radiusFeatureIdx = 0;

  auto logger = getDefaultLogger("TestLogger", Logging::ERROR);
  TrackLengthEdgeFilter filter(cfg, std::move(logger));

  auto output = filter(std::move(input));

  // With weighted nodes, the track should have sufficient length to pass
  // All edges should be kept
  BOOST_CHECK_EQUAL(output.edgeIndex.shape()[1], sources.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(output.edgeIndex.data(),
                                output.edgeIndex.data() + sources.size(),
                                sources.begin(), sources.end());
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace ActsTests
