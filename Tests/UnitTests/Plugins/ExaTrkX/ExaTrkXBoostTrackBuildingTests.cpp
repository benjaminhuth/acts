// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include "Acts/Plugins/ExaTrkX/BoostTrackBuilding.hpp"
#include "Acts/Plugins/ExaTrkX/detail/BoostTrackBuildingUtils.hpp"
#include "Acts/Plugins/ExaTrkX/detail/TensorVectorConversion.hpp"

#include <algorithm>

#include <boost/graph/adjacency_list.hpp>

BOOST_AUTO_TEST_CASE(test_track_building) {
  // Make some spacepoint IDs
  // The spacepoint ids are [100, 101, 102, ...]
  // They should not be zero based to check if the thing also works if the
  // spacepoint IDs do not match the node IDs used for the edges
  std::vector<int> spacepointIds(16);
  auto nodes = torch::rand({16, 3});

  std::iota(spacepointIds.begin(), spacepointIds.end(), 100);

  // Build 4 tracks with 4 hits
  std::vector<std::vector<int>> refTracks;
  for (auto t = 0ul; t < 4; ++t) {
    refTracks.emplace_back(spacepointIds.begin() + 4 * t,
                           spacepointIds.begin() + 4 * (t + 1));
  }

  // Make edges
  std::vector<int64_t> edges;
  for (const auto &track : refTracks) {
    for (auto it = track.begin(); it != track.end() - 1; ++it) {
      // edges must be 0 based, so subtract 100 again
      edges.push_back(*it - 100);
      edges.push_back(*std::next(it) - 100);
    }
  }

  auto edgeTensor =
      Acts::detail::vectorToTensor2D(edges, 2).t().contiguous().clone();
  auto dummyWeights = torch::ones(edges.size() / 2, torch::kFloat32);

  // Run Track building
  auto logger = Acts::getDefaultLogger("TestLogger", Acts::Logging::ERROR);
  Acts::BoostTrackBuilding trackBuilder({}, std::move(logger));

  auto testTracks =
      trackBuilder(nodes, edgeTensor, dummyWeights, spacepointIds);

  // Sort tracks, so we can find them
  std::for_each(testTracks.begin(), testTracks.end(),
                [](auto &t) { std::sort(t.begin(), t.end()); });
  std::for_each(refTracks.begin(), refTracks.end(),
                [](auto &t) { std::sort(t.begin(), t.end()); });

  // Check what we have here
  for (const auto &refTrack : refTracks) {
    auto found = std::find(testTracks.begin(), testTracks.end(), refTrack);
    BOOST_CHECK(found != testTracks.end());
  }
}

struct EdgeProperty {
  float weight;
};

BOOST_AUTO_TEST_CASE(test_graph_cleaning_no_cleaning) {
  using Graph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                            boost::no_property, EdgeProperty>;

  Graph graph;

  // Add one clean track
  boost::add_edge(0, 1, EdgeProperty{1.0}, graph);
  boost::add_edge(1, 2, EdgeProperty{1.0}, graph);
  boost::add_edge(2, 3, EdgeProperty{1.0}, graph);
  boost::add_edge(3, 4, EdgeProperty{1.0}, graph);

  // Add another clean track, but with one weak edge (should have no effect)
  boost::add_edge(5, 6, EdgeProperty{1.0}, graph);
  boost::add_edge(6, 7, EdgeProperty{0.5}, graph);
  boost::add_edge(7, 8, EdgeProperty{1.0}, graph);
  boost::add_edge(8, 9, EdgeProperty{1.0}, graph);

  std::size_t numEdgesBefore = boost::num_edges(graph);
  std::vector<std::size_t> c(boost::num_vertices(graph));
  std::size_t numConnectedBefore = boost::connected_components(graph, c.data());

  Acts::detail::cleanSubgraphs(graph);

  BOOST_CHECK(numEdgesBefore == boost::num_edges(graph));
  BOOST_CHECK(numConnectedBefore ==
              boost::connected_components(graph, c.data()));
}

BOOST_AUTO_TEST_CASE(test_graph_cleaning_one_branch) {
  using Graph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                            boost::no_property, EdgeProperty>;

  Graph graph;

  // Add one clean track
  boost::add_edge(0, 1, EdgeProperty{1.0}, graph);
  boost::add_edge(1, 2, EdgeProperty{1.0}, graph);
  boost::add_edge(2, 3, EdgeProperty{1.0}, graph);
  boost::add_edge(3, 4, EdgeProperty{1.0}, graph);

  // Add another clean track
  // Should be branched in 5-9 and 10-12
  boost::add_edge(5, 6, EdgeProperty{1.0}, graph);

  boost::add_edge(6, 7, EdgeProperty{1.0}, graph);
  boost::add_edge(7, 8, EdgeProperty{1.0}, graph);
  boost::add_edge(8, 9, EdgeProperty{1.0}, graph);

  boost::add_edge(6, 10, EdgeProperty{0.5}, graph);
  boost::add_edge(10, 11, EdgeProperty{1.0}, graph);
  boost::add_edge(11, 12, EdgeProperty{1.0}, graph);

  std::size_t numEdgesBefore = boost::num_edges(graph);
  std::vector<std::size_t> c(boost::num_vertices(graph));
  std::size_t numConnectedBefore = boost::connected_components(graph, c.data());

  auto logger = Acts::getDefaultLogger("TestLogger", Acts::Logging::VERBOSE);
  Acts::detail::cleanSubgraphs(graph, *logger);

  std::cout << "edges count " << numEdgesBefore << " -> "
            << boost::num_edges(graph) << std::endl;
  std::cout << "connected cmp count " << numConnectedBefore << " -> "
            << boost::connected_components(graph, c.data()) << std::endl;

  BOOST_CHECK(boost::num_edges(graph) == numEdgesBefore - 1);
  BOOST_CHECK(boost::connected_components(graph, c.data()) ==
              numConnectedBefore + 1);
}
