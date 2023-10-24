// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Utilities/Logger.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/range/iterator_range_core.hpp>

namespace Acts::detail {

struct SubgraphNodePredicate {
  std::size_t subgraphId = 0;
  std::vector<std::size_t> *labels = nullptr;

  template <typename vertex_t>
  bool operator()(const vertex_t &v) const {
    return labels->at(v) == subgraphId;
  }
};

template <typename graph_t>
struct SubgraphEdgePredicate {
  graph_t *g = nullptr;
  template <typename edge_t>
  bool operator()(const edge_t &e) const {
    return (*g)[e].weight > 0.0;
  }
};

template <typename graph_t>
bool isCleanSubgraph(const graph_t &graph) {
  std::size_t nStart = 0;
  std::size_t nStop = 0;

  for (const auto &n : boost::make_iterator_range(boost::vertices(graph))) {
    nStart += (boost::in_degree(n, graph) == 0);
    nStop += (boost::out_degree(n, graph) == 0);
  }

  return nStart == 1 and nStop == 1;
}

template <typename graph_t>
void cleanSubgraphs(graph_t &graph,
                    const Acts::Logger &logger = Acts::getDummyLogger()) {
  SubgraphEdgePredicate<graph_t> edgeFilter{&graph};

  using Subgraph =
      boost::filtered_graph<graph_t, SubgraphEdgePredicate<graph_t>,
                            SubgraphNodePredicate>;

  std::vector<std::size_t> connectedComponentLabels(boost::num_vertices(graph));
  auto nSubgraphs =
      boost::connected_components(graph, connectedComponentLabels.data());

  for (auto i = 0ul; i < nSubgraphs; ++i) {
    SubgraphNodePredicate nodeFilter{i, &connectedComponentLabels};
    Subgraph subgraph(graph, edgeFilter, nodeFilter);

    if (isCleanSubgraph(subgraph)) {
      continue;
    }

    while (true) {
      // Find edge with minium weight (edges with weight 0 should not occur
      // because of edge filter) However, we only check edges which are
      // branching edges
      auto [edgeBegin, edgeEnd] = boost::edges(subgraph);
      float minWeight = std::numeric_limits<float>::max();
      typename Subgraph::edge_iterator minEdgeDesc = edgeEnd;
      for (auto it = edgeBegin; it != edgeEnd; ++it) {
        assert(graph[*it].weight > 0);
        if (boost::out_degree(boost::source(*it, subgraph), subgraph) < 2 &&
            boost::in_degree(boost::target(*it, subgraph), subgraph) < 2) {
          continue;
        }
        if (graph[*it].weight < minWeight) {
          minWeight = graph[*it].weight;
          minEdgeDesc = it;
        }
      }

      // All edges are clean
      if (minEdgeDesc == edgeEnd) {
        break;
      }

      // Set edge to 0 (should effectively remove the edge from the filtered
      // graph)
      ACTS_VERBOSE("Remove edge " << boost::source(*minEdgeDesc, graph) << ", "
                                  << boost::target(*minEdgeDesc, graph));
      graph[*minEdgeDesc].weight = 0.0;
    }
  }

  // Finally remove all edges with weight == 0
  boost::remove_edge_if([&](auto ed) { return graph[ed].weight == 0.f; },
                        graph);
}

}  // namespace Acts::detail
