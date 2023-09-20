// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/ExaTrkX/BoostTrackBuilding.hpp"

#include "Acts/Utilities/Zip.hpp"

#include <map>

#include <boost/beast/core/span.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <torch/torch.h>

using namespace torch::indexing;

namespace {
template <bool directed, typename vertex_t, typename weight_t>
auto weaklyConnectedComponents(std::size_t numNodes,
                               boost::beast::span<vertex_t>& rowIndices,
                               boost::beast::span<vertex_t>& colIndices,
                               boost::beast::span<weight_t>& edgeWeights,
                               boost::beast::span<float>& nodeRadius,
                               bool ensure2EdgesPerVertex,
                               std::vector<vertex_t>& trackLabels,
                               const Acts::Logger& logger) {
  // Construct Graph
  // using EdgeWeightProperty = boost::property<boost::edge_weight_t, weight_t>;
  struct EdgeProperty {
    weight_t weight;
  };

  using directedTag =
      std::conditional_t<directed, boost::directedS, boost::undirectedS>;

  using Graph =
      boost::adjacency_list<boost::vecS,         // edge list
                            boost::vecS,         // vertex list
                            directedTag,         // directedness
                            boost::no_property,  // property of vertices
                            EdgeProperty         // property of edges
                            >;

  Graph g(numNodes);

  for (const auto [row, col, weight] :
       Acts::zip(rowIndices, colIndices, edgeWeights)) {
    if constexpr (directed) {
      // No operator[] in this type
      const auto rowRadius = *(nodeRadius.begin() + row);
      const auto colRadius = *(nodeRadius.begin() + col);
      if (rowRadius < colRadius) {
        boost::add_edge(row, col, EdgeProperty{weight}, g);
      } else {
        boost::add_edge(col, row, EdgeProperty{weight}, g);
      }
    } else {
      boost::add_edge(row, col, EdgeProperty{weight}, g);
    }
  }

  // Maybe resolve vertices
  if (ensure2EdgesPerVertex) {
    // If we have a directed graph, we can ensure that only one outgoing edge is
    // present. Otherwise, restrict ourselfs to 2 in-out edges per vertex
    constexpr static auto maxOutEdges = directed ? 1 : 2;

    if constexpr (directed) {
      ACTS_DEBUG(
          "Ensure we have at most 1 outgoing edge per vertex in directed "
          "graph");
    } else {
      ACTS_DEBUG(
          "Ensure we have at most 2 edges per vertex in undirected graph");
    }
    ACTS_DEBUG("Before edge removal: Graph has " << boost::num_edges(g));
    for (auto vd : boost::make_iterator_range(vertices(g))) {
      // Even for undirected graphs, boost::graph documentation says we should
      // use the directed API
      if (boost::out_degree(vd, g) <= maxOutEdges) {
        continue;
      }

      // Even for undirected graphs, boost::graph documentation says we should
      // use the directed API
      const auto edgeRange =
          boost::make_iterator_range(boost::out_edges(vd, g));

      weight_t weightCut = 0.f;

      if constexpr (maxOutEdges == 2) {
        // Find second largest weight
        weight_t largest = 0.f;
        weight_t secondLargest = 0.f;
        for (auto ed : edgeRange) {
          const auto w = g[ed].weight;
          if (w > largest) {
            secondLargest = largest;
            largest = w;
          } else if (w > secondLargest) {
            secondLargest = w;
          }
        }
        weightCut = secondLargest;
      } else {
        auto max = std::max_element(edgeRange.begin(), edgeRange.end(),
                                    [&](const auto& a, const auto& b) {
                                      return g[a].weight < g[b].weight;
                                    });
        weightCut = g[*max].weight;
      }

      // Set all other weights to 0
      for (auto ed : edgeRange) {
        if (g[ed].weight < weightCut) {
          g[ed].weight = 0.f;
        }
      }
    }

    // remove edges
    boost::remove_edge_if([&](auto ed) { return g[ed].weight == 0.f; }, g);
    ACTS_DEBUG("After edge removal: Graph has " << boost::num_edges(g));
  }

  return boost::connected_components(g, &trackLabels[0]);
}
}  // namespace

namespace Acts {

std::vector<std::vector<int>> BoostTrackBuilding::operator()(
    std::any nodes, std::any edges, std::any weights,
    std::vector<int>& spacepointIDs, int) {
  ACTS_DEBUG("Start track building");

  // Get nodes
  const auto nodeTensor = std::any_cast<torch::Tensor>(nodes).to(torch::kCPU);
  assert(static_cast<std::size_t>(nodeTensor.size(0)) == spacepointIDs.size());
  // TODO is this clone necessary?
  const auto radiusTensor = nodeTensor.index({Slice{}, 0}).clone();

  // Get edges
  const auto edgeTensor = std::any_cast<torch::Tensor>(edges).to(torch::kCPU);

  // Get weights
  const auto edgeWeightTensor =
      std::any_cast<torch::Tensor>(weights).to(torch::kCPU);

  assert(edgeTensor.size(0) == 2);
  assert(edgeTensor.size(1) == edgeWeightTensor.size(0));

  const auto numSpacepoints = spacepointIDs.size();
  const auto numEdges = static_cast<std::size_t>(edgeWeightTensor.size(0));

  if (numEdges == 0) {
    ACTS_WARNING("No edges remained after edge classification");
    return {};
  }

  using vertex_t = int64_t;
  using weight_t = float;

  boost::beast::span<vertex_t> rowIndices(edgeTensor.data_ptr<vertex_t>(),
                                          numEdges);
  boost::beast::span<vertex_t> colIndices(
      edgeTensor.data_ptr<vertex_t>() + numEdges, numEdges);
  boost::beast::span<weight_t> edgeWeights(edgeWeightTensor.data_ptr<float>(),
                                           numEdges);
  boost::beast::span<float> nodeRadius(radiusTensor.data_ptr<float>(),
                                       radiusTensor.numel());

  std::vector<vertex_t> trackLabels(numSpacepoints);

  const auto numberLabels =
      m_cfg.useDirectedGraph
          ? weaklyConnectedComponents<true, vertex_t, weight_t>(
                numSpacepoints, rowIndices, colIndices, edgeWeights, nodeRadius,
                m_cfg.ensure2EdgesPerVertex, trackLabels, logger())
          : weaklyConnectedComponents<false, vertex_t, weight_t>(
                numSpacepoints, rowIndices, colIndices, edgeWeights, nodeRadius,
                m_cfg.ensure2EdgesPerVertex, trackLabels, logger());

  // Label edges
  ACTS_VERBOSE("Number of track labels: " << trackLabels.size());
  ACTS_VERBOSE("Number of unique track labels: " << [&]() {
    std::vector<vertex_t> sorted(trackLabels);
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    return sorted.size();
  }());

  if (trackLabels.size() == 0) {
    return {};
  }

  std::vector<std::vector<int>> trackCandidates(numberLabels);

  for (const auto [label, id] : Acts::zip(trackLabels, spacepointIDs)) {
    trackCandidates[label].push_back(id);
  }

  return trackCandidates;
}

}  // namespace Acts
