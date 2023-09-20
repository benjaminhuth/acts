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

namespace {
template <typename vertex_t, typename weight_t>
auto weaklyConnectedComponents(vertex_t numNodes,
                               boost::beast::span<vertex_t>& rowIndices,
                               boost::beast::span<vertex_t>& colIndices,
                               boost::beast::span<weight_t>& edgeWeights,
                               std::vector<vertex_t>& trackLabels) {
}
}  // namespace

namespace Acts {

std::vector<std::vector<int>> BoostTrackBuilding::operator()(
    std::any, std::any edges, std::any weights, std::vector<int>& spacepointIDs,
    int) {
  ACTS_DEBUG("Start track building");
  const auto edgeTensor = std::any_cast<torch::Tensor>(edges).to(torch::kCPU);
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

  std::vector<vertex_t> trackLabels(numSpacepoints);
  
  // Construct Graph
  // using EdgeWeightProperty = boost::property<boost::edge_weight_t, weight_t>;
  struct EdgeProperty {
    weight_t weight;
  };
  
  using Graph =
      boost::adjacency_list<boost::vecS,         // edge list
                            boost::vecS,         // vertex list
                            boost::undirectedS,  // directedness
                            boost::no_property,  // property of vertices
                            EdgeProperty   // property of edges
                            >;

  Graph g(numSpacepoints);

  for (const auto [row, col, weight] :
       Acts::zip(rowIndices, colIndices, edgeWeights)) {
    boost::add_edge(row, col, EdgeProperty{weight}, g);
  }
  
  // Maybe resolve vertices
  if(m_cfg.ensure2EdgesPerVertex) {
    ACTS_DEBUG("Ensure we have at most 2 edges per vertex");
    ACTS_DEBUG("Before edge removal: Graph has " << boost::num_edges(g));
    for (auto vd : boost::make_iterator_range(vertices(g))) {
      if( boost::degree(vd, g) <= 2 ) {
        continue;
      }
      
      // Even for undirected graphs, boost::graph documentation says we should use the directed API
      const auto edgeRange = boost::make_iterator_range(boost::out_edges(vd, g));
      
      // Find second largest weight
      weight_t largest = 0.f;
      weight_t secondLargest = 0.f;
      for( auto ed : edgeRange) {
        const auto w = g[ed].weight;
        if (w > largest) {
          secondLargest = largest;
          largest = w;
        } else if (w > secondLargest) {
          secondLargest = w;
        }
      }

      // Set all other weights to 0
      for( auto ed : edgeRange) {
        if(g[ed].weight < secondLargest) {
          g[ed].weight = 0.f;
        }
      }
    }
    
    // remove edges
    boost::remove_edge_if([&](auto ed) { return g[ed].weight == 0.f; }, g);
    ACTS_DEBUG("After edge removal: Graph has " << boost::num_edges(g));
  }

  const auto numberLabels = boost::connected_components(g, &trackLabels[0]);

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
