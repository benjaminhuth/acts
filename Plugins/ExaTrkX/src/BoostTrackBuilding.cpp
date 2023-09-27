// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/ExaTrkX/BoostTrackBuilding.hpp"

#include "Acts/Plugins/ExaTrkX/detail/BoostTrackBuildingUtils.hpp"
#include "Acts/Utilities/Zip.hpp"

#include <map>

#include <boost/beast/core/span.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <torch/torch.h>

using namespace torch::indexing;

template <typename vertex_t, typename weight_t>
auto weaklyConnectedComponents(std::size_t numNodes,
                               boost::beast::span<vertex_t>& rowIndices,
                               boost::beast::span<vertex_t>& colIndices,
                               boost::beast::span<weight_t>& edgeWeights,
                               boost::beast::span<float>& nodeRadius,
                               bool ensure2EdgesPerVertex,
                               std::vector<vertex_t>& trackLabels,
                               const Acts::Logger& logger) {}

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

  // Construct Graph
  struct EdgeProperty {
    weight_t weight;
  };

  using Graph =
      boost::adjacency_list<boost::vecS,            // edge list
                            boost::vecS,            // vertex list
                            boost::bidirectionalS,  // directedness
                            boost::no_property,     // property of vertices
                            EdgeProperty            // property of edges
                            >;

  Graph g(numSpacepoints);

  for (const auto [row, col, weight] :
       Acts::zip(rowIndices, colIndices, edgeWeights)) {
    const auto rowRadius = *(nodeRadius.begin() + row);
    const auto colRadius = *(nodeRadius.begin() + col);
    if (rowRadius < colRadius) {
      boost::add_edge(row, col, EdgeProperty{weight}, g);
    } else {
      boost::add_edge(col, row, EdgeProperty{weight}, g);
    }
  }

  if (m_cfg.cleanSubgraphs) {
    ACTS_VERBOSE("Clean subgraph");
    detail::cleanSubgraphs(g, logger());
  }

  std::vector<vertex_t> trackLabels(numSpacepoints);
  auto numberLabels = boost::connected_components(g, &trackLabels[0]);

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
