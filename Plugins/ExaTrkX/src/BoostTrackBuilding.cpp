// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/ExaTrkX/BoostTrackBuilding.hpp"

#include "Acts/Utilities/Zip.hpp"

#include <algorithm>

#include <boost/beast/core/span.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

namespace Acts {

std::pair<Acts::Tensor<int>, std::size_t> BoostTrackBuilding::operator()(
    PipelineTensors tensors, const ExecutionContext& execContext) {
  ACTS_DEBUG("Start track building");
  using RTF = const Tensor<float>&;
  using RTI = const Tensor<std::int64_t>&;
  const auto& edgeTensor = tensors.edgeIndex.device().isCpu()
                               ? static_cast<RTI>(tensors.edgeIndex)
                               : static_cast<RTI>(tensors.edgeIndex.clone(
                                     {Device::Cpu(), execContext.stream}));
  const auto& scoreTensor = tensors.edgeScores->device().isCpu()
                                ? static_cast<RTF>(*tensors.edgeScores)
                                : static_cast<RTF>(tensors.edgeScores->clone(
                                      {Device::Cpu(), execContext.stream}));

  assert(edgeTensor.shape().at(0) == 2);
  assert(edgeTensor.shape().at(1) == scoreTensor.shape().at(0));

  ExecutionContext cpuContext = {Acts::Device::Cpu(), {}};
  const auto numSpacepoints = tensors.nodeFeatures.shape().at(0);
  const auto numEdges = scoreTensor.shape().at(1);

  if (numEdges == 0) {
    ACTS_WARNING("No edges remained after edge classification");
    return {Tensor<int>::Create({0, 0}, cpuContext), 0};
  }

  using vertex_t = std::int64_t;
  using weight_t = float;

  boost::beast::span<const vertex_t> rowIndices(edgeTensor.data(), numEdges);
  boost::beast::span<const vertex_t> colIndices(edgeTensor.data() + numEdges,
                                                numEdges);
  boost::beast::span<const weight_t> edgeWeights(scoreTensor.data(), numEdges);

  auto trackLabels = Tensor<int>::Create({numSpacepoints, 1ul}, cpuContext);

  using Graph =
      boost::adjacency_list<boost::vecS,         // edge list
                            boost::vecS,         // vertex list
                            boost::undirectedS,  // directedness
                            boost::no_property,  // property of vertices
                            weight_t             // property of edges
                            >;

  Graph g(numSpacepoints);

  for (const auto [row, col, weight] :
       Acts::zip(rowIndices, colIndices, edgeWeights)) {
    boost::add_edge(row, col, weight, g);
  }

  std::size_t numberLabels = boost::connected_components(g, trackLabels.data());

  ACTS_VERBOSE("Number of track labels: " << trackLabels.size());
  ACTS_VERBOSE("Number of unique track labels: " << [&]() {
    std::vector<vertex_t> sorted(trackLabels.data(),
                                 trackLabels.data() + trackLabels.size());
    std::ranges::sort(sorted);
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    return sorted.size();
  }());

  return {std::move(trackLabels), numberLabels};
}

}  // namespace Acts
