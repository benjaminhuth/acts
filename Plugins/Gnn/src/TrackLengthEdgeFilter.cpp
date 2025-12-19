// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsPlugins/Gnn/TrackLengthEdgeFilter.hpp"

#include <span>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/reverse_graph.hpp>
#include <boost/graph/topological_sort.hpp>

#ifndef ACTS_GNN_CPUONLY
#include <cuda_runtime_api.h>

namespace ActsPlugins::detail {
Tensor<std::int64_t> cudaFilterEdgesByTrackLength(
    const Tensor<std::int64_t> &edgeIndex, const Tensor<float> &nodeFeatures,
    std::size_t nNodes, std::size_t minTrackLength, float stripRadius,
    std::size_t radiusFeatureIdx, cudaStream_t stream,
    const Acts::Logger &logger);
}
#endif

struct NodeProperty {
  int weight{1};
  int distance{weight};
  std::size_t accumulated{};
};

struct EdgeProperty {
  int distance{};
  std::size_t accumulated{};
};

using Graph =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                          NodeProperty, EdgeProperty>;

using Vi = std::vector<int>;

// helper function that creates a boost graph from two vectors of int
inline auto make_graph(const Vi &from, const Vi &to, const Vi &weights) {
  Graph g;
  for (std::size_t i = 0; i < from.size(); ++i) {
    boost::add_edge(from[i], to[i], g);
  }
  for (auto i = 0ul; i < boost::num_vertices(g); ++i) {
    g[i].weight = weights.at(i);
  }
  return g;
}

inline auto findMaxDistances(Graph &g, const Acts::Logger &logger) {
  // do topological sort with boost
  std::vector<Graph::vertex_descriptor> topoOrder(boost::num_vertices(g));
  boost::topological_sort(g, topoOrder.begin());

  for (auto vit = topoOrder.rbegin(); vit != topoOrder.rend(); ++vit) {
    auto src = *vit;
    auto [it, end] = boost::out_edges(src, g);
    ACTS_VERBOSE("Vertex: " << src);
    for (; it != end; ++it) {
      auto tgt = boost::target(*it, g);
      g[*it].distance = g[src].distance + g[tgt].weight;
      auto newDist = std::max(g[tgt].distance, g[*it].distance);
      ACTS_VERBOSE("- set edge " << *it << "|" << g[tgt].weight << ": "
                                 << g[tgt].distance << " -> " << newDist);
      g[tgt].distance = newDist;
    }
  }
}

inline void accumulateBackwards(Graph &g, const Acts::Logger &logger) {
  for (auto i = 0ul; i < boost::num_vertices(g); ++i) {
    g[i].accumulated = g[i].distance;
  }

  auto rg = boost::make_reverse_graph(g);
  // do topological sort with boost
  std::vector<Graph::vertex_descriptor> topoOrder(boost::num_vertices(rg));
  boost::topological_sort(rg, topoOrder.begin());

  // Go through the graph in reversed topological order
  for (auto vit = topoOrder.rbegin(); vit != topoOrder.rend(); ++vit) {
    auto src = *vit;
    ACTS_VERBOSE("On vtx " << src << ", accumulated: " << rg[src].accumulated
                           << ", distance " << rg[src].distance);
    auto [it, end] = boost::out_edges(src, rg);
    for (; it != end; ++it) {
      auto tgt = boost::target(*it, rg);
      rg[*it].accumulated =
          rg[src].accumulated -
          (rg[src].distance - rg[tgt].distance - rg[src].weight);
      rg[tgt].accumulated = std::max(rg[tgt].accumulated, rg[*it].accumulated);
      ACTS_VERBOSE("- set vertex " << src << ": " << rg[tgt].accumulated);
    }
  }
}

inline std::pair<std::vector<bool>, std::size_t> filterEdges(
    const Vi &src, const Vi &dst, const Vi &nodeWeights,
    std::size_t trackLengthConstraint, const Acts::Logger &logger) {
  auto g = make_graph(src, dst, nodeWeights);

  ACTS_VERBOSE("Find max distances in forward pass");
  findMaxDistances(g, logger);
  Vi distances;
  for (auto i = 0ul; i < boost::num_vertices(g); ++i) {
    distances.push_back(g[i].distance);
  }

  ACTS_VERBOSE("Propagate max distances back in graph");
  accumulateBackwards(g, logger);
  Vi accumulated;
  for (auto i = 0ul; i < boost::num_vertices(g); ++i) {
    accumulated.push_back(g[i].accumulated);
  }

  std::size_t numKeptEdges = 0;
  std::vector<bool> edgeMask;
  for (auto e : boost::make_iterator_range(boost::edges(g))) {
    edgeMask.push_back(g[e].accumulated >= trackLengthConstraint);
    if (edgeMask.back()) {
      ++numKeptEdges;
    }
  }

  return {edgeMask, numKeptEdges};
}

struct StridedSpan {
  const float *data;
  std::size_t size;
  std::size_t stride;

  float operator[](std::size_t idx) const { return data[idx * stride]; }
};

namespace ActsPlugins {

TrackLengthEdgeFilter::TrackLengthEdgeFilter(
    const Config &cfg, std::unique_ptr<const Acts::Logger> logger)
    : m_logger(std::move(logger)), m_cfg(cfg) {}

TrackLengthEdgeFilter::~TrackLengthEdgeFilter() = default;

PipelineTensors TrackLengthEdgeFilter::filterEdgesCpu(
    PipelineTensors &&tensors, const ExecutionContext &execContext) {
  // Assert that execution context is CPU
  if (!execContext.device.isCpu()) {
    throw std::runtime_error(
        "filterEdgesCpu called with non-CPU execution context");
  }

  // Assert tensors are on CPU
  if (!tensors.edgeIndex.device().isCpu()) {
    throw std::runtime_error(
        "Edge index tensor must be on CPU for CPU filtering");
  }
  if (!tensors.nodeFeatures.device().isCpu()) {
    throw std::runtime_error(
        "Node features tensor must be on CPU for CPU filtering");
  }
  if (tensors.edgeFeatures.has_value() &&
      !tensors.edgeFeatures->device().isCpu()) {
    throw std::runtime_error(
        "Edge features tensor must be on CPU for CPU filtering");
  }

  // Process directly on CPU (no cloning needed - tensors already on CPU)
  std::span<const std::int64_t> srcSpan{tensors.edgeIndex.data(),
                                        tensors.edgeIndex.shape().at(1)};
  std::span<const std::int64_t> tgtSpan{
      tensors.edgeIndex.data() + tensors.edgeIndex.shape().at(1),
      tensors.edgeIndex.shape().at(1)};

  StridedSpan radius{tensors.nodeFeatures.data() + m_cfg.radiusFeatureIdx,
                     tensors.nodeFeatures.shape().at(0),
                     tensors.nodeFeatures.shape().at(1)};

  Vi from, to, weights;
  from.reserve(tensors.edgeIndex.shape().at(1));
  to.reserve(tensors.edgeIndex.shape().at(1));
  weights.reserve(tensors.nodeFeatures.shape().at(0));

  for (auto i = 0ul; i < tensors.edgeIndex.shape().at(1); ++i) {
    from.push_back(static_cast<int>(srcSpan[i]));
    to.push_back(static_cast<int>(tgtSpan[i]));
  }

  for (auto i = 0ul; i < tensors.nodeFeatures.shape().at(0); ++i) {
    weights.push_back(radius[i] < m_cfg.stripRadius ? 1 : 2);
  }

  auto [boolMask, nEdgesKept] =
      filterEdges(from, to, weights, m_cfg.minTrackLength, logger());

  // Convert std::vector<bool> to Tensor<bool> (Tensor expects 2D shape)
  auto mask = Tensor<bool>::Create({1, boolMask.size()}, execContext);
  std::copy(boolMask.begin(), boolMask.end(), mask.data());

  // Apply mask to filter edges and features
  auto [newEdgeIndex, newEdgeFeatures] =
      applyEdgeMask(tensors.edgeIndex, tensors.edgeFeatures, mask, {});

  tensors.edgeIndex = std::move(newEdgeIndex);
  tensors.edgeFeatures = std::move(newEdgeFeatures);

  return std::move(tensors);
}

#ifdef ACTS_GNN_CPUONLY
PipelineTensors TrackLengthEdgeFilter::filterEdgesCuda(
    PipelineTensors &&tensors, const ExecutionContext &execContext) {
  (void)tensors;
  (void)execContext;
  throw std::runtime_error(
      "TrackLengthEdgeFilter CUDA support is not enabled. "
      "Please rebuild with ACTS_GNN_WITH_CUDA=ON.");
}
#endif

PipelineTensors TrackLengthEdgeFilter::operator()(
    PipelineTensors tensors, const ExecutionContext &execContext) {
  if (execContext.device.isCuda()) {
    tensors = filterEdgesCuda(std::move(tensors), execContext);
  } else {
    tensors = filterEdgesCpu(std::move(tensors), execContext);
  }

  return tensors;
}

}  // namespace ActsPlugins
