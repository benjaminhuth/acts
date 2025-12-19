// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Utilities/Logger.hpp"
#include "ActsPlugins/Gnn/Tensor.hpp"
#include "ActsPlugins/Gnn/TrackLengthEdgeFilter.hpp"
#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"

#include <cstdint>

#include <cuda_runtime_api.h>
#include <math.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace ActsPlugins::detail {

// Node weight constants
constexpr int PIXEL_WEIGHT =
    1;  // Weight for pixel layers (radius < stripRadius)
constexpr int STRIP_WEIGHT =
    2;  // Weight for strip layers (radius >= stripRadius)

// Predicate for filtering with boolean mask
struct BoolPredicate {
  __device__ bool operator()(bool x) const { return x; }
};

//
//   1 - 2
//        \
//        3 - 4
//       /
//      1

// step 1
//

template <typename T>
__global__ void fillNodeWeights(T *weights, const float *features,
                                float minStripRadius, std::size_t radiusOffset,
                                std::size_t nNodes, std::size_t nFeatures) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= nNodes) {
    return;
  }

  const int j = radiusOffset + i * nFeatures;
  weights[i] = features[j] < minStripRadius ? PIXEL_WEIGHT : STRIP_WEIGHT;
}

template <typename T1, typename T2, typename T3>
__global__ void forward(const T1 *srcNodes, const T1 *tgtNodes,
                        const T3 *nodeWeights, std::size_t nEdges,
                        const T2 *accumulatedPrev, T2 *accumulatedNext,
                        bool *globalChanged) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= nEdges) {
    return;
  }

  const auto srcNode = srcNodes[i];
  const auto tgtNode = tgtNodes[i];

  const auto nextTgt = accumulatedPrev[srcNode] + nodeWeights[tgtNode];
  const auto prevTgt = accumulatedPrev[tgtNode];

  bool changed = false;
  if (prevTgt < nextTgt) {
    accumulatedNext[tgtNode] = nextTgt;
    changed = true;
  }

  // printf(
  //     "  [FWD] edge %d (%lld->%lld): Prev[src]=%d, Prev[tgt]=%d, nextTgt=%d,
  //     "
  //     "Next[tgt]=%d, changed=%d\n",
  //     i, (long long)srcNode, (long long)tgtNode,
  //     (int)accumulatedPrev[srcNode], (int)prevTgt, (int)nextTgt,
  //     (int)accumulatedNext[tgtNode], changed);

  if (__syncthreads_or(changed) && threadIdx.x == 0) {
    *globalChanged = true;
  }
}

template <typename T1, typename T2, typename T3>
__global__ void backward(const T1 *srcNodes, const T1 *tgtNodes,
                         std::size_t nEdges, const T2 *forwardAccumulated,
                         const T2 *backwardAccumulatedPrev,
                         T2 *backwardAccumulatedNext, const T3 *nodeWeights,
                         bool *globalChanged) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= nEdges) {
    return;
  }

  const auto srcNode = srcNodes[i];
  const auto tgtNode = tgtNodes[i];

  auto diff = forwardAccumulated[tgtNode] - forwardAccumulated[srcNode];

  const int weight = nodeWeights[tgtNode];
  // Use backwardAccumulatedPrev[tgtNode], not forwardAccumulated[tgtNode]!
  const auto nextSrc = backwardAccumulatedPrev[tgtNode] - (diff - weight);
  const auto prevSrc = backwardAccumulatedPrev[srcNode];

  bool changed = false;
  if (prevSrc < nextSrc) {
    backwardAccumulatedNext[srcNode] = nextSrc;
    changed = true;
  }

  // printf(
  //     "  [BWD] edge %d (%lld->%lld): fwd[src]=%d, fwd[tgt]=%d, "
  //     "bwdPrev[src]=%d, bwdPrev[tgt]=%d, diff=%d, nextSrc=%d, Next[src]=%d, "
  //     "changed=%d\n",
  //     i, (long long)srcNode, (long long)tgtNode,
  //     (int)forwardAccumulated[srcNode], (int)forwardAccumulated[tgtNode],
  //     (int)prevSrc, (int)backwardAccumulatedPrev[tgtNode], (int)diff,
  //     (int)nextSrc, (int)backwardAccumulatedNext[srcNode], changed);

  if (__syncthreads_or(changed) && threadIdx.x == 0) {
    *globalChanged = true;
  }
}

template <typename T1, typename T2, typename T3>
__global__ void createEdgeMask(const T1 *srcNodes, const T1 *tgtNodes,
                               std::size_t nEdges, const T2 *forwardAccumulated,
                               const T2 *backwardAccumulated,
                               const T3 *nodeWeights,
                               std::size_t minTrackLength, bool *mask) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= nEdges) {
    return;
  }

  const auto srcNode = srcNodes[i];
  const auto tgtNode = tgtNodes[i];

  // Compute edge accumulated value from node values
  // Following the CPU implementation logic
  const int edgeAccumulated =
      backwardAccumulated[tgtNode] -
      (forwardAccumulated[tgtNode] - forwardAccumulated[srcNode] -
       nodeWeights[tgtNode]);

  mask[i] = edgeAccumulated >= static_cast<int>(minTrackLength);

  // printf(
  //     "  [MASK] edge %d (%lld->%lld): fwd[src]=%d, fwd[tgt]=%d, bwd[tgt]=%d,
  //     "
  //     "edgeAcc=%d, pass=%d (minLen=%d)\n",
  //     i, (long long)srcNode, (long long)tgtNode,
  //     (int)forwardAccumulated[srcNode], (int)forwardAccumulated[tgtNode],
  //     (int)backwardAccumulated[tgtNode], edgeAccumulated, mask[i],
  //     (int)minTrackLength);
}

}  // namespace ActsPlugins::detail

namespace ActsPlugins {

PipelineTensors TrackLengthEdgeFilter::filterEdgesCuda(
    PipelineTensors &&tensors, const ExecutionContext &execContext) {
  // Assert that execution context is CUDA
  if (!execContext.device.isCuda()) {
    throw std::runtime_error(
        "filterEdgesCuda called with non-CUDA execution context");
  }

  // Assert tensors are on CUDA
  if (!tensors.edgeIndex.device().isCuda()) {
    throw std::runtime_error(
        "Edge index tensor must be on CUDA for CUDA filtering");
  }
  if (!tensors.nodeFeatures.device().isCuda()) {
    throw std::runtime_error(
        "Node features tensor must be on CUDA for CUDA filtering");
  }
  if (tensors.edgeFeatures.has_value() &&
      !tensors.edgeFeatures->device().isCuda()) {
    throw std::runtime_error(
        "Edge features tensor must be on CUDA for CUDA filtering");
  }

  ACTS_VERBOSE("Filtering edges using CUDA implementation");

  cudaStream_t stream = execContext.stream.value();
  const std::size_t nEdges = tensors.edgeIndex.shape().at(1);
  const std::size_t nNodes = tensors.nodeFeatures.shape().at(0);

  // Handle empty graph - return empty tensor
  if (nEdges == 0) {
    ACTS_VERBOSE("Empty graph, returning empty edge tensor");
    return std::move(tensors);
  }

  const dim3 blockDim = 1024;
  const dim3 gridDimEdges = (nEdges + blockDim.x - 1) / blockDim.x;
  const dim3 gridDimNodes = (nNodes + blockDim.x - 1) / blockDim.x;

  int *nodeWeights{};
  ACTS_CUDA_CHECK(cudaMallocAsync(&nodeWeights, sizeof(int) * nNodes, stream));

  // Get node features data and dimensions
  const std::size_t nFeatures = tensors.nodeFeatures.shape().at(1);
  const float *nodeFeatureData = tensors.nodeFeatures.data();

  // Fill node weights based on radius feature
  detail::fillNodeWeights<<<gridDimNodes, blockDim, 0, stream>>>(
      nodeWeights, nodeFeatureData, m_cfg.stripRadius, m_cfg.radiusFeatureIdx,
      nNodes, nFeatures);
  ACTS_CUDA_CHECK(cudaGetLastError());

  // Extract source and destination node arrays
  const std::int64_t *srcNodes = tensors.edgeIndex.data();
  const std::int64_t *dstNodes = tensors.edgeIndex.data() + nEdges;

  // Allocate temporary arrays
  bool *cudaChanged{};
  ACTS_CUDA_CHECK(cudaMallocAsync(&cudaChanged, sizeof(bool), stream));

  int *mem1{}, *mem2{};
  ACTS_CUDA_CHECK(cudaMallocAsync(&mem1, sizeof(int) * nNodes, stream));
  ACTS_CUDA_CHECK(cudaMallocAsync(&mem2, sizeof(int) * nNodes, stream));

  // Initialize forward accumulated prev with node weights
  int *forwardAccumulatedPrev = mem2;
  ACTS_CUDA_CHECK(cudaMemcpyAsync(forwardAccumulatedPrev, nodeWeights,
                                  sizeof(int) * nNodes,
                                  cudaMemcpyDeviceToDevice, stream));

  // Initialize forward accumulated with zeros
  int *forwardAccumulatedNext = mem1;
  ACTS_CUDA_CHECK(
      cudaMemsetAsync(forwardAccumulatedNext, 0, sizeof(int) * nNodes, stream));

  bool changed{};

  // Accumulate forward through graph
  int forwardIter = 0;
  do {
    ACTS_VERBOSE("Forward iteration " << forwardIter++);
    ACTS_CUDA_CHECK(cudaMemsetAsync(cudaChanged, 0, sizeof(bool), stream));

    // Initialize Next buffer with Prev values before kernel
    ACTS_CUDA_CHECK(cudaMemcpyAsync(
        forwardAccumulatedNext, forwardAccumulatedPrev, sizeof(int) * nNodes,
        cudaMemcpyDeviceToDevice, stream));

    detail::forward<<<gridDimEdges, blockDim, 0, stream>>>(
        srcNodes, dstNodes, nodeWeights, nEdges, forwardAccumulatedPrev,
        forwardAccumulatedNext, cudaChanged);
    ACTS_CUDA_CHECK(cudaGetLastError());

    ACTS_CUDA_CHECK(cudaMemcpyAsync(&changed, cudaChanged, sizeof(bool),
                                    cudaMemcpyDeviceToHost, stream));
    ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));

    ACTS_VERBOSE("  changed = " << static_cast<int>(changed));

    // Copy Next back to Prev for next iteration
    ACTS_CUDA_CHECK(cudaMemcpyAsync(
        forwardAccumulatedPrev, forwardAccumulatedNext, sizeof(int) * nNodes,
        cudaMemcpyDeviceToDevice, stream));

    if (forwardIter > 100) {
      throw std::runtime_error("Forward propagation exceeded iteration limit");
    }
  } while (changed);

  // Repurpose pointers
  int *forwardAccumulated = forwardAccumulatedPrev;
  int *backwardAccumulatedPrev = forwardAccumulatedNext;
  // Initialize backward with forward values
  ACTS_CUDA_CHECK(cudaMemcpyAsync(backwardAccumulatedPrev, forwardAccumulated,
                                  sizeof(int) * nNodes,
                                  cudaMemcpyDeviceToDevice, stream));

  int *mem3{};
  ACTS_CUDA_CHECK(cudaMallocAsync(&mem3, sizeof(int) * nNodes, stream));

  int *backwardAccumulatedNext = mem3;
  ACTS_CUDA_CHECK(cudaMemsetAsync(backwardAccumulatedNext, 0,
                                  sizeof(int) * nNodes, stream));

  // Propagate backwards
  int backwardIter = 0;
  do {
    ACTS_VERBOSE("Backward iteration " << backwardIter++);
    ACTS_CUDA_CHECK(cudaMemsetAsync(cudaChanged, 0, sizeof(bool), stream));

    // Initialize Next buffer with Prev values before kernel
    ACTS_CUDA_CHECK(cudaMemcpyAsync(
        backwardAccumulatedNext, backwardAccumulatedPrev, sizeof(int) * nNodes,
        cudaMemcpyDeviceToDevice, stream));

    detail::backward<<<gridDimEdges, blockDim, 0, stream>>>(
        srcNodes, dstNodes, nEdges, forwardAccumulated, backwardAccumulatedPrev,
        backwardAccumulatedNext, nodeWeights, cudaChanged);
    ACTS_CUDA_CHECK(cudaGetLastError());

    ACTS_CUDA_CHECK(cudaMemcpyAsync(&changed, cudaChanged, sizeof(bool),
                                    cudaMemcpyDeviceToHost, stream));
    ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));

    ACTS_VERBOSE("  changed = " << static_cast<int>(changed));

    // Copy Next back to Prev for next iteration
    ACTS_CUDA_CHECK(cudaMemcpyAsync(
        backwardAccumulatedPrev, backwardAccumulatedNext, sizeof(int) * nNodes,
        cudaMemcpyDeviceToDevice, stream));

    if (backwardIter > 100) {
      throw std::runtime_error("Backward propagation exceeded iteration limit");
    }
  } while (changed);

  // Repurpose pointer
  int *backwardAccumulated = backwardAccumulatedPrev;

  // Create edge mask tensor (Tensor expects 2D shape)
  auto mask = Tensor<bool>::Create({1, nEdges}, execContext);

  detail::createEdgeMask<<<gridDimEdges, blockDim, 0, stream>>>(
      srcNodes, dstNodes, nEdges, forwardAccumulated, backwardAccumulated,
      nodeWeights, m_cfg.minTrackLength, mask.data());
  ACTS_CUDA_CHECK(cudaGetLastError());

  // Apply mask to filter edges and features
  auto [newEdgeIndex, newEdgeFeatures] = applyEdgeMask(
      tensors.edgeIndex, tensors.edgeFeatures, mask, execContext.stream);

  ACTS_VERBOSE("Edges after filtering: " << newEdgeIndex.shape().at(1)
                                         << " (from " << nEdges << ")");

  // Free temporary arrays
  ACTS_CUDA_CHECK(cudaFreeAsync(nodeWeights, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(cudaChanged, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(mem1, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(mem2, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(mem3, stream));

  ACTS_VERBOSE("CUDA filtering complete");

  // Update tensors
  tensors.edgeIndex = std::move(newEdgeIndex);
  tensors.edgeFeatures = std::move(newEdgeFeatures);

  return std::move(tensors);
}

}  // namespace ActsPlugins
