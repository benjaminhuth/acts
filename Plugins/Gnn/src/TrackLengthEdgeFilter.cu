// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstdint>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include <math.h>

#include "ActsPlugins/Gnn/Tensor.hpp"
#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"

namespace ActsPlugins::detail {

//
//   1 - 2 
//        \
//        3 - 4
//       /
//      1

// step 1
// 


template<typename T1, typename T2>
__global__ void forward(const T1 *srcNodes, const T1 *tgtNodes,
                       std::size_t nEdges, const T2 *accumulatedPrev,
                       T2 *accumulatedNext, bool *globalChanged) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= nEdges) {
    return;
  }

  const auto srcNode = srcNodes[i];
  const auto tgtNode = tgtNodes[i];

  const auto nextTgt = accumulatedPrev[srcNode] + 1.f; //nodeWeights[tgtNode];

  const auto prevTgt = accumulatedPrev[tgtNode];

  bool changed = false;
  if ( prevTgt < nextTgt ) {
    accumulatedNext[tgtNode] = nextTgt;
    changed = true;
  }

  if (__syncthreads_or(changed) && threadIdx.x == 0) {
    *globalChanged = true;
  }
}

template<typename T1, typename T2>
__global__ void backward(const T1 *srcNodes, const T1 *tgtNodes,
    std::size_t nEdges, const T2 *forwardAccumulated,
    const T2 *backwardAccumulatedPrev, T2 *backwardAccumulatedNext, bool *globalChanged) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= nEdges) {
    return;
  }

  const auto srcNode = srcNodes[i];
  const auto tgtNode = tgtNodes[i];

  auto diff = forwardAccumulated[tgtNode] - forwardAccumulated[srcNode];

  const float weight = 1.f;
  const auto nextSrc = forwardAccumulated[tgtNode] - (diff - weight);
  const auto prevSrc = backwardAccumulatedPrev[srcNode];

  bool changed = false;
  if( prevSrc < nextSrc ) {
    backwardAccumulatedNext[srcNode] = nextSrc;
    changed = true;
  }

  if (__syncthreads_or(changed) && threadIdx.x == 0) {
    *globalChanged = true;
  }
}

template<typename T1, typename T2>
__global__ void createEdgeMask(const T1 *srcNodes,
                                const T1 *tgtNodes,
                                std::size_t nEdges,
                                const T2 *forwardAccumulated,
                                const T2 *backwardAccumulated,
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
      (forwardAccumulated[tgtNode] - forwardAccumulated[srcNode] - 1);

  mask[i] = edgeAccumulated >= static_cast<int>(minTrackLength);
}




Tensor<std::int64_t> cudaFilterEdgesByTrackLength(
    const Tensor<std::int64_t> &edgeIndex, std::size_t nNodes,
    std::size_t minTrackLength, cudaStream_t stream) {
  const dim3 blockDim = 1024;
  const std::size_t nEdges = edgeIndex.shape().at(1);
  const dim3 gridDimEdges = (nEdges + blockDim.x - 1) / blockDim.x;
  const dim3 gridDimNodes = (nNodes + blockDim.x - 1) / blockDim.x;

  ExecutionContext execContext{edgeIndex.device(), stream};

  // Extract source and destination node arrays
  const std::int64_t *srcNodes = edgeIndex.data();
  const std::int64_t *dstNodes = edgeIndex.data() + nEdges;

  // Allocate temporary arrays
  bool *cudaChanged{};
  ACTS_CUDA_CHECK(cudaMallocAsync(&cudaChanged, sizeof(bool), stream));

  int *mem1{}, *mem2{};
  ACTS_CUDA_CHECK(
      cudaMallocAsync(&mem1, sizeof(int) * nNodes, stream));
  ACTS_CUDA_CHECK(
      cudaMallocAsync(&mem2, sizeof(int) * nNodes, stream));
  
  int *forwardAccumulatedNext = mem1;
  int *forwardAccumulatedPrev = mem2;
  ACTS_CUDA_CHECK(
      cudaMemsetAsync(forwardAccumulatedPrev, 0, sizeof(int) * nNodes, stream));

  bool changed{};

  // Accumulate forward through graph
  do {
    ACTS_CUDA_CHECK(cudaMemsetAsync(cudaChanged, 0, sizeof(bool), stream));

    forward<<<gridDimEdges, blockDim, 0, stream>>>(srcNodes,
        dstNodes, nEdges, forwardAccumulatedPrev,
        forwardAccumulatedNext, cudaChanged);
    ACTS_CUDA_CHECK(cudaGetLastError());

    ACTS_CUDA_CHECK(cudaMemcpyAsync(&changed, cudaChanged, sizeof(bool),
                                    cudaMemcpyDeviceToHost, stream));
    ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::swap(forwardAccumulatedNext, forwardAccumulatedPrev);
  } while (changed);

  // Repurpose pointers, due to last swap, prev has the final result
  int *forwardAccumulated = forwardAccumulatedPrev;
  int *backwardAccumulatedPrev = forwardAccumulatedNext;
  ACTS_CUDA_CHECK(cudaMemsetAsync(backwardAccumulatedPrev, 0, sizeof(int) * nNodes, stream));

  int *mem3{};
  ACTS_CUDA_CHECK(
      cudaMallocAsync(&mem3, sizeof(int) * nNodes, stream));
  
  int *backwardAccumulatedNext = mem3;
  ACTS_CUDA_CHECK(
      cudaMemsetAsync(backwardAccumulatedNext, 0, sizeof(int) * nNodes, stream));

  // Propagate backwards
  do {
    ACTS_CUDA_CHECK(cudaMemsetAsync(cudaChanged, 0, sizeof(bool), stream));

    backward<<<gridDimEdges, blockDim, 0, stream>>>(srcNodes, dstNodes, nEdges, forwardAccumulated,
        backwardAccumulatedPrev, backwardAccumulatedNext, cudaChanged);
    ACTS_CUDA_CHECK(cudaGetLastError());

    ACTS_CUDA_CHECK(cudaMemcpyAsync(&changed, cudaChanged, sizeof(bool),
                                    cudaMemcpyDeviceToHost, stream));
    ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));

    std::swap(backwardAccumulatedNext, backwardAccumulatedPrev);
  } while (changed);

  // Repurpose pointer, due to last swap, prev has the final result
  int *backwardAccumulated = backwardAccumulatedPrev;

  // Create edge mask
  bool *mask{};
  ACTS_CUDA_CHECK(cudaMallocAsync(&mask, nEdges * sizeof(bool), stream));

  createEdgeMask<<<gridDimEdges, blockDim, 0, stream>>>(
      srcNodes, dstNodes, nEdges, forwardAccumulated, backwardAccumulated,
      minTrackLength, mask);
  ACTS_CUDA_CHECK(cudaGetLastError());

  // Count passing edges
  const std::size_t nEdgesAfter =
      thrust::count(thrust::device.on(stream), mask, mask + nEdges, true);

  // Create output tensor
  auto outputEdgeIndex =
      Tensor<std::int64_t>::Create({2, nEdgesAfter}, execContext);

  // Filter edges using thrust::copy_if
  auto pred = [] __device__(bool x) { return x; };
  thrust::copy_if(thrust::device.on(stream), srcNodes, srcNodes + nEdges, mask,
                  outputEdgeIndex.data(), pred);
  thrust::copy_if(thrust::device.on(stream), dstNodes, dstNodes + nEdges, mask,
                  outputEdgeIndex.data() + nEdgesAfter, pred);

  // Free temporary arrays
  ACTS_CUDA_CHECK(cudaFreeAsync(cudaChanged, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(mem1, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(mem2, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(mem3, stream));
  ACTS_CUDA_CHECK(cudaFreeAsync(mask, stream));

  return outputEdgeIndex;
}

}  // namespace ActsPlugins::detail


