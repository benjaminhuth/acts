// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsPlugins/Gnn/Tensor.hpp"
#include "ActsPlugins/Gnn/detail/CudaUtils.hpp"

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace {

__global__ void sigmoidImpl(std::size_t size, float *array) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) {
    return;
  }

  array[i] = 1.f / (1.f + __expf(-array[i]));
}

__global__ void applyCut(std::size_t size, float cutoff, const float *array,
                         bool *mask) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) {
    return;
  }

  mask[i] = array[i] > cutoff;
}

// Kernel to copy edge feature rows using precomputed output positions
// Each thread handles one input edge
__global__ void copyEdgeFeaturesKernel(const float *inputFeatures,
                                       float *outputFeatures, const bool *mask,
                                       const int *outputPositions,
                                       const std::size_t nEdges,
                                       const std::size_t numFeatures) {
  // Each thread handles one input edge
  const std::size_t inputRow = blockIdx.x * blockDim.x + threadIdx.x;
  if (inputRow >= nEdges) {
    return;
  }

  // Skip edges that didn't pass the mask
  if (!mask[inputRow]) {
    return;
  }

  // Get output position from prefix sum
  const std::size_t outputRow = outputPositions[inputRow];

  // Copy entire row of features
  for (std::size_t f = 0; f < numFeatures; ++f) {
    outputFeatures[outputRow * numFeatures + f] =
        inputFeatures[inputRow * numFeatures + f];
  }
}

}  // namespace

namespace ActsPlugins::detail {

void cudaSigmoid(Tensor<float> &tensor, cudaStream_t stream) {
  dim3 blockDim = 1024;
  dim3 gridDim = (tensor.size() + blockDim.x - 1) / blockDim.x;
  sigmoidImpl<<<gridDim, blockDim, 0, stream>>>(tensor.size(), tensor.data());
  ACTS_CUDA_CHECK(cudaGetLastError());
}

std::tuple<Tensor<float>, Tensor<std::int64_t>, std::optional<Tensor<float>>>
cudaApplyScoreCut(const Tensor<float> &scores,
                  const Tensor<std::int64_t> &edgeIndex,
                  const std::optional<Tensor<float>> &edgeFeatures, float cut,
                  cudaStream_t stream) {
  ExecutionContext execContext{scores.device(), stream};

  // Create mask tensor from scores (Tensor expects 2D shape)
  auto mask = Tensor<bool>::Create({1, scores.size()}, execContext);

  dim3 blockDim = 1024;
  dim3 gridDim = (scores.size() + blockDim.x - 1) / blockDim.x;
  applyCut<<<gridDim, blockDim, 0, stream>>>(scores.size(), cut, scores.data(),
                                             mask.data());
  ACTS_CUDA_CHECK(cudaGetLastError());

  // Count passing edges
  const std::size_t nEdgesAfter = thrust::count(
      thrust::device.on(stream), mask.data(), mask.data() + mask.size(), true);

  // Filter edge index and features using applyEdgeMask
  auto [newEdgeIndex, newEdgeFeatures] =
      applyEdgeMask(edgeIndex, edgeFeatures, mask, stream);

  // Filter scores using same mask
  auto newScores = Tensor<float>::Create({nEdgesAfter, 1}, execContext);
  auto pred = [] __device__(bool x) { return x; };
  thrust::copy_if(thrust::device.on(stream), scores.data(),
                  scores.data() + scores.size(), mask.data(), newScores.data(),
                  pred);

  return {std::move(newScores), std::move(newEdgeIndex),
          std::move(newEdgeFeatures)};
}

std::pair<Tensor<std::int64_t>, std::optional<Tensor<float>>> cpuApplyEdgeMask(
    const Tensor<std::int64_t> &edgeIndex,
    const std::optional<Tensor<float>> &edgeFeatures,
    const Tensor<bool> &mask) {
  ExecutionContext execContext{Device::Cpu()};

  const std::size_t nEdges = edgeIndex.shape().at(1);
  const std::size_t maskSize = mask.size();  // Total elements in mask
  const bool *maskData = mask.data();

  // Ensure mask has exactly one element per edge
  assert(maskSize == nEdges && "Mask size must equal number of edges");

  // Count passing edges
  std::size_t nEdgesAfter = 0;
  for (std::size_t i = 0; i < nEdges; ++i) {
    if (maskData[i]) {
      ++nEdgesAfter;
    }
  }

  // Create output edge index tensor
  auto outputEdgeIndex =
      Tensor<std::int64_t>::Create({2, nEdgesAfter}, execContext);

  // Filter edge indices
  const std::int64_t *srcNodes = edgeIndex.data();
  const std::int64_t *dstNodes = edgeIndex.data() + nEdges;
  std::int64_t *outputSrc = outputEdgeIndex.data();
  std::int64_t *outputDst = outputEdgeIndex.data() + nEdgesAfter;

  for (std::size_t i = 0, j = 0; i < nEdges; ++i) {
    if (maskData[i]) {
      outputSrc[j] = srcNodes[i];
      outputDst[j] = dstNodes[i];
      ++j;
    }
  }

  // Filter edge features if present
  std::optional<Tensor<float>> outputEdgeFeatures;
  if (edgeFeatures.has_value()) {
    const std::size_t numFeatures = edgeFeatures->shape().at(1);
    outputEdgeFeatures =
        Tensor<float>::Create({nEdgesAfter, numFeatures}, execContext);

    const float *inputData = edgeFeatures->data();
    float *outputData = outputEdgeFeatures->data();

    // Copy rows where mask is true
    for (std::size_t i = 0, j = 0; i < nEdges; ++i) {
      if (maskData[i]) {
        std::copy_n(inputData + i * numFeatures, numFeatures,
                    outputData + j * numFeatures);
        ++j;
      }
    }
  }

  return {std::move(outputEdgeIndex), std::move(outputEdgeFeatures)};
}

std::pair<Tensor<std::int64_t>, std::optional<Tensor<float>>> cudaApplyEdgeMask(
    const Tensor<std::int64_t> &edgeIndex,
    const std::optional<Tensor<float>> &edgeFeatures, const Tensor<bool> &mask,
    cudaStream_t stream) {
  ExecutionContext execContext{edgeIndex.device(), stream};

  const std::size_t nEdges = edgeIndex.shape().at(1);
  const std::size_t maskSize = mask.size();  // Total elements in mask
  const bool *maskData = mask.data();

  // Ensure mask has exactly one element per edge
  assert(maskSize == nEdges && "Mask size must equal number of edges");

  // Count passing edges
  const std::size_t nEdgesAfter = thrust::count(
      thrust::device.on(stream), maskData, maskData + nEdges, true);

  // Create output edge index tensor
  auto outputEdgeIndex =
      Tensor<std::int64_t>::Create({2, nEdgesAfter}, execContext);

  // Filter edge indices using thrust::copy_if
  auto pred = [] __device__(bool x) { return x; };
  const std::int64_t *srcNodes = edgeIndex.data();
  const std::int64_t *dstNodes = edgeIndex.data() + nEdges;

  thrust::copy_if(thrust::device.on(stream), srcNodes, srcNodes + nEdges,
                  maskData, outputEdgeIndex.data(), pred);
  thrust::copy_if(thrust::device.on(stream), dstNodes, dstNodes + nEdges,
                  maskData, outputEdgeIndex.data() + nEdgesAfter, pred);

  // Filter edge features if present
  std::optional<Tensor<float>> outputEdgeFeatures;
  if (edgeFeatures.has_value()) {
    const std::size_t numFeatures = edgeFeatures->shape().at(1);

    // Allocate memory for prefix sum output positions
    int *outputPositions{};
    ACTS_CUDA_CHECK(
        cudaMallocAsync(&outputPositions, nEdges * sizeof(int), stream));

    // Compute prefix sum of mask to get output positions
    thrust::exclusive_scan(thrust::device.on(stream), maskData,
                           maskData + nEdges, outputPositions, 0);

    // Create output tensor
    outputEdgeFeatures =
        Tensor<float>::Create({nEdgesAfter, numFeatures}, execContext);

    // Launch kernel to copy features
    const dim3 blockDim = 256;
    const dim3 gridDim = (nEdges + blockDim.x - 1) / blockDim.x;
    copyEdgeFeaturesKernel<<<gridDim, blockDim, 0, stream>>>(
        edgeFeatures->data(), outputEdgeFeatures->data(), maskData,
        outputPositions, nEdges, numFeatures);
    ACTS_CUDA_CHECK(cudaGetLastError());

    // Free temporary memory
    ACTS_CUDA_CHECK(cudaFreeAsync(outputPositions, stream));
  }

  return {std::move(outputEdgeIndex), std::move(outputEdgeFeatures)};
}

}  // namespace ActsPlugins::detail
