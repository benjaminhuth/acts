// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/ExaTrkX/CudaTrackBuilding.hpp"
#include "Acts/Plugins/ExaTrkX/detail/ConnectedComponents.cuh"
#include "Acts/Plugins/ExaTrkX/detail/CudaUtils.cuh"
#include "Acts/Plugins/ExaTrkX/detail/CudaUtils.hpp"
#include "Acts/Plugins/ExaTrkX/detail/JunctionRemoval.hpp"
#include "Acts/Utilities/Zip.hpp"

namespace Acts {

std::vector<std::vector<int>> CudaTrackBuilding::operator()(
    PipelineTensors tensors, std::vector<int>& spacepointIDs,
    const ExecutionContext& execContext) {
  ACTS_VERBOSE("Start CUDA track building");
  if (!(tensors.edgeIndex.device().isCuda() &&
        tensors.edgeScores.value().device().isCuda())) {
    throw std::runtime_error(
        "CudaTrackBuilding expects tensors to be on CUDA!");
  }

  const auto numSpacepoints = spacepointIDs.size();
  auto numEdges = static_cast<std::size_t>(tensors.edgeIndex.shape().at(1));

  if (numEdges == 0) {
    ACTS_DEBUG("No edges remained after edge classification");
    return {};
  }

  auto stream = execContext.stream.value();

  auto cudaSrcPtr = tensors.edgeIndex.data();
  auto cudaTgtPtr = tensors.edgeIndex.data() + numEdges;

  auto t0 = ACTS_TIME_STREAM_SYNC(Acts::Logging::DEBUG, execContext.stream);

  if (m_cfg.doJunctionRemoval) {
    assert(tensors.edgeScores->shape().at(0) ==
           tensors.edgeIndex.shape().at(1));
    auto cudaScorePtr = tensors.edgeScores->data();

    ACTS_DEBUG("Do junction removal...");
    auto [cudaSrcPtrJr, numEdgesOut] = detail::junctionRemovalCuda(
        numEdges, numSpacepoints, cudaScorePtr, cudaSrcPtr, cudaTgtPtr, stream);
    cudaSrcPtr = cudaSrcPtrJr;
    cudaTgtPtr = cudaSrcPtrJr + numEdgesOut;

    if (numEdgesOut == 0) {
      ACTS_WARNING(
          "No edges remained after junction removal, this should not happen!");
      ACTS_CUDA_CHECK(cudaFreeAsync(cudaSrcPtrJr, stream));
      ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
      return {};
    }

    ACTS_DEBUG("Removed " << numEdges - numEdgesOut
                          << " edges in junction removal");
    numEdges = numEdgesOut;
  }

  auto t1 = ACTS_TIME_STREAM_SYNC(Acts::Logging::DEBUG, execContext.stream);

  int* cudaLabels{};
  ACTS_CUDA_CHECK(
      cudaMallocAsync(&cudaLabels, numSpacepoints * sizeof(int), stream));

  std::size_t numberLabels = detail::connectedComponentsCuda(
      numEdges, cudaSrcPtr, cudaTgtPtr, numSpacepoints, cudaLabels, stream,
      m_cfg.useOneBlockImplementation);

  auto t2 = ACTS_TIME_STREAM_SYNC(Acts::Logging::DEBUG, execContext.stream);

  // TODO not sure why there is an issue that is not detected in the unit tests
  numberLabels += 1;

  std::vector<int> trackLabels(numSpacepoints);
  ACTS_CUDA_CHECK(cudaMemcpyAsync(trackLabels.data(), cudaLabels,
                                  numSpacepoints * sizeof(int),
                                  cudaMemcpyDeviceToHost, stream));

  // Free Memory
  ACTS_CUDA_CHECK(cudaFreeAsync(cudaLabels, stream));
  if (m_cfg.doJunctionRemoval) {
    ACTS_CUDA_CHECK(cudaFreeAsync(cudaSrcPtr, stream));
  }

  ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  ACTS_CUDA_CHECK(cudaGetLastError());

  auto t3 = ACTS_TIME_STREAM_SYNC(Acts::Logging::DEBUG, execContext.stream);

  ACTS_VERBOSE("Found " << numberLabels << " track candidates");

  std::vector<std::vector<int>> trackCandidates(numberLabels);

  for (const auto [label, id] : Acts::zip(trackLabels, spacepointIDs)) {
    trackCandidates[label].reserve(32);
    trackCandidates[label].push_back(id);
  }

  auto t4 = ACTS_TIME_STREAM_SYNC(Acts::Logging::DEBUG, execContext.stream);

  auto ms = [](auto t0, auto t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
  };

  ACTS_DEBUG("Junction removal:     " << ms(t0, t1) << " ms");
  ACTS_DEBUG("Connected components: " << ms(t1, t2) << " ms");
  ACTS_DEBUG("Copy to host:         " << ms(t2, t3) << " ms");
  ACTS_DEBUG("Build vector<vector>: " << ms(t3, t4) << " ms");

  return trackCandidates;
}

}  // namespace Acts
