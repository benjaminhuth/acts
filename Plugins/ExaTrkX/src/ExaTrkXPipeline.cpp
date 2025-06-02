// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/ExaTrkX/ExaTrkXPipeline.hpp"

#include "Acts/Utilities/Helpers.hpp"
#include "Acts/Utilities/Zip.hpp"

#include <span>

#ifdef ACTS_EXATRKX_WITH_CUDA
#include "Acts/Plugins/ExaTrkX/detail/CudaUtils.hpp"

namespace {
struct CudaStreamGuard {
  cudaStream_t stream{};
  CudaStreamGuard() { ACTS_CUDA_CHECK(cudaStreamCreate(&stream)); }
  ~CudaStreamGuard() {
    ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
    ACTS_CUDA_CHECK(cudaStreamDestroy(stream));
  }
};
}  // namespace
#endif

namespace Acts {

namespace detail {

std::vector<std::vector<int>> unpackTrackLabels(
    const Acts::Tensor<int> &trackLabels, std::size_t numberLabels,
    const std::vector<int> &spacepointIDs) {
  if (trackLabels.size() == 0) {
    return {};
  }

  std::vector<std::vector<int>> trackCandidates(numberLabels);

  std::span<const int> trackLabelSpan(trackLabels.data(), trackLabels.size());
  for (const auto [label, id] : Acts::zip(trackLabelSpan, spacepointIDs)) {
    trackCandidates.at(label).push_back(id);
  }

  return trackCandidates;
}

}  // namespace detail

ExaTrkXPipeline::ExaTrkXPipeline(
    std::shared_ptr<GraphConstructionBase> graphConstructor,
    std::vector<std::shared_ptr<EdgeClassificationBase>> edgeClassifiers,
    std::shared_ptr<TrackBuildingBase> trackBuilder,
    std::unique_ptr<const Acts::Logger> logger)
    : m_logger(std::move(logger)),
      m_graphConstructor(std::move(graphConstructor)),
      m_edgeClassifiers(std::move(edgeClassifiers)),
      m_trackBuilder(std::move(trackBuilder)) {
  if (!m_graphConstructor) {
    throw std::invalid_argument("Missing graph construction module");
  }
  if (!m_trackBuilder) {
    throw std::invalid_argument("Missing track building module");
  }
  if (m_edgeClassifiers.empty() ||
      rangeContainsValue(m_edgeClassifiers, nullptr)) {
    throw std::invalid_argument("Missing graph construction module");
  }
}

std::vector<std::vector<int>> ExaTrkXPipeline::run(
    const std::vector<float> &features,
    const std::vector<std::uint64_t> &moduleIds,
    const std::vector<int> &spacepointIDs, Acts::Device device,
    const ExaTrkXHook &hook, ExaTrkXTiming *timing) const {
  ExecutionContext execCtx;
  execCtx.device = device;
#ifdef ACTS_EXATRKX_WITH_CUDA
  std::optional<CudaStreamGuard> streamGuard;
  if (execCtx.device.type == Acts::Device::Type::eCUDA) {
    streamGuard.emplace();
    execCtx.stream = streamGuard->stream;
  }
#endif

  const auto numSpacepoints = spacepointIDs.size();
  const auto numFeatures = features.size() / numSpacepoints;

  auto nodeTensorHost = Acts::Tensor<float>::Create(
      {numSpacepoints, numFeatures}, {Acts::Device::Cpu(), {}});
  std::copy(features.begin(), features.end(), nodeTensorHost.data());
  auto nodeFeaturesTarget = nodeTensorHost.clone(execCtx);

  std::optional<Acts::Tensor<std::uint64_t>> moduleIdsTarget;
  if (!moduleIds.empty()) {
    auto moduleIdsHost = Acts::Tensor<std::uint64_t>::Create(
        {numSpacepoints, 1}, {Acts::Device::Cpu(), {}});
    std::copy(moduleIds.begin(), moduleIds.end(), moduleIdsHost.data());
    moduleIdsTarget.emplace(moduleIdsHost.clone(execCtx));
  }

  auto [labelTensor, nLabels] =
      run(std::move(nodeFeaturesTarget), std::move(moduleIdsTarget), execCtx,
          hook, timing);

  auto labelTensorHost = labelTensor.clone({Device::Cpu(), execCtx.stream});
  return detail::unpackTrackLabels(labelTensorHost, nLabels, spacepointIDs);
}

std::pair<Acts::Tensor<int>, std::size_t> ExaTrkXPipeline::run(
    Acts::Tensor<float> nodeFeatures,
    std::optional<Acts::Tensor<std::uint64_t>> moduleIds,
    const ExecutionContext &execCtx, const ExaTrkXHook &hook,
    ExaTrkXTiming *timing) const {
  try {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto tensors = (*m_graphConstructor)(std::move(nodeFeatures),
                                         std::move(moduleIds), execCtx);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (timing != nullptr) {
      timing->graphBuildingTime = t1 - t0;
    }

    hook(tensors, execCtx);

    if (timing != nullptr) {
      timing->classifierTimes.clear();
    }

    for (const auto &edgeClassifier : m_edgeClassifiers) {
      t0 = std::chrono::high_resolution_clock::now();
      tensors = (*edgeClassifier)(std::move(tensors), execCtx);
      t1 = std::chrono::high_resolution_clock::now();

      if (timing != nullptr) {
        timing->classifierTimes.push_back(t1 - t0);
      }

      hook(tensors, execCtx);
    }

    t0 = std::chrono::high_resolution_clock::now();
    auto res = (*m_trackBuilder)(std::move(tensors), execCtx);
    t1 = std::chrono::high_resolution_clock::now();

    if (timing != nullptr) {
      timing->trackBuildingTime = t1 - t0;
    }

    return res;
  } catch (Acts::NoEdgesError &) {
    ACTS_DEBUG("No edges left in GNN pipeline, return 0 track candidates");
    if (timing != nullptr) {
      while (timing->classifierTimes.size() < m_edgeClassifiers.size()) {
        timing->classifierTimes.push_back({});
      }
    }
    return {Acts::Tensor<int>::Create({0ul, 0ul}, execCtx), 0};
  }
}

}  // namespace Acts
