// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/ExaTrkX/TracccFeatureCreation.hpp"
#include "Acts/Plugins/ExaTrkX/detail/CudaUtils.hpp"

namespace {

struct Radius {
  template <typename T>
  __device__ float operator()(std::size_t i, T *x, T *y, T *z = nullptr) {
    return std::hypot(x[i], y[i]);
  }
};

struct Phi {
  template <typename T>
  __device__ float operator()(std::size_t i, T *x, T *y, T *z = nullptr) {
    return std::atan2(y[i], x[i]);
  }
};

struct Eta {
  template <typename T>
  __device__ float operator()(std::size_t i, T *x, T *y, T *z) {
    return std::asinh(z[i] / std::hypot(x[i], y[i]));
  }
};

struct Z {
  template <typename T>
  __device__ float operator()(std::size_t i, T *x, T *y, T *z) {
    return z[i];
  }
};

__global__ void copyDoubleToStridedFloat(std::size_t n, const double *from,
                                         float *to, std::size_t stride,
                                         float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    to[i * stride] = static_cast<float>(from[i]) / scale;
  }
}

__global__ void rToStridedFloat(std::size_t n, const double *fromX,
                                const double *fromY, float *to,
                                std::size_t stride, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double r = std::hypot(fromX[i], fromY[i]);
    to[i * stride] = Radius{}(i, fromX, fromY) / scale;
  }
}

__global__ void phiToStridedFloat(std::size_t n, const double *fromX,
                                  const double *fromY, float *to,
                                  std::size_t stride, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    to[i * stride] = Phi{}(i, fromX, fromY) / scale;
  }
}

__global__ void etaToStridedFloat(std::size_t n, const double *fromX,
                                  const double *fromY, const double *fromZ,
                                  float *to, std::size_t stride, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    to[i * stride] = Eta{}(i, fromX, fromY, fromZ) / scale;
  }
}

template <typename F>
__global__ void copyClusterFeature(std::size_t n, unsigned clIndex,
                                   const unsigned *clIndex1,
                                   const unsigned *clIndex2, const float *clx,
                                   const float *cly, const float *clz,
                                   float *to, std::size_t stride, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  assert(clIndex == 0 || clIndex == 1);

  constexpr auto invalid =
      traccc::edm::spacepoint_collection::device::INVALID_MEASUREMENT_INDEX;

  unsigned cli = clIndex1[i];
  if (clIndex == 1 && clIndex2[i] != invalid) {
    cli = clIndex2[i];
  }

  to[i * stride] = F{}(cli, clx, cly, clz) / scale;
}

}  // namespace

namespace Acts {

Acts::Tensor<float> createInputTensor(
    const std::vector<std::string_view> &features,
    const std::vector<float> &featureScales,
    const traccc::edm::spacepoint_collection::const_device &sps,
    const ExecutionContext &execContext,
    const std::optional<vecmem::device_vector<float>> &clXglobal,
    const std::optional<vecmem::device_vector<float>> &clYglobal,
    const std::optional<vecmem::device_vector<float>> &clZglobal) {
  const dim3 blockSize = 1024;
  const dim3 gridSize = (sps.size() + blockSize.x - 1) / blockSize.x;

  assert(execContext.device.isCuda());
  assert(execContext.stream.has_value());
  auto stream = execContext.stream.value();

  auto tensor =
      Acts::Tensor<float>::Create({sps.size(), features.size()}, execContext);

  const std::size_t stride = features.size();

  for (std::size_t offset = 0; offset < features.size(); ++offset) {
    const auto &feat = features.at(offset);
    const auto scale = featureScales.at(offset);
    if (feat == "x") {
      copyDoubleToStridedFloat<<<gridSize, blockSize, 0, stream>>>(
          sps.size(), sps.x().data(), tensor.data() + offset, stride, scale);
    } else if (feat == "y") {
      copyDoubleToStridedFloat<<<gridSize, blockSize, 0, stream>>>(
          sps.size(), sps.y().data(), tensor.data() + offset, stride, scale);
    } else if (feat == "z") {
      copyDoubleToStridedFloat<<<gridSize, blockSize, 0, stream>>>(
          sps.size(), sps.z().data(), tensor.data() + offset, stride, scale);
    } else if (feat == "r") {
      rToStridedFloat<<<gridSize, blockSize, 0, stream>>>(
          sps.size(), sps.x().data(), sps.y().data(), tensor.data() + offset,
          stride, scale);
    } else if (feat == "phi") {
      phiToStridedFloat<<<gridSize, blockSize, 0, stream>>>(
          sps.size(), sps.x().data(), sps.y().data(), tensor.data() + offset,
          stride, scale);
    } else if (feat == "eta") {
      etaToStridedFloat<<<gridSize, blockSize, 0, stream>>>(
          sps.size(), sps.x().data(), sps.y().data(), sps.z().data(),
          tensor.data() + offset, stride, scale);
    } else if (feat == "cl1_r" || feat == "cl2_r") {
      unsigned cl = feat == "cl1_r" ? 0 : 1;
      copyClusterFeature<Radius><<<gridSize, blockSize, 0, stream>>>(
          sps.size(), cl, sps.measurement_index_1().data(),
          sps.measurement_index_2().data(), clXglobal.value().data(),
          clYglobal.value().data(), clZglobal.value().data(),
          tensor.data() + offset, stride, scale);
    } else if (feat == "cl1_phi" || feat == "cl2_phi") {
      unsigned cl = feat == "cl1_phi" ? 0 : 1;
      copyClusterFeature<Phi><<<gridSize, blockSize, 0, stream>>>(
          sps.size(), cl, sps.measurement_index_1().data(),
          sps.measurement_index_2().data(), clXglobal.value().data(),
          clYglobal.value().data(), clZglobal.value().data(),
          tensor.data() + offset, stride, scale);
    } else if (feat == "cl1_eta" || feat == "cl2_eta") {
      unsigned cl = feat == "cl1_eta" ? 0 : 1;
      copyClusterFeature<Eta><<<gridSize, blockSize, 0, stream>>>(
          sps.size(), cl, sps.measurement_index_1().data(),
          sps.measurement_index_2().data(), clXglobal.value().data(),
          clYglobal.value().data(), clZglobal.value().data(),
          tensor.data() + offset, stride, scale);
    } else if (feat == "cl1_z" || feat == "cl2_z") {
      unsigned cl = feat == "cl1_z" ? 0 : 1;
      copyClusterFeature<Z><<<gridSize, blockSize, 0, stream>>>(
          sps.size(), cl, sps.measurement_index_1().data(),
          sps.measurement_index_2().data(), clXglobal.value().data(),
          clYglobal.value().data(), clZglobal.value().data(),
          tensor.data() + offset, stride, scale);
    } else {
      throw std::runtime_error("Unknown feature: " + std::string(feat));
    }
    ACTS_CUDA_CHECK(cudaGetLastError());
    ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));

    ++offset;
  }

  return tensor;
}

}  // namespace Acts
