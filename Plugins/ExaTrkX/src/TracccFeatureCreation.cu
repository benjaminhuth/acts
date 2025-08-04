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
  __device__ float operator()(std::size_t i, const T *x, const T *y,
                              const T *z = nullptr) const {
    return std::hypot(x[i], y[i]);
  }
  template <typename T>
  __device__ float operator()(T x, T y, T z) const {
    return std::hypot(x, y);
  }
};

struct Phi {
  template <typename T>
  __device__ float operator()(std::size_t i, const T *x, const T *y,
                              const T *z = nullptr) const {
    return std::atan2(y[i], x[i]);
  }
  template <typename T>
  __device__ float operator()(T x, T y, T z) const {
    return std::atan2(y, x);
  }
};

struct Eta {
  template <typename T>
  __device__ float operator()(std::size_t i, const T *x, const T *y,
                              const T *z) const {
    return std::asinh(z[i] / std::hypot(x[i], y[i]));
  }
  template <typename T>
  __device__ float operator()(T x, T y, T z) const {
    return std::asinh(z / std::hypot(x, y));
  }
};

struct ForwardZ {
  template <typename T>
  __device__ float operator()(std::size_t i, const T *x, const T *y,
                              const T *z) const {
    return z[i];
  }
  template <typename T>
  __device__ float operator()(T x, T y, T z) const {
    return z;
  }
};

struct ForwardX {
  template <typename T>
  __device__ float operator()(std::size_t i, const T *x, const T *y,
                              const T *z) const {
    return x[i];
  }
  template <typename T>
  __device__ float operator()(T x, T y, T z) {
    return x;
  }
};

struct ForwardY {
  template <typename T>
  __device__ float operator()(std::size_t i, const T *x, const T *y,
                              const T *z) const {
    return y[i];
  }
  template <typename T>
  __device__ float operator()(T x, T y, T z) const {
    return y;
  }
};

template <typename F>
__global__ void toStridedFloat(
    traccc::edm::spacepoint_collection::const_view spsView, float *to,
    std::size_t stride, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  traccc::edm::spacepoint_collection::const_device sps(spsView);

  if (i >= sps.size()) {
    return;
  }

  to[i * stride] = F{}(sps.at(i).x(), sps.at(i).y(), sps.at(i).z()) / scale;
}

template <typename F>
__global__ void copyClusterFeature(
    traccc::edm::spacepoint_collection::const_view spsView, unsigned clIndex,
    vecmem::data::vector_view<const float> clxView,
    vecmem::data::vector_view<const float> clyView,
    vecmem::data::vector_view<const float> clzView, float *to,
    std::size_t stride, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  traccc::edm::spacepoint_collection::const_device sps(spsView);
  vecmem::device_vector<const float> clx(clxView);
  vecmem::device_vector<const float> cly(clyView);
  vecmem::device_vector<const float> clz(clzView);

  if (i >= sps.size()) {
    return;
  }

  assert(clIndex == 0 || clIndex == 1);

  constexpr auto invalid =
      traccc::edm::spacepoint_collection::device::INVALID_MEASUREMENT_INDEX;

  unsigned cli = sps.measurement_index_1()[i];
  if (clIndex == 1 && sps.measurement_index_2()[i] != invalid) {
    cli = sps.measurement_index_2()[i];
  }

  to[i * stride] = F{}(cli, clx.data(), cly.data(), clz.data()) / scale;
}

}  // namespace

namespace Acts {

Acts::Tensor<float> createInputTensor(
    const std::vector<std::string_view> &features,
    const std::vector<float> &featureScales,
    traccc::edm::spacepoint_collection::const_view sps,
    const ExecutionContext &execContext,
    std::optional<vecmem::data::vector_view<const float>> clXglobal,
    std::optional<vecmem::data::vector_view<const float>> clYglobal,
    std::optional<vecmem::data::vector_view<const float>> clZglobal) {
  const dim3 blockSize = 1024;
  const dim3 gridSize = (sps.capacity() + blockSize.x - 1) / blockSize.x;

  assert(execContext.device.isCuda());
  assert(execContext.stream.has_value());
  auto stream = execContext.stream.value();

  auto tensor = Acts::Tensor<float>::Create({sps.capacity(), features.size()},
                                            execContext);

  const std::size_t stride = features.size();

  for (std::size_t offset = 0; offset < features.size(); ++offset) {
    const auto &feat = features.at(offset);
    const auto scale = featureScales.at(offset);
    float *ptr = tensor.data() + offset;
    if (feat == "x") {
      toStridedFloat<ForwardX>
          <<<gridSize, blockSize, 0, stream>>>(sps, ptr, stride, scale);
    } else if (feat == "y") {
      toStridedFloat<ForwardY>
          <<<gridSize, blockSize, 0, stream>>>(sps, ptr, stride, scale);
    } else if (feat == "z") {
      toStridedFloat<ForwardZ>
          <<<gridSize, blockSize, 0, stream>>>(sps, ptr, stride, scale);
    } else if (feat == "r") {
      toStridedFloat<Radius>
          <<<gridSize, blockSize, 0, stream>>>(sps, ptr, stride, scale);
    } else if (feat == "phi") {
      toStridedFloat<Phi>
          <<<gridSize, blockSize, 0, stream>>>(sps, ptr, stride, scale);
    } else if (feat == "eta") {
      toStridedFloat<Eta>
          <<<gridSize, blockSize, 0, stream>>>(sps, ptr, stride, scale);
    } else if (feat == "cl1_r" || feat == "cl2_r") {
      unsigned cl = feat == "cl1_r" ? 0 : 1;
      copyClusterFeature<Radius><<<gridSize, blockSize, 0, stream>>>(
          sps, cl, clXglobal.value(), clYglobal.value(), clZglobal.value(), ptr,
          stride, scale);
    } else if (feat == "cl1_phi" || feat == "cl2_phi") {
      unsigned cl = feat == "cl1_phi" ? 0 : 1;
      copyClusterFeature<Phi><<<gridSize, blockSize, 0, stream>>>(
          sps, cl, clXglobal.value(), clYglobal.value(), clZglobal.value(), ptr,
          stride, scale);
    } else if (feat == "cl1_eta" || feat == "cl2_eta") {
      unsigned cl = feat == "cl1_eta" ? 0 : 1;
      copyClusterFeature<Eta><<<gridSize, blockSize, 0, stream>>>(
          sps, cl, clXglobal.value(), clYglobal.value(), clZglobal.value(), ptr,
          stride, scale);
    } else if (feat == "cl1_z" || feat == "cl2_z") {
      unsigned cl = feat == "cl1_z" ? 0 : 1;
      copyClusterFeature<ForwardZ><<<gridSize, blockSize, 0, stream>>>(
          sps, cl, clXglobal.value(), clYglobal.value(), clZglobal.value(), ptr,
          stride, scale);
    } else {
      throw std::runtime_error("Unknown feature: " + std::string(feat));
    }
    ACTS_CUDA_CHECK(cudaGetLastError());
    ACTS_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  return tensor;
}

}  // namespace Acts
