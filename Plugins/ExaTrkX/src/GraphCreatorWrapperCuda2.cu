// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/ExaTrkX/detail/GraphCreatorWrapper.hpp"

#include <CUDA_graph_creator_new>
#include <TTree_hits>
#include <algorithm>
#include <graph>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/torch.h>

namespace {

inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA error: " << cudaGetErrorString(code) << ", " << file << ":"
       << line;
    // throw std::runtime_error(ss.str());
    std::cout << ss.str() << std::endl;
  }
  cudaDeviceSynchronize();
}

#define CUDA_CHECK(ans) \
  { cudaAssert((ans), __FILE__, __LINE__); }

template <typename T>
cudaError_t cudaMallocT(T **ptr, std::size_t size) {
  return cudaMalloc((void **)ptr, size);
}

template <class T>
__global__ void computeXandY(std::size_t nbHits, T *cuda_x, T *cuda_y,
                             const T *cuda_R, const T *cuda_phi) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nbHits) {
    return;
  }

  double r = cuda_R[i];
  double phi = cuda_phi[i];

  cuda_x[i] = r * std::cos(phi);
  cuda_y[i] = r * std::sin(phi);
}

template <class T>
__global__ void rescaleFeature(std::size_t nbHits, T *data, T scale) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nbHits) {
    return;
  }

  data[i] *= scale;
}

constexpr float g_pi = 3.141592654f;

template <typename T>
__device__ T resetAngle(T angle) {
  if (angle > g_pi) {
    return angle - 2.f * g_pi;
  }
  if (angle < -g_pi) {
    return angle + 2.f * g_pi;
  }
  return angle;
};

template <typename T>
__global__ void makeEdgeFeatures(std::size_t nEdges, const int *srcEdges,
                                 const int *tgtEdges, std::size_t nNodeFeatures,
                                 const T *nodeFeatures, T *edgeFeatures) {
  enum NodeFeatures { r = 0, phi, z, eta };
  constexpr static int nEdgeFeatures = 6;

  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nEdges) {
    return;
  }

  const int src = srcEdges[i];
  const int tgt = tgtEdges[i];

  const T *srcNodeFeatures = nodeFeatures + src * nNodeFeatures;
  const T *tgtNodeFeatures = nodeFeatures + tgt * nNodeFeatures;

  T dr = tgtNodeFeatures[r] - srcNodeFeatures[r];
  T dphi =
      resetAngle(g_pi * (tgtNodeFeatures[phi] - srcNodeFeatures[phi])) / g_pi;
  T dz = tgtNodeFeatures[z] - srcNodeFeatures[z];
  T deta = tgtNodeFeatures[eta] - srcNodeFeatures[eta];
  T phislope = 0.0;
  T rphislope = 0.0;

  if (dr != 0.0) {
    phislope = std::clamp(dphi / dr, -100.f, 100.f);
    T avgR = T{0.5} * (tgtNodeFeatures[r] + srcNodeFeatures[r]);
    rphislope = avgR * phislope;
  }

  T *efPtr = edgeFeatures + i * nEdgeFeatures;
  efPtr[0] = dr;
  efPtr[1] = dphi;
  efPtr[2] = dz;
  efPtr[3] = deta;
  efPtr[4] = phislope;
  efPtr[5] = rphislope;
}

template <typename T>
void copyFromDeviceAndPrint(T *data, std::size_t size, std::string_view name) {
  std::vector<T> data_cpu(size);
  CUDA_CHECK(cudaMemcpy(data_cpu.data(), data, size * sizeof(T),
                        cudaMemcpyDeviceToHost));
  std::cout << name << "[" << size << "]: ";
  for (int i = 0; i < size; ++i) {
    std::cout << data_cpu.at(i) << "  ";
  }
  std::cout << std::endl;
}

}  // namespace

template <typename T>
void printPrefixSum(const std::vector<T> &vec) {
  T prev = vec.front();
  std::cout << 0 << ":" << vec[0] << "  ";
  for (auto i = 0ul; i < vec.size(); ++i) {
    if (vec[i] != prev) {
      std::cout << i << ":" << vec[i] << "  ";
      prev = vec[i];
    }
  }
}

namespace Acts::detail {

GraphCreatorWrapperCuda::GraphCreatorWrapperCuda(const std::string &path,
                                                 int device, int blocks) {
  m_graphCreator =
      std::make_unique<CUDA_graph_creator<float>>(blocks, device, path);
}

GraphCreatorWrapperCuda::~GraphCreatorWrapperCuda() {}

std::pair<at::Tensor, at::Tensor> GraphCreatorWrapperCuda::build(
    const std::vector<float> &features,
    const std::vector<std::uint64_t> &moduleIds, const Acts::Logger &logger) {
  using GC = CUDA_graph_creator<float>;
  const auto nHits = moduleIds.size();
  const auto nFeatures = features.size() / moduleIds.size();

  dim3 blockDim = 512;
  dim3 gridDim = (nHits + blockDim.x - 1) / blockDim.x;

  dim3 block_dim = blockDim;
  dim3 grid_dim = gridDim;

  // TODO understand this algorithm
  std::vector<int> hit_indice;
  {
    const auto &module_map =
        m_graphCreator->get_module_map_doublet().module_map();
    std::vector<int> nb_hits(module_map.size(), 0);
    std::vector<int> hits_bool_mask(nHits, 0);

    for (auto i = 0ul; i < nHits; ++i) {
      auto it = module_map.find(moduleIds.at(i));
      if (it != module_map.end() && hits_bool_mask.at(i) == 0) {
        hits_bool_mask.at(i) = 1;
        nb_hits[it->second] += 1;
      }
    }

    hit_indice.push_back(0);
    for (std::size_t i = 0; i < nb_hits.size(); i++) {
      hit_indice.push_back(hit_indice[i] + nb_hits[i]);
    }
  }

  assert(!std::all_of(hit_indice.begin(), hit_indice.end(),
                      [](auto v) { return v == 0; }));

  // std::cout << "my prefix sum (" << hit_indice.size() << "): ";
  // printPrefixSum(hit_indice);
  // std::cout << std::endl;

  float *cudaNodeFeatures{};
  cudaMalloc(&cudaNodeFeatures, features.size() * sizeof(float));
  cudaMemcpy(cudaNodeFeatures, features.data(), features.size() * sizeof(float),
             cudaMemcpyHostToDevice);

#if 1
  CUDA_hit_data<float> inputData;
  inputData.size = nHits;

  std::size_t rOffset = 0;
  std::size_t phiOffset = 1;
  std::size_t zOffset = 2;
  std::size_t etaOffset = 3;

  // const auto srcStride = sizeof(float) * nFeatures;
  // const auto dstStride = sizeof(float);  // contiguous in destination
  // const auto width = sizeof(float);      // only copy 1 column
  // const auto height = nHits;

  std::vector<float> hostR(nHits), hostPhi(nHits), hostZ(nHits), hostEta(nHits);
  for (auto i = 0ul; i < nHits; ++i) {
    hostR.at(i) = features.at(i * nFeatures + rOffset);
    hostPhi.at(i) = features.at(i * nFeatures + phiOffset);
    hostZ.at(i) = features.at(i * nFeatures + zOffset);
    hostEta.at(i) = features.at(i * nFeatures + etaOffset);
  }

  CUDA_CHECK(cudaMallocT(&inputData.cuda_R, nHits * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(inputData.cuda_R, hostR.data(), nHits * sizeof(float),
                        cudaMemcpyHostToDevice));
  // CUDA_CHECK(cudaMemcpy2D(inputData.cuda_R, dstStride,
  //                         features.data() + rOffset, srcStride, width,
  //                         height, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMallocT(&inputData.cuda_phi, nHits * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(inputData.cuda_phi, hostPhi.data(),
                        nHits * sizeof(float), cudaMemcpyHostToDevice));
  // CUDA_CHECK(cudaMemcpy2D(inputData.cuda_phi, dstStride,
  //                         features.data() + phiOffset, srcStride, width,
  //                         height, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMallocT(&inputData.cuda_z, nHits * sizeof(float)))
  CUDA_CHECK(cudaMemcpy(inputData.cuda_z, hostZ.data(), nHits * sizeof(float),
                        cudaMemcpyHostToDevice));
  // CUDA_CHECK(cudaMemcpy2D(inputData.cuda_z, dstStride,
  //                         features.data() + zOffset, srcStride, width,
  //                         height, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMallocT(&inputData.cuda_eta, nHits * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(inputData.cuda_eta, hostEta.data(),
                        nHits * sizeof(float), cudaMemcpyHostToDevice));
  // CUDA_CHECK(cudaMemcpy2D(inputData.cuda_eta, dstStride,
  //                         features.data() + etaOffset, srcStride, width,
  //                         height, cudaMemcpyHostToDevice));

  rescaleFeature<<<gridDim, blockDim>>>(nHits, inputData.cuda_z, 1000.f);
  CUDA_CHECK(cudaGetLastError());
  rescaleFeature<<<gridDim, blockDim>>>(nHits, inputData.cuda_R, 1000.f);
  CUDA_CHECK(cudaGetLastError());
  rescaleFeature<<<gridDim, blockDim>>>(nHits, inputData.cuda_phi,
                                        3.141592654f);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMallocT(&inputData.cuda_x, nHits * sizeof(float)));
  CUDA_CHECK(cudaMallocT(&inputData.cuda_y, nHits * sizeof(float)));

  computeXandY<<<gridDim, blockDim>>>(nHits, inputData.cuda_x, inputData.cuda_y,
                                      inputData.cuda_R, inputData.cuda_phi);
  CUDA_CHECK(cudaGetLastError());

  int *cuda_hit_indice = nullptr;
  CUDA_CHECK(cudaMallocT(&cuda_hit_indice, hit_indice.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(cuda_hit_indice, hit_indice.data(),
                        hit_indice.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaGetLastError());
#else
  hits<float> hitsCollection(false, false);

  for (auto i = 0ul; i < nHits; ++i) {
    // TODO Use std::span when we move to C++20
    const float *hitFeatures = features.data() + i * nFeatures;

    int hitId = static_cast<int>(i);

    // Needs to be rescaled because ModuleMapGraph expects unscaled features
    float r = hitFeatures[0] * 1000.f;         // rScale;
    float phi = hitFeatures[1] * 3.141592654;  // phiScale;
    float z = hitFeatures[2] * 1000.f;         // zScale;

    float x = r * std::cos(phi);
    float y = r * std::sin(phi);

    std::uint64_t particleId = 0;  // We do not know
    std::uint64_t moduleId = moduleIds[i];
    std::string hardware = "";      // now hardware
    int barrelEndcap = 0;           // unclear, is this a flag???
    std::uint64_t particleID1 = 0;  // unclear
    std::uint64_t particleID2 = 0;  // unclear

    hit<float> hit(hitId, x, y, z, particleId, moduleId, hardware, barrelEndcap,
                   particleID1, particleID2);

    hitsCollection += hit;
  }

  TTree_hits<float> hitsTree = hitsCollection;

  CUDA_TTree_hits<float> input_hits;
  std::string event = "0";
  input_hits.add_event(event, hitsTree,
                       m_graphCreator->get_module_map_doublet().module_map());
  input_hits.HostToDevice();

  TTree_hits_constants<<<grid_dim, block_dim>>>(
      input_hits.size(), input_hits.cuda_x(), input_hits.cuda_y(),
      input_hits.cuda_z(), input_hits.cuda_R(), input_hits.cuda_eta(),
      input_hits.cuda_phi());

  CUDA_hit_data<float> input_hit_data;
  input_hit_data.size = input_hits.size();
  input_hit_data.cuda_R = input_hits.cuda_R();
  input_hit_data.cuda_z = input_hits.cuda_z();
  input_hit_data.cuda_eta = input_hits.cuda_eta();
  input_hit_data.cuda_phi = input_hits.cuda_phi();
  input_hit_data.cuda_x = input_hits.cuda_x();
  input_hit_data.cuda_y = input_hits.cuda_y();

  auto &inputData = input_hit_data;
  int *cuda_hit_indice = input_hits.cuda_hit_indice();
#endif

  /*
    std::vector<int> data_cpu(
        m_graphCreator->get_module_map_doublet().module_map().size() + 1);
    CUDA_CHECK(
        cudaMemcpy(data_cpu.data(), cuda_hit_indice,
                   (m_graphCreator->get_module_map_doublet().module_map().size()
    + 1) * sizeof(int), cudaMemcpyDeviceToHost)); assert(data_cpu.size() ==
    hit_indice.size()); std::cout << "MMG prefix sum: ";
    printPrefixSum(data_cpu);
    std::cout << std::endl;  assert(data_cpu == hit_indice);

    copyFromDeviceAndPrint(inputData.cuda_x, nHits, "cuda_x");
    copyFromDeviceAndPrint(inputData.cuda_y, nHits, "cuda_y");
    copyFromDeviceAndPrint(inputData.cuda_z, nHits, "cuda_z");
    copyFromDeviceAndPrint(inputData.cuda_R, nHits, "cuda_R");
    copyFromDeviceAndPrint(inputData.cuda_phi, nHits, "cuda_phi");
    copyFromDeviceAndPrint(inputData.cuda_eta, nHits, "cuda_eta");
  */

  cudaDeviceSynchronize();
  auto edgeData = m_graphCreator->build(inputData, cuda_hit_indice);
  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();

  std::cout << "Made " << edgeData.size << " edges" << std::endl;
  assert(edgeData.size > 0 && edgeData.size < 100'000'000);

  int *edgeIndexPtr{};
  CUDA_CHECK(cudaMallocT(&edgeIndexPtr, 2 * edgeData.size * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(edgeIndexPtr, edgeData.cuda_graph_M1_hits,
                        edgeData.size * sizeof(int), cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(edgeIndexPtr + edgeData.size,
                        edgeData.cuda_graph_M2_hits,
                        edgeData.size * sizeof(int), cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  auto edgeIndex =
      torch::from_blob(edgeIndexPtr, 2 * static_cast<long>(edgeData.size),
                       at::TensorOptions().device(at::kCUDA).dtype(at::kInt));

  edgeIndex =
      edgeIndex.reshape({2, static_cast<long>(edgeData.size)}).to(at::kLong);

  float *edgeFeaturePtr{};
  cudaMallocT(&edgeFeaturePtr, 6 * edgeData.size * sizeof(float));

  dim3 gridDimEdges = (edgeData.size + blockDim.x - 1) / blockDim.x;
  makeEdgeFeatures<<<gridDimEdges, blockDim>>>(
      edgeData.size, edgeData.cuda_graph_M1_hits, edgeData.cuda_graph_M2_hits,
      nFeatures, cudaNodeFeatures, edgeFeaturePtr);
  /*
    CUDA_CHECK(cudaMemcpy2D(edgeFeaturePtr, 6 * sizeof(float),
                            edgeData.cuda_graph_dR, sizeof(float),
    sizeof(float), edgeData.size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(
        edgeFeaturePtr + 1, 6 * sizeof(float), edgeData.cuda_graph_dphi,
        sizeof(float), sizeof(float), edgeData.size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(edgeFeaturePtr + 2, 6 * sizeof(float),
                            edgeData.cuda_graph_dz, sizeof(float),
    sizeof(float), edgeData.size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(
        edgeFeaturePtr + 3, 6 * sizeof(float), edgeData.cuda_graph_deta,
        sizeof(float), sizeof(float), edgeData.size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(
        edgeFeaturePtr + 4, 6 * sizeof(float), edgeData.cuda_graph_phi_slope,
        sizeof(float), sizeof(float), edgeData.size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(
        edgeFeaturePtr + 5, 6 * sizeof(float), edgeData.cuda_graph_r_phi_slope,
        sizeof(float), sizeof(float), edgeData.size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
  */

  auto edgeFeatures =
      torch::from_blob(edgeFeaturePtr, {static_cast<long>(edgeData.size), 6},
                       //[](void *ptr) { CUDA_CHECK(cudaFree(ptr)); },
                       at::TensorOptions().device(at::kCUDA).dtype(at::kFloat));

  auto edgeFeaturesNew = edgeFeatures.clone();
  // copyFromDeviceAndPrint(edgeData.cuda_graph_dR, edgeData.size,
  // "cuda_graph_dR");
  //std::cout << "edgeIndex:\n" << edgeIndex << std::endl;
  //std::cout << "edgeFeatures:\n" << edgeFeaturesNew << std::endl;
  CUDA_CHECK(cudaDeviceSynchronize());

  // std::cout << "dR (0->1): " << hostR.at(1) - hostR.at(0) << std::endl;
  // std::cout << "dR (0->2): " << hostR.at(2) - hostR.at(0) << std::endl;

  return {edgeIndex.clone(), edgeFeaturesNew};
}

}  // namespace Acts::detail
