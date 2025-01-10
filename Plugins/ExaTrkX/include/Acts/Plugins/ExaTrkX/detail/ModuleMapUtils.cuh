// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

#include <cuda_runtime_api.h>

namespace Acts::detail {

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

inline __global__ void setHitId(std::size_t nHits, std::uint64_t *hit_ids) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nHits) {
    return;
  }
  hit_ids[i] = i;
}

__global__ void remapEdges(std::size_t nEdges, int *srcNodes, int *tgtNodes,
                           const std::uint64_t *hit_ids, std::size_t nAllNodes,
                           std::size_t nCompressedNodes) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nEdges) {
    return;
  }

  srcNodes[i] = hit_ids[srcNodes[i]];
  tgtNodes[i] = hit_ids[tgtNodes[i]];
}

template <typename T>
void copyFromDeviceAndPrint(T *data, std::size_t size, std::string_view name) {
  std::vector<T> data_cpu(size);
  cudaMemcpy(data_cpu.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost);
  std::cout << name << "[" << size << "]: ";
  for (int i = 0; i < size; ++i) {
    std::cout << data_cpu.at(i) << "  ";
  }
  std::cout << std::endl;
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

inline void __global__ mapModuleIdsToNbHits(int *nbHitsOnModule,
                                            std::size_t nHits,
                                            const std::uint64_t *moduleIds,
                                            std::size_t moduleMapSize,
                                            const std::uint64_t *moduleMapKey,
                                            const int *moduleMapVal) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= nHits) {
    return;
  }

  auto mId = moduleIds[i];

  // bisect moduleMapKey to find mId
  int left = 0;
  int right = moduleMapSize - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (moduleMapKey[mid] == mId) {
      // atomic add 1 to hitIndice[moduleMapVal[mid]]
      atomicAdd(&nbHitsOnModule[moduleMapVal[mid]], 1);
      return;
    }
    if (moduleMapKey[mid] < mId) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
}

/// Counting kernel to allow counting the edges
template <class T>
__global__ void count_doublet_edges(
    int nb_doublets, const int *modules1, const int *modules2, const T *R,
    const T *z, const T *eta, const T *phi, T *z0_min, T *z0_max, T *deta_min,
    T *deta_max, T *phi_slope_min, T *phi_slope_max, T *dphi_min, T *dphi_max,
    const int *indices, T pi, T max, int *nb_edges_total,
    int *nb_edges_doublet) {
  // loop over module1 SP
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets)
    return;

  int module1 = modules1[i];
  int module2 = modules2[i];
  int edges = 0;

  for (int k = indices[module1]; k < indices[module1 + 1]; k++) {
    T phi_SP1 = phi[k];
    T eta_SP1 = eta[k];
    T R_SP1 = R[k];
    T z_SP1 = z[k];

    for (int l = indices[module2]; l < indices[module2 + 1]; l++) {
      T z0, phi_slope, deta, dphi;
      hits_geometric_cuts<T>(R_SP1, R[l], z_SP1, z[l], eta_SP1, eta[l], phi_SP1,
                             phi[l], pi, max, z0, phi_slope, deta, dphi);

      if (apply_geometric_cuts(i, z0, phi_slope, deta, dphi, z0_min, z0_max,
                               deta_min, deta_max, phi_slope_min, phi_slope_max,
                               dphi_min, dphi_max)) {
        edges++;
      }
    }
  }

  // increase global and local counter
  nb_edges_doublet[i] = edges;
  atomicAdd(nb_edges_total, edges);
}

/// New kernel that use precounted number of edges
template <class T>
__global__ void doublet_cuts_new(int nb_doublets, const int *modules1,
                                 const int *modules2, const T *R, const T *z,
                                 const T *eta, const T *phi, T *z0_min,
                                 T *z0_max, T *deta_min, T *deta_max,
                                 T *phi_slope_min, T *phi_slope_max,
                                 T *dphi_min, T *dphi_max, const int *indices,
                                 T pi, T max, int *M1_SP, int *M2_SP,
                                 const int *nb_edges) {
  // loop over module1 SP
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb_doublets)
    return;

  int module1 = modules1[i];
  int module2 = modules2[i];
  int edges = nb_edges[i];

  for (int k = indices[module1]; k < indices[module1 + 1]; k++) {
    T phi_SP1 = phi[k];
    T eta_SP1 = eta[k];
    T R_SP1 = R[k];
    T z_SP1 = z[k];

    for (int l = indices[module2]; l < indices[module2 + 1]; l++) {
      T z0, phi_slope, deta, dphi;
      hits_geometric_cuts<T>(R_SP1, R[l], z_SP1, z[l], eta_SP1, eta[l], phi_SP1,
                             phi[l], pi, max, z0, phi_slope, deta, dphi);

      if (apply_geometric_cuts(i, z0, phi_slope, deta, dphi, z0_min, z0_max,
                               deta_min, deta_max, phi_slope_min, phi_slope_max,
                               dphi_min, dphi_max)) {
        M1_SP[edges] = k;
        M2_SP[edges] = l;
        edges++;
      }
    }
  }
}

}  // namespace Acts::detail
