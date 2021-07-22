// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/MultiComponentBoundTrackParameters.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/TrackParameters.hpp"

#include "GsfUtils.hpp"

namespace Acts {

namespace detail {

/// @brief computes the Kullback-Leibler distance between two components as shown in https://arxiv.org/abs/2001.00727v1, while ignoring the weights as done in Athena
template <typename source_link_t>
auto computeKLDistance(const GsfComponentCache<source_link_t> &a,
                       const GsfComponentCache<source_link_t> &b) {
  const auto parsA = a.predictedPars[eBoundQOverP];
  const auto parsB = b.predictedPars[eBoundQOverP];
  const auto covA = (*a.predictedCov)(eBoundQOverP, eBoundQOverP);
  const auto covB = (*b.predictedCov)(eBoundQOverP, eBoundQOverP);

  const auto kl = covA * (1 / covB) + covB * (1 / covA) +
                  (parsA - parsB) * (1 / covA + 1 / covB) * (parsA - parsB);

  throw_assert(kl >= 0.0, "kl-distance should be positive");
  return kl;
}

template <typename source_link_t>
auto mergeComponents(const GsfComponentCache<source_link_t> &a,
                     const GsfComponentCache<source_link_t> &b) {
  throw_assert(a.weight > 0.0 && b.weight > 0.0, "weight error");

  std::array range = {std::ref(a), std::ref(b)};
  auto [mergedPars, mergedCov] =
      combineComponentRange(range.begin(), range.end(), [](auto &a) {
        return std::tie(a.get().weight, a.get().predictedPars,
                        a.get().predictedCov);
      });

  GsfComponentCache<source_link_t> ret = a;
  ret.predictedPars = mergedPars;
  ret.predictedCov = mergedCov;
  ret.weight = a.weight + b.weight;

  return ret;
}

/// @brief Class representing a symmetric distance matrix
class SymmetricKLDistanceMatrix {
  Eigen::VectorXd m_data;
  std::vector<std::pair<std::size_t, std::size_t>> m_mapToPair;
  std::size_t m_N;

 public:
  template <typename source_link_t>
  SymmetricKLDistanceMatrix(
      const std::vector<GsfComponentCache<source_link_t>> &cmps)
      : m_data(cmps.size() * (cmps.size() - 1) / 2),
        m_mapToPair(m_data.size()),
        m_N(cmps.size()) {
    for (auto i = 1ul; i < m_N; ++i) {
      const auto indexConst = (i - 1) * i / 2;
      for (auto j = 0ul; j < i; ++j) {
        m_mapToPair.at(indexConst + j) = {i, j};
        m_data[indexConst + j] = computeKLDistance(cmps[i], cmps[j]);
      }
    }
  }

  auto at(std::size_t i, std::size_t j) const {
    return m_data[i * (i - 1) / 2 + j];
  }

  template <typename source_link_t>
  void recomputeAssociatedDistances(
      std::size_t n,
      const std::vector<GsfComponentCache<source_link_t>> &cmps) {
    const auto indexConst = (n - 1) * n / 2;

    throw_assert(cmps.size() == m_N, "size mismatch");

    // Rows
    for (auto i = 0ul; i < n; ++i) {
      m_data[indexConst + i] = computeKLDistance(cmps[n], cmps[i]);
    }

    // Columns
    for (auto i = n + 1; i < cmps.size(); ++i) {
      m_data[(i - 1) * i / 2 + n] = computeKLDistance(cmps[n], cmps[i]);
    }
  }

  void resetAssociatedDistances(std::size_t n, double value) {
    const auto indexConst = (n - 1) * n / 2;

    // Rows
    for (auto i = 0ul; i < n; ++i) {
      m_data[indexConst + i] = value;
    }

    // Columns
    for (auto i = n + 1; i < m_N; ++i) {
      m_data[(i - 1) * i / 2 + n] = value;
    }
  }

  auto minDistancePair() const {
    // TODO eigen minCoeff does not work for some reason??
    // return m_mapToPair.at(m_data.minCoeff());
    return m_mapToPair.at(std::distance(
        &m_data[0], std::min_element(&m_data[0], &m_data[m_data.size() - 1])));
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const SymmetricKLDistanceMatrix &m) {
    for (auto i = 1ul; i < m.m_N; ++i) {
      const auto indexConst = (i - 1) * i / 2;
      Eigen::RowVectorXd vals;
      vals.resize(i);
      for (auto j = 0ul; j < i; ++j) {
        vals[j] = m.m_data[indexConst + j];
      }
      os << vals << "\n";
    }

    return os;
  }
};

template <typename source_link_t>
void reduceWithKLDistance(
    std::vector<GsfComponentCache<source_link_t>> &cmpCache,
    std::size_t maxCmpsAfterMerge) {
  if (cmpCache.size() <= maxCmpsAfterMerge) {
    return;
  }

  SymmetricKLDistanceMatrix distances(cmpCache);

  auto remainingComponents = cmpCache.size();

  while (remainingComponents > maxCmpsAfterMerge) {
    const auto [minI, minJ] = distances.minDistancePair();

    cmpCache[minI] = mergeComponents(cmpCache[minI], cmpCache[minJ]);
    distances.recomputeAssociatedDistances(minI, cmpCache);
    remainingComponents--;

    // Reset removed components so that it won't have the shortest distance
    // ever, and so that we can sort them by weight in the end to remove them
    cmpCache[minJ].weight = -1.0;
    cmpCache[minJ].predictedPars[eBoundQOverP] =
        std::numeric_limits<double>::max();
    (*cmpCache[minJ].predictedCov)(eBoundQOverP, eBoundQOverP) =
        std::numeric_limits<double>::max();
    distances.resetAssociatedDistances(minJ,
                                       std::numeric_limits<double>::max());

    // Print current mean
    Acts::BoundVector mean = Acts::BoundVector::Zero();
    for(const auto &c : cmpCache)
    {
        if( c.weight != -1.0 )
            mean += c.weight * c.predictedPars;
    }
    std::cout << "current with " << remainingComponents << " components: " << mean.transpose() << "\n";
  }

  // Remove all components which are labled with weight -1
  std::sort(cmpCache.begin(), cmpCache.end(),
            [](const auto &a, const auto &b) { return a.weight < b.weight; });
  cmpCache.erase(std::remove_if(cmpCache.begin(), cmpCache.end(),
                                [](const auto &a) { return a.weight == -1.0; }),
                 cmpCache.end());

  throw_assert(cmpCache.size() == maxCmpsAfterMerge,
               "size mismatch, should be " << maxCmpsAfterMerge << ", but is "
                                           << cmpCache.size());
}

}  // namespace detail

}  // namespace Acts
