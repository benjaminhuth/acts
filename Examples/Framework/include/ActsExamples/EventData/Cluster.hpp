// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsFatras/Digitization/Segmentizer.hpp"

#include <numeric>
#include <vector>
#include <optional>

namespace ActsExamples {

/// Simple struct holding cluster information.
struct Cluster {
  using Cell = ActsFatras::Segmentizer::ChannelSegment;
  std::size_t sizeLoc0 = 0;
  std::size_t sizeLoc1 = 0;
  std::vector<Cell> channels;

  // TODO make this be provided by Fatras?
  Acts::Vector3 globalPosition = Acts::Vector3::Zero();

  double sumActivations() const {
    return std::accumulate(
        channels.begin(), channels.end(), 0.0,
        [](double s, const Cluster::Cell& c) { return s + c.activation; });
  }
};

/// Clusters have a one-to-one relation with measurements
using ClusterContainer = std::vector<Cluster>;

}  // namespace ActsExamples
