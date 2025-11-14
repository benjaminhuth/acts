// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Utilities/Logger.hpp"
#include "ActsPlugins/Gnn/Stages.hpp"

#include <memory>

namespace ActsPlugins {

class TrackLengthEdgeFilter final : public EdgeClassificationBase {
 public:
  struct Config {
    float stripRadius = 0.f;
    std::size_t radiusFeatureIdx = 0;
    std::size_t minTrackLength = 7;
  };

  TrackLengthEdgeFilter(const Config &cfg,
                        std::unique_ptr<const Acts::Logger> logger);
  ~TrackLengthEdgeFilter();

  PipelineTensors operator()(PipelineTensors tensors,
                             const ExecutionContext &execContext = {}) override;

  Config config() const { return m_cfg; }

 private:
  std::unique_ptr<const Acts::Logger> m_logger;
  const auto &logger() const { return *m_logger; }

  Config m_cfg;
};

}  // namespace ActsPlugins
