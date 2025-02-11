// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/ExaTrkX/Stages.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <memory>

#include <torch/script.h>

namespace torch::jit {
class Module;
}

namespace c10 {
enum class DeviceType : std::int8_t;
}

namespace Acts {

class DummyEdgeClassifier final : public Acts::EdgeClassificationBase {
 public:
  struct Config {
    bool keepAll = false;
  };

  DummyEdgeClassifier(const Config &cfg, std::unique_ptr<const Logger> logger);
  ~DummyEdgeClassifier();

  std::tuple<std::any, std::any, std::any, std::any> operator()(
      std::any nodeFeatures, std::any edgeIndex, std::any edgeFeatures = {},
      const ExecutionContext &execContext = {}) override;

  Config config() const { return m_cfg; }
  torch::Device device() const override { return torch::kCPU; };

 private:
  std::unique_ptr<const Acts::Logger> m_logger;
  const auto &logger() const { return *m_logger; }

  Config m_cfg;
};

}  // namespace Acts
