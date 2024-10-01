// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/ExaTrkX/Stages.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <memory>

#include <torch/torch.h>

namespace torch::inductor {
class AOTIModelContainerRunnerCuda;
}

namespace c10 {
enum class DeviceType : std::int8_t;
}

namespace Acts {

class TorchEdgeClassifierAOT final : public Acts::EdgeClassificationBase {
 public:
  struct Config {
    std::string modelPath;
    std::vector<int> selectedFeatures = {};
    float cut = 0.21;
    int nChunks = 1;  // NOTE for GNN use 1
    bool undirected = false;
    int deviceID = 0;
    bool useEdgeFeatures = false;
  };

  TorchEdgeClassifierAOT(const Config &cfg,
                         std::unique_ptr<const Logger> logger);
  ~TorchEdgeClassifierAOT();

  std::tuple<std::any, std::any, std::any, std::any> operator()(
      std::any nodeFeatures, std::any edgeIndex, std::any edgeFeatures = {},
      torch::Device device = torch::Device(torch::kCPU)) override;

  Config config() const { return m_cfg; }
  torch::Device device() const override { return m_device; };

 private:
  std::unique_ptr<const Acts::Logger> m_logger;
  const auto &logger() const { return *m_logger; }

  Config m_cfg;
  c10::DeviceType m_deviceType;
  torch::Device m_device;
  std::unique_ptr<torch::inductor::AOTIModelContainerRunnerCuda> m_model;
};

}  // namespace Acts