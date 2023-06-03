// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/ExaTrkX/TorchMetricLearning.hpp"

#include "Acts/Plugins/ExaTrkX/buildEdges.hpp"

#include <torch/script.h>
#include <torch/torch.h>

using namespace torch::indexing;

namespace Acts {

TorchMetricLearning::TorchMetricLearning(const Config &cfg) : m_cfg(cfg) {
  c10::InferenceMode guard(true);
  m_deviceType = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

  try {
    m_model = std::make_unique<torch::jit::Module>();
    *m_model = torch::jit::load(m_cfg.modelPath, m_deviceType);
    m_model->eval();
  } catch (const c10::Error &e) {
    throw std::invalid_argument("Failed to load models: " + e.msg());
  }
}

TorchMetricLearning::~TorchMetricLearning() {}

std::tuple<std::any, std::any> TorchMetricLearning::operator()(
    boost::multi_array<float, 2> &nodeFeatures, const Logger &logger) {
  c10::InferenceMode guard(true);
  const torch::Device device(m_deviceType);

  // Clone models (solve memory leak? members can be const...)
  auto e_model = m_model->clone();
  e_model.to(device);

  // printout the r,phi,z of the first spacepoint
  ACTS_VERBOSE("First spacepoint information [r, phi, z]: "
               << nodeFeatures[0][0] << ", " << nodeFeatures[0][1] << ", "
               << nodeFeatures[0][2]);
  ACTS_VERBOSE("Max and min spacepoint: "
               << *std::max_element(nodeFeatures.data(), nodeFeatures.data()+nodeFeatures.num_elements())
               << ", "
               << *std::min_element(nodeFeatures.data(), nodeFeatures.data()+nodeFeatures.num_elements()));
  // print_current_cuda_meminfo(logger);

  // **********
  // Embedding
  // **********

  const auto shape = nodeFeatures.shape();
  
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto nodeFeatureTensor =
      torch::from_blob(nodeFeatures.data(), {static_cast<long>(shape[0]), static_cast<long>(shape[1])}, opts);

  std::vector<torch::jit::IValue> inputTensors;
  inputTensors.push_back(nodeFeatureTensor.index({Slice(), Slice(None, m_cfg.spacepointFeatures)}));
  
  auto output = e_model.forward(inputTensors).toTensor();
  inputTensors.clear();

  ACTS_VERBOSE("Embedding space of the first SP:\n"
               << output.slice(/*dim=*/0, /*start=*/0, /*end=*/1));

  // ****************
  // Building Edges
  // ****************

  torch::Tensor edgeList = buildEdges(
      output, shape[0], m_cfg.embeddingDim, m_cfg.rVal, m_cfg.knnVal);

  ACTS_VERBOSE("Shape of built edges: (" << edgeList.size(0) << ", "
                                         << edgeList.size(1));
  ACTS_VERBOSE("Slice of edgelist:\n" << edgeList.slice(1, 0, 5));
  // print_current_cuda_meminfo(logger);

  return {nodeFeatureTensor, edgeList};
}
}  // namespace Acts
