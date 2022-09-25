// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/ExaTrkX/ExaTrkXTrackFindingCpu.hpp"

#include <boost/filesystem.hpp>

#include "torchInference.hpp"
#include "buildEdgesTBB.hpp"

namespace Acts {

ExaTrkXTrackFindingCpu::ExaTrkXTrackFindingCpu(
    const ExaTrkXTrackFindingCpu::Config& config)
    : ExaTrkXTrackFindingBase("ExaTrkXTorch"), m_cfg(config) {
  using Path = boost::filesystem::path;

  const Path embedModelPath = Path(m_cfg.modelDir) / "embed.pt";
  const Path filterModelPath = Path(m_cfg.modelDir) / "filter.pt";
  const Path gnnModelPath = Path(m_cfg.modelDir) / "gnn.pt";
  c10::InferenceMode guard(true);

  try {
    m_embeddingModel = std::make_unique<torch::jit::Module>();
    *m_embeddingModel = torch::jit::load(embedModelPath.c_str());
    m_embeddingModel->eval();

    m_filterModel = std::make_unique<torch::jit::Module>();
    *m_filterModel = torch::jit::load(filterModelPath.c_str());
    m_filterModel->eval();

    m_gnnModel = std::make_unique<torch::jit::Module>();
    *m_gnnModel = torch::jit::load(gnnModelPath.c_str());
    m_gnnModel->eval();
  } catch (const c10::Error& e) {
    throw std::invalid_argument("Failed to load models: " + e.msg());
  }
}

ExaTrkXTrackFindingCpu::~ExaTrkXTrackFindingCpu() {}

std::optional<ExaTrkXTime> ExaTrkXTrackFindingCpu::getTracks(
    std::vector<float>& inputValues, std::vector<int>& spacepointIDs,
    std::vector<std::vector<int> >& trackCandidates, LoggerWrapper logger,
    bool recordTiming) const {


  auto e_model = m_embeddingModel->clone();
  auto f_model = m_filterModel->clone();
  auto g_model = m_gnnModel->clone();


  return detail::torchInference(m_cfg, inputValues, spacepointIDs, trackCandidates, logger, recordTiming, at::kCPU, buildEdgesTBB, [](auto){}, e_model, f_model, g_model);
}

}  // namespace Acts

