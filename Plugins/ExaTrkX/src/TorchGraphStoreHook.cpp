// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/ExaTrkX/TorchGraphStoreHook.hpp"

#include "Acts/Plugins/ExaTrkX/detail/TensorVectorConversion.hpp"

#include <torch/torch.h>

Acts::TorchGraphStoreHook::TorchGraphStoreHook() {
  m_storedGraph = std::make_unique<std::vector<int64_t>>();
}

void Acts::TorchGraphStoreHook::operator()(const std::any&,
                                                  const std::any& edges) const {
  *m_storedGraph = detail::tensor2DToVector<int64_t>(std::any_cast<torch::Tensor>(edges));
}

