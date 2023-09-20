// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/ExaTrkX/ExaTrkXPipeline.hpp"
#include "Acts/Plugins/ExaTrkX/detail/CantorEdge.hpp"
#include "Acts/Utilities/Logger.hpp"

namespace Acts {

class TorchGraphStoreHook : public ExaTrkXHook {
  std::unique_ptr<std::vector<int64_t>> m_storedGraph;

 public:
  TorchGraphStoreHook();
  ~TorchGraphStoreHook() override {}

  void operator()(const std::any &, const std::any &edges) const override;

  const std::vector<int64_t> &storedGraph() const { return *m_storedGraph; }
};

}  // namespace Acts
