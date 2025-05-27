// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/ExaTrkX/TracccFeatureCreation.hpp"

namespace {}

namespace Acts {

Acts::Tensor<float> createInputTensor(
    const std::vector<std::string> &features,
    const std::vector<float> &featureScales,
    const traccc::edm::spacepoint_collection::const_device &sps,
    const std::unordered_map<std::string, std::vector<float>>
        &additionalFeatures,
    const ExecutionContext &execContext) {
  assert(execContext.device.isCuda());
  assert(execContext.stream.has_value());

  auto tensor =
      Acts::Tensor<float>::Create({sps.size(), features.size()}, execContext);

  return tensor;
}

}  // namespace Acts
