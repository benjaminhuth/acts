// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/ExaTrkX/Tensor.hpp"

#include <string_view>
#include <unordered_map>
#include <vector>

#include "traccc/edm/spacepoint_collection.hpp"

namespace Acts {

Acts::Tensor<float> createInputTensor(
    const std::vector<std::string_view> &features,
    const std::vector<float> &featureScales,
    const traccc::edm::spacepoint_collection::const_device &sps,
    const ExecutionContext &execContext,
    const std::optional<vecmem::device_vector<float>> &clXglobal = {},
    const std::optional<vecmem::device_vector<float>> &clYglobal = {},
    const std::optional<vecmem::device_vector<float>> &clZglobal = {});

}
