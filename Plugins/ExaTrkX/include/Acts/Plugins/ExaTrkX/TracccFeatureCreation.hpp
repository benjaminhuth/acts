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
#include <vector>

#include "traccc/edm/spacepoint_collection.hpp"

namespace Acts {

Acts::Tensor<float> createInputTensor(
    const std::vector<std::string_view> &features,
    const std::vector<float> &featureScales,
    traccc::edm::spacepoint_collection::const_view sps,
    const ExecutionContext &execContext,
    std::optional<vecmem::data::vector_view<const float>> clXglobal,
    std::optional<vecmem::data::vector_view<const float>> clYglobal,
    std::optional<vecmem::data::vector_view<const float>> clZglobal);

}
