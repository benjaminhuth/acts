// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Utilities/ThrowAssert.hpp"

#include <limits>
#include <vector>

namespace Acts {

inline auto get_beampline_id(double pos_z, std::vector<double> bpsplit) {
  // set new highest and lowest boundary
  bpsplit.front() = std::numeric_limits<double>::lowest();
  bpsplit.back() = std::numeric_limits<double>::max();

  auto it = bpsplit.cbegin();
  for (; it != std::prev(bpsplit.cend()); ++it) {
    if (pos_z >= *it && pos_z < *std::next(it))
      break;
  }

  const auto id = static_cast<std::size_t>(std::distance(bpsplit.cbegin(), it));

  throw_assert(id < (bpsplit.size() - 1ul), "");
  return id;
}

}  // namespace Acts
