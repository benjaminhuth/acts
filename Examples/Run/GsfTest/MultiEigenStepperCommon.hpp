// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Propagator/EigenStepper.hpp"

namespace Acts {
namespace detail {

// Common representation of a MultiStepper component for the use e.g., in the
// GSF
struct CommonComponentRep {
  EigenStepper<>::BoundState boundState;
  ActsScalar weight;
  ActsScalar pathLengthSinceLast;
};

}  // namespace detail
}  // namespace Acts
