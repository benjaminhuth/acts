// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <system_error>

namespace Acts {

enum class GsfError {
  // ensure all values are non-zero
  NavigationFailed = 1,
  ComponentNumberMismatch = 2,
  NoComponentCreated = 3,
  NoStatesCreated = 4,
  StartParametersNotOnStartSurface = 5
};

std::error_code make_error_code(Acts::GsfError e);

}  // namespace Acts

// register with STL
namespace std {
template <>
struct is_error_code_enum<Acts::GsfError> : std::true_type {};
}  // namespace std
