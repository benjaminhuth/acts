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
};

std::error_code make_error_code(Acts::GsfError e);

}  // namespace Acts

// register with STL
namespace std {
template <>
struct is_error_code_enum<Acts::GsfError> : std::true_type {};
}  // namespace std

// On the long run, this can be in a C++ file
namespace {

class GsfErrorCategory : public std::error_category {
 public:
  // Return a short descriptive name for the category.
  const char* name() const noexcept final { return "MultiStepperError"; }

  // Return what each enum means in text.
  std::string message(int c) const final {
    using Acts::GsfError;

    switch (static_cast<GsfError>(c)) {
      case GsfError::NavigationFailed:
        return "Navigation failed, forward and backward pass incompatible";
      case GsfError::ComponentNumberMismatch:
        return "Component Number changed during two GSF calls";
      case GsfError::NoComponentCreated:
        return "No component has been created in the filter step";
      case GsfError::NoStatesCreated:
        return "No states where created in the MultiTrajectory";
      default:
        return "unknown";
    }
  }
};

}  // namespace

std::error_code Acts::make_error_code(Acts::GsfError e) {
  static GsfErrorCategory c;
  return {static_cast<int>(e), c};
}
