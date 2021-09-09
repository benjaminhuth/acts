// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "GsfError.hpp"

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
