// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsFatras/Digitization/DigitizationError.hpp"

namespace {

/// Custom error category for digitization errors.
class DigitizationErrorCategory : public std::error_category {
 public:
  /// Return a short descriptive name for the category.
  const char* name() const noexcept final { return "DigitizationError"; }

  /// Return what each enum means in text.
  std::string message(int c) const final {
    using ActsFatras::DigitizationError;

    switch (static_cast<DigitizationError>(c)) {
      case DigitizationError::SmearingOutOfRange:
        return "smeared out of surface bounds";
      case DigitizationError::SmearingError:
        return "smearing error occured";
      case DigitizationError::UndefinedSurface:
        return "surface undefined for this operation";
      case DigitizationError::MaskingError:
        return "surface mask could not be applied";
      case DigitizationError::IntersectionFailed:
        return "intersection of hit with surface failed";
      case DigitizationError::IntersectionPathToLarge:
        return "intersection pathlength of hit to large";
      default:
        return "unknown";
    }
  }
};

}  // namespace

std::error_code ActsFatras::make_error_code(ActsFatras::DigitizationError e) {
  static DigitizationErrorCategory c;
  return {static_cast<int>(e), c};
}
