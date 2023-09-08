// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/SourceLink.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "ActsExamples/EventData/Index.hpp"

#include <cassert>

namespace ActsExamples {

/// A source link that stores just an index.
///
/// This is intentionally kept as barebones as possible. The source link
/// is just a reference and will be copied, moved around, etc. often.
/// Keeping it small and separate from the actual, potentially large,
/// measurement data should result in better overall performance.
/// Using an index instead of e.g. a pointer, means source link and
/// measurement are decoupled and the measurement representation can be
/// easily changed without having to also change the source link.
class IndexSourceLink final {
 public:
  /// Construct from geometry identifier and index.
  constexpr IndexSourceLink(Acts::GeometryIdentifier gid, Index idx)
      : m_geometryId(gid), m_index(idx) {}

  // Construct an invalid source link. Must be default constructible to
  /// satisfy SourceLinkConcept.
  IndexSourceLink() = default;
  IndexSourceLink(const IndexSourceLink&) = default;
  IndexSourceLink(IndexSourceLink&&) = default;
  IndexSourceLink& operator=(const IndexSourceLink&) = default;
  IndexSourceLink& operator=(IndexSourceLink&&) = default;

  /// Access the index.
  constexpr Index index() const { return m_index; }

  Acts::GeometryIdentifier geometryId() const { return m_geometryId; }

  struct SurfaceAccessor {
    const Acts::TrackingGeometry& trackingGeometry;

    const Acts::Surface* operator()(const Acts::SourceLink& sourceLink) const {
      const auto& indexSourceLink = sourceLink.get<IndexSourceLink>();
      return trackingGeometry.findSurface(indexSourceLink.geometryId());
    }
  };

 private:
  Acts::GeometryIdentifier m_geometryId;
  Index m_index = 0;

  friend bool operator==(const IndexSourceLink& lhs,
                         const IndexSourceLink& rhs) {
    return (lhs.geometryId() == rhs.geometryId()) and
           (lhs.m_index == rhs.m_index);
  }
  friend bool operator!=(const IndexSourceLink& lhs,
                         const IndexSourceLink& rhs) {
    return not(lhs == rhs);
  }
};

}
