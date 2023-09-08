// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "ActsExamples/EventData/detail/IndexSourceLinkImpl.hpp"
#include "ActsExamples/EventData/GeometryContainers.hpp"

namespace ActsExamples {

/// Container of index source links.
///
/// Since the source links provide a `.geometryId()` accessor, they can be
/// stored in an ordered geometry container.
using IndexSourceLinkContainer = GeometryIdMultiset<IndexSourceLink>;
/// Accessor for the above source link container
///
/// It wraps up a few lookup methods to be used in the Combinatorial Kalman
/// Filter
struct IndexSourceLinkAccessor : GeometryIdMultisetAccessor<IndexSourceLink> {
  using BaseIterator = GeometryIdMultisetAccessor<IndexSourceLink>::Iterator;

  using Iterator = Acts::SourceLinkAdapterIterator<BaseIterator>;

  // get the range of elements with requested geoId
  std::pair<Iterator, Iterator> range(const Acts::Surface& surface) const {
    assert(container != nullptr);
    auto [begin, end] = container->equal_range(surface.geometryId());
    return {Iterator{begin}, Iterator{end}};
  }
};
}  // namespace ActsExamples
