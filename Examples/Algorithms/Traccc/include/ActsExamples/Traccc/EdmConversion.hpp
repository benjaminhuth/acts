// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <ActsExamples/EventData/Measurement.hpp>
#include <ActsExamples/EventData/SimSpacePoint.hpp>

#include <traccc/edm/measurement.hpp>
#include <traccc/edm/spacepoint_collection.hpp>

namespace ActsExamples {

void convertToTraccc(traccc::edm::spacepoint_collection::host &outputSps,
                     const ActsExamples::SimSpacePointContainer &inputSps);

void convertToTraccc(
    traccc::measurement_collection_types::host &outputMeas,
    const ActsExamples::MeasurementContainer &inputMeas,
    const std::unordered_map<Acts::GeometryIdentifier,
                             detray::geometry::barcode> &surfaceMap);

}  // namespace ActsExamples
