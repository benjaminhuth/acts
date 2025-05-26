// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/Traccc/EdmConversion.hpp"

namespace ActsExamples {

void convertToTraccc(traccc::edm::spacepoint_collection::host &outputSps,
                     const ActsExamples::SimSpacePointContainer &inputSps) {
  for (std::size_t i = 0; i < inputSps.size(); i++) {
    const auto sls = inputSps.at(i).sourceLinks();
    auto idx1 = sls.at(0).get<ActsExamples::IndexSourceLink>().index();
    auto idx2 = sls.size() == 2
                    ? sls.at(1).get<ActsExamples::IndexSourceLink>().index()
                    : traccc::edm::spacepoint_collection::host::
                          INVALID_MEASUREMENT_INDEX;

    outputSps.push_back(
        {idx1,
         idx2,
         {inputSps.at(i).x(), inputSps.at(i).y(), inputSps.at(i).z()},
         inputSps.at(i).varianceZ(),
         inputSps.at(i).varianceR()});
  }
}

void convertToTraccc(
    traccc::measurement_collection_types::host &outputMeas,
    const ActsExamples::MeasurementContainer &inputMeas,
    const std::unordered_map<Acts::GeometryIdentifier,
                             detray::geometry::barcode> &surfaceMap) {
  for (std::size_t i = 0; i < inputMeas.size(); i++) {
    traccc::measurement meas;

    meas.measurement_id = i;
    meas.local = {inputMeas.at(i).parameters()[Acts::eBoundLoc0],
                  inputMeas.at(i).parameters()[Acts::eBoundLoc1]};

    meas.variance = {
        inputMeas.at(i).covariance()(Acts::eBoundLoc0, Acts::eBoundLoc0),
        inputMeas.at(i).covariance()(Acts::eBoundLoc1, Acts::eBoundLoc1)};

    meas.surface_link = surfaceMap.at(inputMeas.at(i).geometryId());
    meas.meas_dim = inputMeas.at(i).size();
    meas.subs = std::decay_t<decltype(meas.subs)>{
        {0u, 1u}};  // Assuming 2D measurements

    outputMeas.push_back(meas);
  }
}

}  // namespace ActsExamples
