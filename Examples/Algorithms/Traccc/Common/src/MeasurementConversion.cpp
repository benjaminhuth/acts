// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Traccc/Common/Conversion/MeasurementConversion.hpp"

namespace ActsExamples::Traccc::Common::Conversion {
typename traccc::measurement_collection_types::host createTracccMeasurements(
    const MeasurementContainer &measurements,
    const std::map<std::uint64_t, detray::geometry::barcode> &actsBarcodeMap) {
  std::map<detray::geometry::barcode, unsigned int> barcodeMap;

  traccc::measurement_collection_types::host result_measurements;
  traccc::cell_module_collection_types::host result_modules;

  // Read the measurements from the input file.
  for (const auto &varMeasurement : measurements) {
    std::visit(
        [&](const auto &actsMeasurement) {
          const auto idxSl = actsMeasurement.sourceLink().template get<IndexSourceLink>();
          const auto geoId = idxSl.geometryId();

          // Establish the "correct" geometry ID.
          detray::geometry::barcode barcode;
          auto it = actsBarcodeMap.find(geoId.value());
          if (it != actsBarcodeMap.end()) {
            barcode = (*it).second;
          } else {
            throw std::runtime_error("Barcode not found for geometry ID");
          }

          unsigned int link;
          auto it2 = barcodeMap.find(barcode);

          if (it2 != barcodeMap.end()) {
              link = (*it2).second;
          } else {
              link = result_modules.size();
              barcodeMap[barcode] = link;
              traccc::cell_module mod;
              mod.surface_link = barcode;
              result_modules.push_back(mod);
          }

        // Construct the measurement object.
        traccc::measurement meas;
        std::array<typename traccc::transform3::size_type, 2u> indices{0u, 0u};
        meas.meas_dim = 0u;

        // Local key is a 8 bit char and first and last bit are dummy value. 2 -
        // 7th bits are for 6 bound track parameters.
        // Ex1) 0000010 or 2 -> meas dim = 1 and [loc0] active -> strip or wire
        // Ex2) 0000110 or 6 -> meas dim = 2 and [loc0, loc1] active -> pixel
        // Ex3) 0000100 or 4 -> meas dim = 1 and [loc1] active -> annulus
        for (unsigned int ipar = 0; ipar < 2u; ++ipar) {
            if (actsMeasurement.contains(static_cast<Acts::BoundIndices>(ipar))) {
                switch (ipar) {
                    case traccc::e_bound_loc0: {
                        meas.local[0] = actsMeasurement.parameters()[0];
                        meas.variance[0] = actsMeasurement.parameters()(0,0);
                        indices[meas.meas_dim++] = ipar;
                    }; break;
                    case traccc::e_bound_loc1: {
                        meas.local[1] = actsMeasurement.parameters()[1];
                        meas.variance[1] = actsMeasurement.parameters()[1];
                        indices[meas.meas_dim++] = ipar;
                    }; break;
                }
            }
        }

        meas.subs.set_indices(indices);
        meas.surface_link = barcode;
        meas.module_link = link;

        // Keeps measurement_id for ambiguity resolution
        meas.measurement_id = idxSl.index();

        result_measurements.push_back(meas);
        }, varMeasurement);
  }

  return result_measurements;
}

}  // namespace ActsExamples::Traccc::Common::Conversion
