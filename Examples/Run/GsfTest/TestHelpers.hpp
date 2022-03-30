// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/CuboidVolumeBuilder.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingGeometryBuilder.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Visualization/GeometryView3D.hpp"
#include "Acts/Visualization/ObjVisualization3D.hpp"

// #include "GsfInfrastructure/MultiSteppingLogger.hpp"

// Global constant Context objects
const Acts::GeometryContext geoCtx;
const Acts::MagneticFieldContext magCtx;

inline void export_detector_to_obj(const Acts::TrackingGeometry& detector, const std::string &output_dir) {
  const double output_scalor = 1.0;
  const size_t output_recision = 6;

  Acts::ObjVisualization3D objVis(output_recision, output_scalor);

  struct {
    Acts::ViewConfig containerView = Acts::ViewConfig({220, 220, 220});
    Acts::ViewConfig volumeView = Acts::ViewConfig({220, 220, 0});
    Acts::ViewConfig sensitiveView = Acts::ViewConfig({0, 180, 240});
    Acts::ViewConfig passiveView = Acts::ViewConfig({240, 280, 0});
    Acts::ViewConfig gridView = Acts::ViewConfig({220, 0, 0});
  } draw_config;

  Acts::GeometryView3D::drawTrackingVolume(
      objVis, *detector.highestTrackingVolume(), geoCtx,
      draw_config.containerView, draw_config.volumeView,
      draw_config.passiveView, draw_config.sensitiveView, draw_config.gridView, true, "", output_dir);
}

inline void export_tracks_to_obj(
    const std::vector<std::vector<Acts::detail::Step>>& tracks,
    std::string filename_without_extension) {
  const double output_scalor = 1.0;

  std::ofstream os(filename_without_extension + ".obj",
                   std::ofstream::out | std::ofstream::trunc);
  if (!os) {
    throw std::ios_base::failure("Could not open '" +
                                 filename_without_extension + ".obj' to write");
  }

  unsigned int vertex_counter = 0;

  for (auto& steps : tracks) {
    if (steps.size() > 2) {
      ++vertex_counter;
      for (auto& step : steps) {
        os << "v " << output_scalor * step.position.x() << " "
           << output_scalor * step.position.y() << " "
           << output_scalor * step.position.z() << '\n';
      }

      size_t vBreak = vertex_counter + steps.size() - 1;
      for (; vertex_counter < vBreak; ++vertex_counter)
        os << "l " << vertex_counter << " " << vertex_counter + 1 << '\n';
    }
  }
}

using MagneticField = Acts::ConstantBField;

template <typename stepper_t>
auto make_propagator(std::shared_ptr<Acts::ConstantBField> magField,
                     std::shared_ptr<const Acts::TrackingGeometry> tgeo) {
  Acts::Navigator::Config cfg;
  cfg.trackingGeometry = tgeo;
  Acts::Navigator navigator(cfg);

  using Propagator = Acts::Propagator<stepper_t, Acts::Navigator>;

  return Propagator(stepper_t(magField), navigator);
  ;
}
