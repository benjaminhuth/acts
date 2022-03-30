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

#include "TestHelpers.hpp"

inline auto build_tracking_geometry() {
  using namespace Acts::UnitLiterals;

  // Make some planar Surfaces:
  const Acts::Vector3 start_pos{0., 0., 0.};
  const Acts::Vector3 normal{1., 0., 0.};
  const Acts::RotationMatrix3 surface_rotation = Acts::RotationMatrix3(
      Eigen::AngleAxisd(0.5 * M_PI, Acts::Vector3::UnitY()));
  const auto surface_distance = 100._mm;
  const auto surface_width = 300._mm;
  const auto surface_thickness = 1._mm;
  const auto n_surfaces = 5;
  const auto surface_material = std::shared_ptr<const Acts::ISurfaceMaterial>();
  const auto surface_bounds = std::make_shared<Acts::RectangleBounds>(
      surface_width / 2., surface_width / 2.);

  Acts::CuboidVolumeBuilder::VolumeConfig volume_config{};

  // Start Surface
  {
    Acts::CuboidVolumeBuilder::SurfaceConfig srf_cfg{
        start_pos,
        surface_rotation,
        std::make_shared<Acts::RectangleBounds>(10._mm, 10._mm),
        surface_material,
        surface_thickness,
        {/* no detector element factory */}};

    Acts::CuboidVolumeBuilder::LayerConfig layer_cfg{srf_cfg, nullptr, true};

    volume_config.layerCfg.push_back(layer_cfg);
  }

  // Surfaces
  for (int i = 0; i < n_surfaces; ++i) {
    Acts::CuboidVolumeBuilder::SurfaceConfig srf_cfg{
        start_pos + (i + 1) * surface_distance * normal,
        surface_rotation,
        surface_bounds,
        surface_material,
        surface_thickness,
        {/* no detector element factory */}};

    Acts::CuboidVolumeBuilder::LayerConfig layer_cfg;
    layer_cfg.active = true;
    layer_cfg.surfaceCfg = srf_cfg;

    volume_config.layerCfg.push_back(layer_cfg);
  }

  volume_config.position = {250._mm, 0., 0.};
  volume_config.length = {500._mm, 300._mm, 300._mm};

  Acts::CuboidVolumeBuilder::Config builder_config;
  builder_config.volumeCfg = {volume_config};
  builder_config.position = {250._mm, 0., 0.};
  builder_config.length = {500._mm, 300._mm, 300._mm};

  Acts::CuboidVolumeBuilder builder(builder_config);

  Acts::TrackingGeometryBuilder::Config tgeo_builder_config;
  tgeo_builder_config.trackingVolumeBuilders.push_back(
      [=](const auto& context, const auto& inner, const auto&) {
        return builder.trackingVolume(context, inner, nullptr);
      });

  Acts::TrackingGeometryBuilder tgeo_builder(tgeo_builder_config);

  return std::make_pair(
      builder.buildSurface(geoCtx, volume_config.layerCfg.front().surfaceCfg),
      std::shared_ptr{tgeo_builder.trackingGeometry(geoCtx)});
}
