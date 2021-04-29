// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/CuboidVolumeBuilder.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingGeometryBuilder.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/MultiEigenStepper.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/detail/SteppingLogger.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Visualization/GeometryView3D.hpp"
#include "Acts/Visualization/ObjVisualization3D.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Acts::UnitLiterals;

using MagneticField = Acts::ConstantBField;

using SingleStepper = Acts::EigenStepper<>;
using MultiStepper = Acts::MultiEigenStepper<>;

using SinglePropagator = Acts::Propagator<SingleStepper, Acts::Navigator>;
using MultiPropagator = Acts::Propagator<MultiStepper, Acts::Navigator>;

const Acts::GeometryContext geoCtx;
const Acts::MagneticFieldContext magCtx;

auto build_tracking_geometry() {
  // Make some planar Surfaces:
  const Acts::Vector3 start_pos{0., 0., 0.};
  const Acts::Vector3 normal{0., 0., 1.};
  const Acts::RotationMatrix3 surface_rotation =
      Acts::RotationMatrix3::Identity();
  const auto surface_distance = 100._mm;
  const auto surface_width = 300._mm;
  const auto surface_thickness = 0._mm;
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

  volume_config.position = {0., 0., 250._mm};
  volume_config.length = {300._mm, 300._mm, 500._mm};

  Acts::CuboidVolumeBuilder::Config builder_config;
  builder_config.volumeCfg = {volume_config};
  builder_config.position = {0., 0., 250._mm};
  builder_config.length = {300._mm, 300._mm, 500._mm};

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

void export_detector_to_obj(const Acts::TrackingGeometry& detector) {
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
      draw_config.passiveView, draw_config.sensitiveView, draw_config.gridView);
}

void export_tracks_to_obj(
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

inline std::pair<SinglePropagator, MultiPropagator> make_propagators(
    double bx, std::shared_ptr<const Acts::TrackingGeometry> tgeo) {
  auto magField = std::make_shared<MagneticField>(Acts::Vector3(bx, 0.0, 0.0));
  Acts::Navigator navigator(tgeo);

  return std::pair{SinglePropagator(SingleStepper(magField), navigator),
                   MultiPropagator(MultiStepper(magField), navigator)};
}

int main(int argc, char** argv) {
  // Cmd args
  const std::vector<std::string> args(argv, argv + argc);
  const std::string bfield_flag = "--bfield-value";
  const std::string export_flag = "--export-obj";

  if (std::find(begin(args), end(args), "--help") != args.end()) {
    std::cout << "Usage: " << argv[0] << " <options>\n";
    std::cout << "\t" << bfield_flag << " <val>   bfield value in Tesla\n";
    std::cout
        << "\t" << export_flag
        << "           boolean flag wether to export geometry as *.obj files\n";
    return 0;
  }

  // Logger
  const auto single_logger =
      Acts::getDefaultLogger("GSF Test - Single", Acts::Logging::VERBOSE);
  const auto multi_logger =
      Acts::getDefaultLogger("GSF Test - Single", Acts::Logging::VERBOSE);

  // Bfield
  double bfield_value = 2._T;
  if (auto found = std::find(begin(args), end(args), bfield_flag);
      found != args.end() && std::next(found) != args.end()) {
    bfield_value = Acts::UnitConstants::T * std::stod(*std::next(found));
  }

  // Make detector geometry
  auto [start_surface, detector] = build_tracking_geometry();

  // Make Propagators
  auto [single_prop, multi_prop] = make_propagators(bfield_value, detector);

  // Setup propagation
  const Acts::BoundSymMatrix cov = Acts::BoundSymMatrix::Identity();
  const double l0{0.}, l1{0.}, theta{0.}, phi{0.}, p{50._GeV}, q{-1.}, t{0.};
  Acts::BoundVector pars;
  pars << l0, l1, phi, theta, q / p, t;

  Acts::BoundTrackParameters start_pars(start_surface, std::move(pars),
                                        std::move(cov));

  // Action list and abort list
  using ActionList =
      Acts::ActionList<Acts::detail::SteppingLogger, Acts::MaterialInteractor>;
  using AbortList = Acts::AbortList<Acts::EndOfWorldReached>;
  using PropagatorOptions =
      Acts::DenseStepperPropagatorOptions<ActionList, AbortList>;

  PropagatorOptions single_options(geoCtx, magCtx,
                                   Acts::LoggerWrapper(*single_logger));

  PropagatorOptions multi_options(geoCtx, magCtx,
                                  Acts::LoggerWrapper(*multi_logger));

  // Propagation
  auto single_result = single_prop.propagate(start_pars, single_options);

  const auto single_stepper_result =
      single_result.value().get<Acts::detail::SteppingLogger::result_type>();

  auto multi_result = multi_prop.propagate(start_pars, multi_options);

  const auto multi_stepper_result =
      multi_result.value().get<Acts::detail::SteppingLogger::result_type>();

  // Export
  if (std::find(begin(args), end(args), export_flag) != args.end()) {
    export_detector_to_obj(*detector);
    export_tracks_to_obj(
        std::vector<decltype(single_stepper_result.steps)>{
            single_stepper_result.steps},
        "propagation_single");
    export_tracks_to_obj(
        std::vector<decltype(multi_stepper_result.steps)>{
            multi_stepper_result.steps},
        "propagation_single");
  }
}
