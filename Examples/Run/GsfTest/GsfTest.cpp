// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/Charge.hpp"
#include "Acts/EventData/MultiComponentBoundTrackParameters.hpp"
#include "Acts/Geometry/CuboidVolumeBuilder.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingGeometryBuilder.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Material/Material.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Visualization/GeometryView3D.hpp"
#include "Acts/Visualization/ObjVisualization3D.hpp"

#include <chrono>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "MultiEigenStepper.hpp"
#include "MultiEigenStepperSIMD.hpp"
#include "MultiSteppingLogger.hpp"
#include "NewGenericDefaultExtension.hpp"
#include "NewGenericDenseEnvironmentExtension.hpp"
#include "NewStepperExtensionList.hpp"
#include "SimdHelpers.hpp"

using namespace Acts::UnitLiterals;

using MagneticField = Acts::ConstantBField;

const Acts::GeometryContext geoCtx;
const Acts::MagneticFieldContext magCtx;

auto build_tracking_geometry() {
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

template <typename stepper_t>
auto make_propagator(double bz,
                     std::shared_ptr<const Acts::TrackingGeometry> tgeo) {
  auto magField = std::make_shared<MagneticField>(Acts::Vector3(0.0, 0.0, bz));
  Acts::Navigator navigator(tgeo);
  navigator.resolvePassive = true;

  using Propagator = Acts::Propagator<stepper_t, Acts::Navigator>;

  return Propagator(stepper_t(magField), navigator);
  ;
}

int main(int argc, char** argv) {
  // Cmd args
  const std::vector<std::string_view> args(argv, argv + argc);
  const std::string_view bfield_flag = "--bfield-value";
  const std::string_view stepper_flag = "--stepper";
  const std::string_view export_flag = "--export-obj";
  const std::string_view verbose_flag = "-v";

  const auto cmd_arg_exists = [&args](const std::string_view& arg) {
    return std::find(begin(args), end(args), arg) != args.end();
  };

  if (cmd_arg_exists("--help")) {
    std::cout << "Usage: " << argv[0] << " <options>\n";
    std::cout << "\t" << bfield_flag << " <val>\n";
    std::cout << "\t" << stepper_flag << " <single/loop/simd/all>\n";
    std::cout << "\t" << export_flag << "\n";
    std::cout << "\t" << verbose_flag << "\n";
    return 0;
  }

  const bool do_obj_export = cmd_arg_exists(export_flag);
  const auto log_level = cmd_arg_exists(verbose_flag) ? Acts::Logging::VERBOSE
                                                      : Acts::Logging::INFO;

  // Bfield
  const double bfield_value = [&]() {
    if (auto found = std::find(begin(args), end(args), bfield_flag);
        found != args.end())
      return Acts::UnitConstants::T * std::stod(std::string(*std::next(found)));
    else
      return 2._T;
  }();

  std::cout << "B-Field strenth: " << bfield_value / Acts::UnitConstants::T
            << "T\n";

  // Which stepper?
  const std::string_view stepper_type = [&]() {
    if (auto found = std::find(begin(args), end(args), stepper_flag);
        found != args.end())
      return *std::next(found);
    else
      return std::string_view("single");
  }();

  // Logger
  const auto single_logger = Acts::getDefaultLogger("Single", log_level);
  const auto multi_logger = Acts::getDefaultLogger("Multi", log_level);

  // Make detector geometry
  auto [start_surface, detector] = build_tracking_geometry();

  // Setup tracks
#ifdef NTRACKS
  constexpr int N = NTRACKS;
#else
  constexpr int N = 4;
#endif
  std::cout << "Using " << N << " parallel tracks\n";

  const auto track_data_vector = []() {
    const double l0{0.}, l1{0.}, theta{0.5 * M_PI}, phi{0.}, p{50._GeV}, q{-1.},
        t{0.};
    std::vector<std::tuple<double, Acts::BoundVector, Acts::BoundSymMatrix>>
        vec;

    for (int i = 0; i < N; ++i) {
      Acts::BoundVector pars;
      pars << l0, l1, phi, theta, q / (/*(i + 1) * */ p), t;
      vec.push_back({1. / N, pars, Acts::BoundSymMatrix::Identity()});
    }

    return vec;
  }();

  Acts::MultiComponentBoundTrackParameters<Acts::SinglyCharged> multi_pars(
      start_surface, track_data_vector);

  // Action list and abort list
  using SingleActionList =
      Acts::ActionList<Acts::detail::SteppingLogger, Acts::MaterialInteractor>;
  using MultiActionList =
      Acts::ActionList<MultiSteppingLogger, Acts::MaterialInteractor>;
  using AbortList = Acts::AbortList<Acts::EndOfWorldReached>;

  // Propagator options
  using SinglePropagatorOptions =
      Acts::DenseStepperPropagatorOptions<SingleActionList, AbortList>;
  using MultiPropagatorOptions =
      Acts::DenseStepperPropagatorOptions<MultiActionList, AbortList>;

  SinglePropagatorOptions single_options(geoCtx, magCtx,
                                         Acts::LoggerWrapper(*single_logger));

  MultiPropagatorOptions multi_options(geoCtx, magCtx,
                                       Acts::LoggerWrapper(*multi_logger));

  std::cout << "Stepper type: " << stepper_type << "\n";

  constexpr int Iter = 1000;

  //////////////////////////
  // SINGLE Stepper
  //////////////////////////
  if (stepper_type == "single" || stepper_type == "all") {
    const auto prop =
        make_propagator<Acts::EigenStepper<>>(bfield_value, detector);

    using SingleResult =
        decltype(prop.propagate(std::declval<Acts::BoundTrackParameters>(),
                                std::declval<SinglePropagatorOptions>()));

    // Prepare parameters
    std::vector<SingleResult> results;
    results.reserve(N);

    std::vector<Acts::BoundTrackParameters> pars;

    for (int i = 0; i < N; ++i) {
      pars.push_back(Acts::BoundTrackParameters(
          start_surface, std::get<Acts::BoundVector>(track_data_vector[i]),
          std::get<Acts::BoundSymMatrix>(track_data_vector[i])));
    }

    // Run propagation an measure time
    const auto t0 = std::chrono::high_resolution_clock::now();

    for (int n = 0; n < Iter; ++n) {
      results.clear();
      for (int i = 0; i < N; ++i) {
        results.push_back(prop.propagate(pars[i], single_options));
      }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();

    const auto min_step_res = std::min_element(
        begin(results), end(results),
        [](auto& a, auto& b) { return (*a).steps < (*b).steps; });
    const auto max_step_res = std::max_element(
        begin(results), end(results),
        [](auto& a, auto& b) { return (*a).steps < (*b).steps; });

    std::cout << "\nSINGLE STEPPER\n";
    std::cout << "\tsteps [min, max]: " << min_step_res->value().steps << ", "
              << max_step_res->value().steps << "\n";
    std::cout << "\ttime : "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() /
                     Iter
              << " ms\n";

    // Process results
    for (int i = 0; i < N; ++i) {
      const auto single_stepper_result =
          results[i].value().get<Acts::detail::SteppingLogger::result_type>();

      const auto steplog = std::vector<decltype(single_stepper_result.steps)>{
          single_stepper_result.steps};

      if (do_obj_export)
        export_tracks_to_obj(steplog, "single-stepper-" + std::to_string(i));
    }
  }

  //////////////////////////
  // LOOP Stepper
  //////////////////////////
  if (stepper_type == "loop" || stepper_type == "all") {
    const auto prop =
        make_propagator<Acts::MultiEigenStepper<>>(bfield_value, detector);

    // One dummy run to create object
    auto multi_result = prop.propagate(multi_pars, multi_options);

    // Run propagation and measure time
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < Iter; ++n) {
      multi_result = prop.propagate(multi_pars, multi_options);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "\nLOOP STEPPER\n";
    std::cout << "\tsteps: " << multi_result.value().steps << "\n";
    std::cout << "\ttime:  "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() /
                     Iter
              << " ms\n";

    // Process results
    const auto multi_step_logs =
        multi_result.value().get<MultiSteppingLogger::result_type>();

    const auto average_steplog = std::vector<std::vector<Acts::detail::Step>>{
        multi_step_logs.averaged_steps};

    if (do_obj_export) {
      export_tracks_to_obj(multi_step_logs.steps, "components-loop-stepper");
      export_tracks_to_obj(average_steplog, "average-loop-stepper");
    }
  }

  //////////////////////////
  // SIMD Stepper
  //////////////////////////
  if (stepper_type == "simd" || stepper_type == "all") {
    using SimdScalar = Acts::SimdType<N>;
    using DefaultExt = Acts::detail::NewGenericDefaultExtension<SimdScalar>;
    using DenseExt =
        Acts::detail::NewGenericDenseEnvironmentExtension<SimdScalar>;
    using ExtList = Acts::NewStepperExtensionList<DefaultExt, DenseExt>;
    using Reducer = Acts::WeightedComponentReducer<N>;

    const auto prop =
        make_propagator<Acts::MultiEigenStepperSIMD<N, Reducer, ExtList>>(
            bfield_value, detector);

    // One dummy run to get object
    auto multi_result = prop.propagate(multi_pars, multi_options);

    // Run propagation and measure time
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < Iter; ++n) {
      multi_result = prop.propagate(multi_pars, multi_options);
    }
    const auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "\nSIMD STEPPER\n";
    std::cout << "\tsteps: " << multi_result.value().steps << "\n";
    std::cout << "\ttime:  "
              << std::chrono::duration<double, std::milli>(t1 - t0).count() /
                     Iter
              << " ms\n";

    // Process results
    const auto multi_step_logs =
        multi_result.value().get<MultiSteppingLogger::result_type>();

    const auto average_steplog = std::vector<std::vector<Acts::detail::Step>>{
        multi_step_logs.averaged_steps};

    if (do_obj_export) {
      export_tracks_to_obj(multi_step_logs.steps, "components-simd-stepper");
      export_tracks_to_obj(average_steplog, "average-simd-stepper");
    }
  }

  //////////////////////////
  // Wrong stepper arg
  //////////////////////////
  if (stepper_type != "all" && stepper_type != "single" &&
      stepper_type != "loop" && stepper_type != "simd") {
    std::cerr << "Error: invalid stepper type '" << stepper_type << "'\n";
    return EXIT_FAILURE;
  }

  if (do_obj_export) {
    export_detector_to_obj(*detector);
  }
}
