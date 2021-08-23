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
#include "Acts/Material/Material.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <chrono>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "GsfInfrastructure/MultiEigenStepperLoop.hpp"
#include "GsfInfrastructure/MultiEigenStepperSIMD.hpp"
#include "GsfInfrastructure/NewGenericDefaultExtension.hpp"
#include "GsfInfrastructure/NewGenericDenseEnvironmentExtension.hpp"
#include "GsfInfrastructure/NewStepperExtensionList.hpp"
#include "GsfInfrastructure/SimdHelpers.hpp"
#include "TestHelpers.hpp"

int main(int argc, char** argv) {
  using namespace Acts::UnitLiterals;
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

  auto magField =
      std::make_shared<MagneticField>(Acts::Vector3(0.0, 0.0, bfield_value));

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

  // Determine number of tracks and iterations
#ifdef NITER
  constexpr int Iter = NITER;
#else
  constexpr int Iter = 1000;
#endif

#ifdef NTRACKS
  constexpr int N = NTRACKS;
#else
  constexpr int N = 4;
#endif

  std::cout << "Using " << N << " parallel tracks (" << Iter
            << " iterations)\n";

  // Setup tracks
  const auto track_data_vector = []() {
    const double l0{0.}, l1{0.}, theta{0.5 * M_PI}, phi{0.}, p{50._GeV}, q{-1.},
        t{0.};
    std::vector<std::tuple<double, Acts::BoundVector, Acts::BoundSymMatrix>>
        vec;

    const double factor = 0.1;

    for (int i = 0; i < N; ++i) {
      Acts::BoundVector pars;
      pars << l0, l1, phi, theta, q / ((factor * i + 1) * p), t;
      vec.push_back({1. / N, pars, Acts::BoundSymMatrix::Identity()});
    }

    return vec;
  }();

  Acts::MultiComponentBoundTrackParameters<Acts::SinglyCharged> multi_pars(
      start_surface, track_data_vector);

  // Action list and abort list TODO currently problem wiht material interactor
  using SingleActionList = Acts::ActionList<
      Acts::detail::SteppingLogger /*, Acts::MaterialInteractor*/>;
  using MultiActionList =
      Acts::ActionList<MultiSteppingLogger /*, Acts::MaterialInteractor*/>;
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

  //////////////////////////
  // SINGLE Stepper
  //////////////////////////
  if (stepper_type == "single" || stepper_type == "all") {
    const auto prop = make_propagator<Acts::EigenStepper<>>(magField, detector);

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
    using DefaultExt = Acts::detail::GenericDefaultExtension<Acts::ActsScalar>;
    using DenseExt =
        Acts::detail::GenericDenseEnvironmentExtension<Acts::ActsScalar>;
    using ExtList = Acts::StepperExtensionList<DefaultExt, DenseExt>;
    const auto prop = make_propagator<Acts::MultiEigenStepperLoop<ExtList>>(
        magField, detector);

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

    const auto prop = make_propagator<Acts::MultiEigenStepperSIMD<N, ExtList>>(
        magField, detector);

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
