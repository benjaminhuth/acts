// This file is part of the Acts project.
//
// Copyright (C) 2017-2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/MultiComponentTrackParameters.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/MagneticField/MagneticFieldContext.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/Tests/CommonHelpers/BenchmarkTools.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <iostream>

#include <boost/program_options.hpp>

template <typename multi_stepper_variant_factor_t>
int benchmarkMultiStepper(
    const multi_stepper_variant_factor_t &multiStepperVariantFactory,
    const std::string &name, int argc, char **argv) {
  namespace po = boost::program_options;
  using namespace Acts;
  using namespace Acts::UnitLiterals;

  unsigned int toys = 1;
  double ptInGeV = 1;
  double BzInT = 1;
  double maxPathInM = 1;
  unsigned int lvl = Acts::Logging::INFO;
  bool withCov = true;
  std::size_t components;

  try {
    po::options_description desc("Allowed options");
    // clang-format off
  desc.add_options()
      ("help", "produce help message")
      ("toys",po::value<unsigned int>(&toys)->default_value(20000),"number of tracks to propagate")
      ("pT",po::value<double>(&ptInGeV)->default_value(1),"transverse momentum in GeV")
      ("B",po::value<double>(&BzInT)->default_value(2),"z-component of B-field in T")
      ("path",po::value<double>(&maxPathInM)->default_value(5),"maximum path length in m")
      ("cov",po::value<bool>(&withCov)->default_value(true),"propagation with covariance matrix")
      ("components",po::value<std::size_t>(&components)->default_value(1), "components of the multistepper")
      ("verbose",po::value<unsigned int>(&lvl)->default_value(Acts::Logging::INFO),"logging level");
    // clang-format on
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
      std::cout << desc << std::endl;
      return 0;
    }
  } catch (std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }

  ACTS_LOCAL_LOGGER(getDefaultLogger("Main", Acts::Logging::Level(lvl)));

  // print information about profiling setup
  ACTS_INFO("Benchmarking " << name << "(" << components << ")")
  ACTS_INFO("propagating " << toys << " tracks with pT = " << ptInGeV
                           << "GeV in a " << BzInT << "T B-field");

  // Create a test context
  GeometryContext tgContext = GeometryContext();
  MagneticFieldContext mfContext = MagneticFieldContext();

  PropagatorOptions<> options(tgContext, mfContext);
  options.pathLimit = maxPathInM * UnitConstants::m;

  FreeVector free = FreeVector::Zero();
  free.segment<3>(eFreeDir0) = Vector3(1, 0, 0);
  free[eFreeQOverP] = +1 / ptInGeV;

  BoundSquareMatrix cov;
  // clang-format off
  cov << 10_mm, 0, 0, 0, 0, 0,
         0, 10_mm, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0,
         0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1_e / 10_GeV, 0,
         0, 0, 0, 0, 0, 0;
  // clang-format on

  std::optional<BoundSquareMatrix> covOpt = std::nullopt;
  if (withCov) {
    covOpt = cov;
  }

  // go to bound
  auto surface = Acts::Surface::makeShared<PerigeeSurface>(Vector3{0., 0., 0.});
  auto bound =
      Acts::detail::transformFreeToBoundParameters(free, *surface, tgContext)
          .value();

  std::vector<std::tuple<double, BoundVector, std::optional<BoundSquareMatrix>>>
      vec(components, {1. / components, bound, covOpt});
  MultiComponentBoundTrackParameters multiBound(
      surface->getSharedPtr(), std::move(vec), ParticleHypothesis::pion());

  auto bField =
      std::make_shared<ConstantBField>(Vector3{0, 0, BzInT * UnitConstants::T});
  auto variantStepper =
      multiStepperVariantFactory(std::move(bField), lvl, components);

  return std::visit(
      [&](auto &stepper) {
        using Stepper_type = std::decay_t<decltype(stepper)>;
        using Propagator_type = Propagator<Stepper_type>;

        Propagator_type propagator(std::move(stepper), detail::VoidNavigator{},
                                   logger().cloneWithSuffix("Propagator"));

        double totalPathLength = 0;
        size_t num_iters = 0;

        Vector3 endPos = Vector3::Zero();
        std::size_t steps = 0;
        const auto propagation_bench_result = Acts::Test::microBenchmark(
            [&] {
              auto r = propagator.propagate(multiBound, options).value();
              if (totalPathLength == 0.) {
                endPos = r.endParameters->position(tgContext);
                steps = r.steps;
              }
              totalPathLength += r.pathLength;
              ++num_iters;
              return r;
            },
            30, toys);

        ACTS_INFO("In first iteration reached position "
                  << endPos.transpose() << " in " << steps << " steps");

        ACTS_INFO("Execution stats: " << propagation_bench_result);
        ACTS_INFO("Average path length = " << totalPathLength / num_iters / 1_mm
                                           << "mm");
        return 0;
      },
      variantStepper);
}
