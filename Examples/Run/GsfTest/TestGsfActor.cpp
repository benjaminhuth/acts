// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <iostream>

#include "GsfActor.hpp"
#include "MultiEigenStepperLoop.hpp"
#include "MultiEigenStepperSIMD.hpp"
#include "NewGenericDefaultExtension.hpp"
#include "NewGenericDenseEnvironmentExtension.hpp"
#include "NewStepperExtensionList.hpp"
#include "TestHelpers.hpp"

constexpr int N = 4;

int main() {
  using namespace Acts::UnitLiterals;

  const double bfield_value = 2_T;
  auto [start_surface, detector] = build_tracking_geometry();

  const auto logger =
      Acts::getDefaultLogger("GsfActorTest", Acts::Logging::VERBOSE);

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

  // Options
  struct DummySourceLink {
    Acts::GeometryIdentifier geometryId() const { return {}; }
  };

  using GSF = Acts::GaussianSumFitter;

  using MultiActionList =
      Acts::ActionList<MultiSteppingLogger, GSF::Actor<DummySourceLink>/*, Acts::MaterialInteractor*/>;
  using AbortList = Acts::AbortList<Acts::EndOfWorldReached>;
  using MultiPropagatorOptions =
      Acts::DenseStepperPropagatorOptions<MultiActionList, AbortList>;

  MultiPropagatorOptions multi_options(geoCtx, magCtx,
                                       Acts::LoggerWrapper(*logger));

  //////////////////////////
  // LOOP Stepper
  //////////////////////////
  {
    using DefaultExt = Acts::detail::GenericDefaultExtension<Acts::ActsScalar>;
    using DenseExt =
        Acts::detail::GenericDenseEnvironmentExtension<Acts::ActsScalar>;
    using ExtList = Acts::StepperExtensionList<DefaultExt, DenseExt>;
    const auto prop = make_propagator<Acts::MultiEigenStepperLoop<ExtList>>(
        bfield_value, detector);

    auto multi_result = prop.propagate(multi_pars, multi_options);
  }

  //////////////////////////
  // SIMD Stepper
  //////////////////////////
  {
    using SimdScalar = Acts::SimdType<N>;
    using DefaultExt = Acts::detail::NewGenericDefaultExtension<SimdScalar>;
    using DenseExt =
        Acts::detail::NewGenericDenseEnvironmentExtension<SimdScalar>;
    using ExtList = Acts::NewStepperExtensionList<DefaultExt, DenseExt>;

    const auto prop = make_propagator<Acts::MultiEigenStepperSIMD<N, ExtList>>(
        bfield_value, detector);

    auto multi_result = prop.propagate(multi_pars, multi_options);
  }
}
