// This file is part of the Acts project.
//
// Copyright (C) 2017-2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <experimental/simd>

template <typename T, typename Abi>
bool operator||(bool a, const std::experimental::simd_mask<T, Abi> &s) {
  bool r = true;
  for (std::size_t i = 0; i < s.size(); ++i) {
    r = r && (a || s[i]);
  }
  return r;
}

#include "Acts/Propagator/MultiEigenStepperSIMD.hpp"

#include <variant>

#include <Eigen/Dense>

#include "MultiStepperBenchmark.hpp"

template <int N>
using SimdExtension = Acts::StepperExtensionList<
    Acts::detail::GenericDefaultExtension<Acts::SimdType<N>>>;

template <int N>
using SimdStepper = Acts::MultiEigenStepperSIMD<N, SimdExtension<N>>;

int main(int argc, char *argv[]) {
  using Variant = std::variant<SimdStepper<1>, SimdStepper<2>, SimdStepper<4>,
                               SimdStepper<8>, SimdStepper<12>, SimdStepper<16>,
                               SimdStepper<32>>;

  auto factory = [](const auto &bfield, auto loglevel, auto components) {
    auto name = "SIMDStepper" + std::to_string(components);
    auto logger = Acts::getDefaultLogger(
        name, static_cast<Acts::Logging::Level>(loglevel));
    std::optional<Variant> v;
    if (components == 1) {
      v = SimdStepper<1>(bfield, std::move(logger));
    } else if (components == 2) {
      v = SimdStepper<2>(bfield, std::move(logger));
    } else if (components == 4) {
      v = SimdStepper<4>(bfield, std::move(logger));
    } else if (components == 8) {
      v = SimdStepper<8>(bfield, std::move(logger));
    } else if (components == 12) {
      v = SimdStepper<12>(bfield, std::move(logger));
    } else if (components == 16) {
      v = SimdStepper<16>(bfield, std::move(logger));
    } else if (components == 32) {
      v = SimdStepper<32>(bfield, std::move(logger));
    } else {
      throw std::runtime_error("SIMD stepper not compiled for " +
                               std::to_string(components) + " components");
    }
    Variant vv = std::move(*v);
    return vv;
  };

  return benchmarkMultiStepper(factory, "MultiStepperSIMD", argc, argv);
}
