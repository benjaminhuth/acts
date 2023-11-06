// This file is part of the Acts project.
//
// Copyright (C) 2017-2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Propagator/MultiEigenStepperLoop.hpp"

#include <variant>

#include "MultiStepperBenchmark.hpp"

int main(int argc, char* argv[]) {
  using Variant = std::variant<Acts::MultiEigenStepperLoop<>>;

  auto factory = [](const auto& bfield, auto loglevel, auto components) {
    auto name = "LoopStepper" + std::to_string(components);
    Variant v = Acts::MultiEigenStepperLoop<>(
        bfield, Acts::getDefaultLogger(
                    name, static_cast<Acts::Logging::Level>(loglevel)));
    return v;
  };

  return benchmarkMultiStepper(factory, "MultiStepperLoop", argc, argv);
}
