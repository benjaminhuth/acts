// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "GsfAlgorithmFunction.hpp"

namespace {
bool gsfAbortOnError = false;
std::size_t maxComponents = 4;
}  // namespace

void setGsfAbortOnError(bool aoe) {
  gsfAbortOnError = aoe;
}

bool getGsfAbortOnError() {
  return gsfAbortOnError;
}

void setGsfMaxComponents(std::size_t c) {
  maxComponents = c;
}

std::size_t getGsfMaxComponents() {
  return maxComponents;
}
