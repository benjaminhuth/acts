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
bool doLoopProtection = false;
bool applyMaterialEffects = true;
std::size_t maxComponents = 4;
std::size_t maxSteps = 100;
StepperInteface interface = StepperInteface::average;
std::string bethe_heitler_low_x0_path;
std::string bethe_heitler_high_x0_path;
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

void setGsfMaxSteps(std::size_t s) {
  maxSteps = s;
}

std::size_t getGsfMaxSteps() {
  return maxSteps;
}

void setGsfLoopProtection(bool lp) {
  doLoopProtection = lp;
}

bool getGsfLoopProtection() {
  return doLoopProtection;
}

void setGsfApplyMaterialEffects(bool e) {
    applyMaterialEffects = e;
}

bool getGsfApplyMaterialEffects() {
    return applyMaterialEffects;
}

void setGsfStepperInterface(StepperInteface i) {
    interface = i;
}

StepperInteface getGsfStepperInterface()
{
    return interface;
}

void setBetheHeitlerHighX0Path(const std::string& path) {
    bethe_heitler_high_x0_path = path;
}

const std::string &getBetheHeitlerHighX0Path() {
    return bethe_heitler_high_x0_path;
}

void setBetheHeitlerLowX0Path(const std::string& path) {
    bethe_heitler_low_x0_path = path;
}

const std::string &getBetheHeitlerLowX0Path() {
    return bethe_heitler_low_x0_path;
}
