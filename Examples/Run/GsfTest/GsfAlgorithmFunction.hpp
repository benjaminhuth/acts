// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "ActsExamples/TrackFitting/TrackFittingAlgorithm.hpp"

/// Make function of
std::shared_ptr<
    ActsExamples::TrackFittingAlgorithm::DirectedTrackFitterFunction>
makeGsfDirectFitterFunction(
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
    Acts::LoggerWrapper logger);

std::shared_ptr<ActsExamples::TrackFittingAlgorithm::TrackFitterFunction>
makeGsfStandardFitterFunction(
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
    Acts::LoggerWrapper logger);

void setGsfAbortOnError(bool);
bool getGsfAbortOnError();

void setGsfLoopProtection(bool);
bool getGsfLoopProtection();

void setGsfMaxComponents(std::size_t);
std::size_t getGsfMaxComponents();

void setGsfMaxSteps(std::size_t);
std::size_t getGsfMaxSteps();

void setGsfApplyMaterialEffects(bool);
bool getGsfApplyMaterialEffects();

/// Choose the interface of the stepper to the navigator
enum class StepperInteface {
    average,
    maxMomentum
};

void setGsfStepperInterface(StepperInteface);
StepperInteface getGsfStepperInterface();

void setBetheHeitlerHighX0Path(const std::string &path);
const std::string &getBetheHeitlerHighX0Path();

void setBetheHeitlerLowX0Path(const std::string &path);
const std::string &getBetheHeitlerLowX0Path();
