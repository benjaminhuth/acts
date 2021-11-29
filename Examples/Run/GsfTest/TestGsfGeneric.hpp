// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Utilities/Logger.hpp"

#include "ActsExamples/TrackFinding/SpacePointMaker.hpp"
#include "ActsExamples/Digitization/DigitizationConfig.hpp"
#include "ActsExamples/Framework/IContextDecorator.hpp"

#include "GsfAlgorithmFunction.hpp"

#include <memory>

struct GsfTestSettings {
    Acts::Logging::Level globalLogLevel;
    Acts::Logging::Level gsfLogLevel;
    uint64_t seed;
    
    std::shared_ptr<const Acts::TrackingGeometry> geometry;
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField;
    std::vector<std::shared_ptr<ActsExamples::IContextDecorator>> contextDecorators;
    
    bool doGsf;
    bool doKalman;
    bool doRefit;
    bool doDirectNavigation;
    bool estimateParsFromSeed;

    StepperInteface stepperInterface;
    
    std::size_t maxComponents;
    std::size_t maxSteps;
    bool gsfAbortOnError;
    bool gsfLoopProtection;
    bool gsfApplyMaterialEffects;
    
    double inflation;
    
    std::size_t numParticles;
    double phiMin;
    double phiMax;
    double thetaMin;
    double thetaMax;
    double pMin;
    double pMax;
    
    std::function<ActsExamples::DigitizationConfig()> digiConfigFactory;
    ActsExamples::SpacePointMaker::Config spmConfig;
    
    std::string objOutputDir;
};


int testGsf(const GsfTestSettings &settings);
