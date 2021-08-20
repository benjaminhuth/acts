// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "GsfAlgorithmFunction.hpp"

#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/TrackFitting/GainMatrixSmoother.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/TrackFitting/detail/VoidKalmanComponents.hpp"
#include "ActsExamples/EventData/Measurement.hpp"

#include "GaussianSumFitter.hpp"
#include "MultiEigenStepperLoop.hpp"

namespace {
bool gsfThrowOnAbort = false;
std::size_t maxComponents = 4;
}  // namespace

void setGsfAbortOnError(bool aoe) {
  gsfThrowOnAbort = aoe;
}

bool getGsfAbortOnError() {
  return gsfThrowOnAbort;
}

void setGsfMaxComponents(std::size_t c) {
  maxComponents = c;
}

std::size_t getGsfMaxComponents() {
  return maxComponents;
}

// Kalman Components
using Updater = Acts::GainMatrixUpdater;
using Smoother = Acts::GainMatrixSmoother;
using OutlierFinder = Acts::VoidOutlierFinder;
using Calibrator = ActsExamples::MeasurementCalibrator;

// Stepper and Propagator
using DefaultExt = Acts::detail::GenericDefaultExtension<Acts::ActsScalar>;
using ExtList = Acts::StepperExtensionList<DefaultExt>;
using Stepper = Acts::MultiEigenStepperLoop<ExtList>;
using Propagator = Acts::Propagator<Stepper, Acts::DirectNavigator>;

// The Fitter
using Fitter = Acts::GaussianSumFitter<Propagator>;

struct GsfFitterFunction
    : public ActsExamples::TrackFittingAlgorithm::DirectedTrackFitterFunction {
  Fitter trackFitter;

  GsfFitterFunction(Fitter&& f) : trackFitter(std::move(f)) {}
  ~GsfFitterFunction() {}

  ActsExamples::TrackFittingAlgorithm::TrackFitterResult operator()(
      const std::vector<ActsExamples::IndexSourceLink>& sourceLinks,
      const ActsExamples::TrackParameters& initialParameters,
      const ActsExamples::TrackFittingAlgorithm::TrackFitterOptions&
          kalmanOptions,
      const std::vector<const Acts::Surface*>& sSequence) const {
    Acts::GsfOptions<Calibrator, OutlierFinder> gsfOptions{
        kalmanOptions.calibrator,
        kalmanOptions.outlierFinder,
        kalmanOptions.geoContext,
        kalmanOptions.magFieldContext,
        kalmanOptions.referenceSurface,
        kalmanOptions.logger,
        gsfThrowOnAbort,
        maxComponents};

    return trackFitter.fit(sourceLinks, initialParameters, gsfOptions,
                           sSequence);
  };
};

std::shared_ptr<ActsExamples::TrackFittingAlgorithm::DirectedTrackFitterFunction>
makeGsfFitterFunction(
    std::shared_ptr<const Acts::TrackingGeometry> /*trackingGeometry*/,
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
    Acts::LoggerWrapper logger) {
  using namespace Acts::UnitLiterals;

  Stepper stepper(std::move(magneticField), logger);
  stepper.setOverstepLimit(1_mm);
  Acts::DirectNavigator navigator;
  Propagator propagator(std::move(stepper), std::move(navigator));
  Fitter trackFitter(std::move(propagator));

  // build the fitter functions. owns the fitter object.
  return std::make_shared<GsfFitterFunction>(std::move(trackFitter));
}
