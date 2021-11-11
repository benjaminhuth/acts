// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/TrackFitting/GainMatrixSmoother.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/TrackFitting/detail/VoidKalmanComponents.hpp"
#include "ActsExamples/EventData/Measurement.hpp"

#include "GsfAlgorithmFunction.hpp"
#include "GsfInfrastructure/GaussianSumFitter.hpp"
#include "GsfInfrastructure/MultiEigenStepperLoop.hpp"

/////////////////////////////////////////
// The Fitter for the Standard Navigator
/////////////////////////////////////////

// Kalman Components
using Updater = Acts::GainMatrixUpdater;
using Smoother = Acts::GainMatrixSmoother;
using OutlierFinder = Acts::VoidOutlierFinder;
using Calibrator = ActsExamples::MeasurementCalibrator;

// Stepper and Propagator
using DefaultExt = Acts::detail::GenericDefaultExtension<Acts::ActsScalar>;
using ExtList = Acts::StepperExtensionList<DefaultExt>;
using Stepper = Acts::MultiEigenStepperLoop<ExtList>;

using StandardPropagator = Acts::Propagator<Stepper, Acts::Navigator>;
using StandardFitter = Acts::GaussianSumFitter<StandardPropagator>;

struct GsfStandardFitterFunction
    : public ActsExamples::TrackFittingAlgorithm::TrackFitterFunction {
  StandardFitter trackFitter;

  GsfStandardFitterFunction(StandardFitter&& f) : trackFitter(std::move(f)) {}
  ~GsfStandardFitterFunction() {}

  ActsExamples::TrackFittingAlgorithm::TrackFitterResult operator()(
      const std::vector<std::reference_wrapper<const ActsExamples::IndexSourceLink>>& sourceLinks,
      const ActsExamples::TrackParameters& initialParameters,
      const ActsExamples::TrackFittingAlgorithm::TrackFitterOptions&
          kalmanOptions) const {
    Acts::GsfOptions<Calibrator, OutlierFinder> gsfOptions{
        kalmanOptions.calibrator,
        kalmanOptions.outlierFinder,
        kalmanOptions.geoContext,
        kalmanOptions.magFieldContext,
        kalmanOptions.referenceSurface,
        kalmanOptions.logger,
        getGsfAbortOnError(),
        getGsfMaxComponents(),
        getGsfMaxSteps(),
        getGsfLoopProtection()};

    return trackFitter.fit(sourceLinks.begin(), sourceLinks.end(), initialParameters, gsfOptions);
  };
};

std::shared_ptr<ActsExamples::TrackFittingAlgorithm::TrackFitterFunction>
makeGsfStandardFitterFunction(
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
    Acts::LoggerWrapper logger) {
  using namespace Acts::UnitLiterals;

  Stepper stepper(std::move(magneticField), logger);
  stepper.setOverstepLimit(1_mm);

  Acts::Navigator::Config cfg;
  cfg.trackingGeometry = trackingGeometry;

  Acts::Navigator navigator(cfg);
  StandardPropagator propagator(std::move(stepper), std::move(navigator));
  StandardFitter trackFitter(std::move(propagator));

  // build the fitter functions. owns the fitter object.
  return std::make_shared<GsfStandardFitterFunction>(std::move(trackFitter));
}
