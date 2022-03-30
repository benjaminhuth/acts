// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Propagator/MultiEigenStepperLoop.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/TrackFitting/GainMatrixSmoother.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/TrackFitting/GaussianSumFitter.hpp"
#include "Acts/TrackFitting/detail/VoidKalmanComponents.hpp"
#include "ActsExamples/EventData/Measurement.hpp"

#include "GsfAlgorithmFunction.hpp"

/////////////////////////////////////////
// The Fitter for the Standard Navigator
/////////////////////////////////////////

// Stepper and Propagator
using DefaultExt = Acts::detail::GenericDefaultExtension<Acts::ActsScalar>;
using ExtList = Acts::StepperExtensionList<DefaultExt>;
using AvgStepper = Acts::MultiEigenStepperLoop<>;
using MaxStepper =
    Acts::MultiEigenStepperLoop<ExtList, Acts::MaxMomentumReducerLoop>;

template <typename fitter_t>
struct GsfStandardFitterFunction
    : public ActsExamples::TrackFittingAlgorithm::TrackFitterFunction {
  fitter_t trackFitter;

  GsfStandardFitterFunction(fitter_t&& f) : trackFitter(std::move(f)) {}
  ~GsfStandardFitterFunction() {}

  ActsExamples::TrackFittingAlgorithm::TrackFitterResult operator()(
      const std::vector<std::reference_wrapper<
          const ActsExamples::IndexSourceLink>>& sourceLinks,
      const ActsExamples::TrackParameters& initialParameters,
      const ActsExamples::TrackFittingAlgorithm::TrackFitterOptions&
          kalmanOptions) const {
    const bool disableMaterialHandling = !getGsfApplyMaterialEffects();
            
    Acts::GsfOptions gsfOptions{kalmanOptions.geoContext,
                                kalmanOptions.magFieldContext,
                                kalmanOptions.calibrationContext,
                                kalmanOptions.extensions,
                                kalmanOptions.logger,
                                kalmanOptions.propagatorPlainOptions,
                                kalmanOptions.referenceSurface,
                                getGsfAbortOnError(),
                                getGsfMaxComponents(),
                                disableMaterialHandling};

    gsfOptions.propagatorPlainOptions.maxSteps = getGsfMaxSteps();
    gsfOptions.propagatorPlainOptions.loopProtection = getGsfLoopProtection();

    return trackFitter.fit(sourceLinks.begin(), sourceLinks.end(),
                           initialParameters, gsfOptions);
  };
};

std::shared_ptr<ActsExamples::TrackFittingAlgorithm::TrackFitterFunction>
makeGsfStandardFitterFunction(
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
    Acts::LoggerWrapper logger) {
  using namespace Acts::UnitLiterals;

  auto makeFitter = [&](auto&& stepper) {
    using Stepper = std::decay_t<decltype(stepper)>;
    using Propagator = Acts::Propagator<Stepper, Acts::Navigator>;
    using Fitter = Acts::GaussianSumFitter<Propagator>;

    Acts::Navigator::Config cfg;
    cfg.trackingGeometry = trackingGeometry;

    Acts::Navigator navigator(cfg);
    Propagator propagator(std::move(stepper), std::move(navigator));

    Fitter trackFitter = [&]() {
      if (getBetheHeitlerHighX0Path().empty() &&
          getBetheHeitlerLowX0Path().empty()) {
        return Fitter(std::move(propagator));
      } else {
        const auto low_data = Acts::detail::load_bethe_heitler_data<6,5>(getBetheHeitlerLowX0Path());
        const auto high_data = Acts::detail::load_bethe_heitler_data<6,5>(getBetheHeitlerLowX0Path());

        Acts::detail::BetheHeitlerApprox<6,5> bethe_heitler_approx(low_data, high_data);

        return Fitter(std::move(propagator), std::move(bethe_heitler_approx));
      }
    }();

    // build the fitter functions. owns the fitter object.
    return std::make_shared<GsfStandardFitterFunction<Fitter>>(
        std::move(trackFitter));
  };

  switch (getGsfStepperInterface()) {
    case StepperInteface::average: {
      AvgStepper stepper(std::move(magneticField), logger);
      // TODO dont know why this was there, but in case of some problems
      // hopefully remember of it
      //       stepper.setOverstepLimit(1_mm);
      return makeFitter(stepper);
    }

    case StepperInteface::maxMomentum: {
      throw std::runtime_error(
          "not compiled for build-time reasons at the moment");
      //       MaxStepper stepper(std::move(magneticField), logger);
      //       stepper.setOverstepLimit(1_mm);
      //       return makeFitter(stepper);
    }
  }
}
