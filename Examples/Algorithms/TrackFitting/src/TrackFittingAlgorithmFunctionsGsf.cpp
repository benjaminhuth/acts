// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Propagator/MultiEigenStepperLoop.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/TrackFitting/BetheHeitlerApprox.hpp"
#include "Acts/TrackFitting/GainMatrixSmoother.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/TrackFitting/GaussianSumFitter.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "ActsExamples/MagneticField/MagneticField.hpp"
#include "ActsExamples/TrackFitting/TrackFittingAlgorithm.hpp"

#include <filesystem>

#define USE_SINGLE_CMP_BETHE_HEITLER 1

using namespace ActsExamples;

namespace {
template <typename FitterFunction>
auto makeGsfOptions(
    const FitterFunction& f,
    const TrackFittingAlgorithm::GeneralFitterOptions& options) {
  Acts::GsfExtensions<Acts::VectorMultiTrajectory> extensions;
  extensions.updater.connect<
      &Acts::GainMatrixUpdater::operator()<Acts::VectorMultiTrajectory>>(
      &f.updater);

  Acts::GsfOptions<Acts::VectorMultiTrajectory> gsfOptions{
      options.geoContext,
      options.magFieldContext,
      options.calibrationContext,
      extensions,
      options.logger,
      options.propOptions,
      &(*options.referenceSurface),
      f.maxComponents,
      f.abortOnError,
      f.disableAllMaterialHandling};

  return gsfOptions;
}

template <typename Fitter>
struct GsfFitterFunctionImpl
    : public ActsExamples::TrackFittingAlgorithm::TrackFitterFunction {
  Fitter trackFitter;
  Acts::GainMatrixUpdater updater;

  std::size_t maxComponents;
  bool abortOnError;
  bool disableAllMaterialHandling;

  GsfFitterFunctionImpl(Fitter&& f) : trackFitter(std::move(f)) {}

  ActsExamples::TrackFittingAlgorithm::TrackFitterResult operator()(
      const std::vector<std::reference_wrapper<
          const ActsExamples::IndexSourceLink>>& sourceLinks,
      const ActsExamples::TrackParameters& initialParameters,
      const ActsExamples::TrackFittingAlgorithm::GeneralFitterOptions& options,
      std::shared_ptr<Acts::VectorMultiTrajectory>& trajectory) const override {
    auto gsfOptions = makeGsfOptions(*this, options);
    gsfOptions.extensions.calibrator
        .template connect<&ActsExamples::MeasurementCalibrator::calibrate>(
            &options.calibrator.get());

    return trackFitter.fit(sourceLinks.begin(), sourceLinks.end(),
                           initialParameters, gsfOptions, trajectory);
  }
};

template <typename Fitter>
struct DirectedFitterFunctionImpl
    : public ActsExamples::TrackFittingAlgorithm::DirectedTrackFitterFunction {
  Fitter trackFitter;
  Acts::GainMatrixUpdater updater;

  std::size_t maxComponents;
  bool abortOnError;
  bool disableAllMaterialHandling;

  DirectedFitterFunctionImpl(Fitter&& f) : trackFitter(std::move(f)) {}

  ActsExamples::TrackFittingAlgorithm::TrackFitterResult operator()(
      const std::vector<std::reference_wrapper<
          const ActsExamples::IndexSourceLink>>& sourceLinks,
      const ActsExamples::TrackParameters& initialParameters,
      const ActsExamples::TrackFittingAlgorithm::GeneralFitterOptions& options,
      const std::vector<const Acts::Surface*>& sSequence,
      std::shared_ptr<Acts::VectorMultiTrajectory>& trajectory) const override {
    auto gsfOptions = makeGsfOptions(*this, options);
    gsfOptions.extensions.calibrator
        .template connect<&ActsExamples::MeasurementCalibrator::calibrate>(
            &options.calibrator.get());

    return trackFitter.fit(sourceLinks.begin(), sourceLinks.end(),
                           initialParameters, gsfOptions, sSequence,
                           trajectory);
  }
};
}  // namespace

std::shared_ptr<TrackFittingAlgorithm::TrackFitterFunction>
TrackFittingAlgorithm::makeGsfFitterFunction(
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry,
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField,
    std::string lowParametersPath, std::string highParametersPath,
    std::size_t maxComponents, bool abortOnError,
    bool disableAllMaterialHandling) {
  Acts::MultiEigenStepperLoop stepper(std::move(magneticField));
  Acts::Navigator::Config cfg{trackingGeometry};
  cfg.resolvePassive = false;
  cfg.resolveMaterial = true;
  cfg.resolveSensitive = true;
  Acts::Navigator navigator(cfg);
  Acts::Propagator propagator(std::move(stepper), std::move(navigator));

#if USE_SINGLE_CMP_BETHE_HEITLER
  std::vector<double> ts;
  std::generate(ts.begin(), ts.end(), [n = 0.0]() mutable { return std::exp(-0.001 * n++); });
  auto bhapp = Acts::BetheHeitlerSimulatedAnnealingMinimizer<9>(ts);
#else
  auto makeBehteHeitlerApprox = [&]() {
    if (std::filesystem::exists(lowParametersPath) &&
        std::filesystem::exists(highParametersPath)) {
      return Acts::AtlasBetheHeitlerApprox<6, 5>::loadFromFile(
          lowParametersPath, highParametersPath);
    } else {
      std::cout << "WARNING: Could not find files, use standard configuration\n";
      return Acts::AtlasBetheHeitlerApprox<6, 5>(Acts::bh_cdf_cmps6_order5_data,
                                                 Acts::bh_cdf_cmps6_order5_data,
                                                 true, true);
    }
  };

  auto bhapp = makeBehteHeitlerApprox();
#endif

  Acts::GaussianSumFitter<decltype(propagator), decltype(bhapp),
                          Acts::VectorMultiTrajectory>
      trackFitter(std::move(propagator), std::move(bhapp));

  // build the fitter functions. owns the fitter object.
  auto fitterFunction =
      std::make_shared<GsfFitterFunctionImpl<decltype(trackFitter)>>(
          std::move(trackFitter));
  fitterFunction->maxComponents = maxComponents;
  fitterFunction->abortOnError = abortOnError;
  fitterFunction->disableAllMaterialHandling = disableAllMaterialHandling;

  return fitterFunction;
}

std::shared_ptr<TrackFittingAlgorithm::DirectedTrackFitterFunction>
TrackFittingAlgorithm::makeGsfFitterFunction(
    std::shared_ptr<const Acts::MagneticFieldProvider> /*magneticField*/,
    std::size_t /*maxComponents*/, bool /*abortOnError*/,
    bool /*disableAllMaterialHandling*/) {
#if 0
  Acts::MultiEigenStepperLoop stepper(std::move(magneticField));
  Acts::DirectNavigator navigator;
  Acts::Propagator propagator(std::move(stepper), navigator);
#if USE_SINGLE_CMP_BETHE_HEITLER
  auto bhapp = Acts::detail::BetheHeitlerApproxSingleCmp();
#else
  auto bhapp = Acts::detail::BetheHeitlerApprox<6, 5>(
      Acts::detail::bh_cdf_cmps6_order5_data);
#endif

  Acts::GaussianSumFitter<decltype(propagator), Acts::VectorMultiTrajectory,
                          decltype(bhapp)>
      trackFitter(std::move(propagator), std::move(bhapp));

  // build the fitter functions. owns the fitter object.
  auto fitterFunction =
      std::make_shared<DirectedFitterFunctionImpl<decltype(trackFitter)>>(
          std::move(trackFitter));
  fitterFunction->maxComponents = maxComponents;
  fitterFunction->abortOnError = abortOnError;
  fitterFunction->disableAllMaterialHandling = disableAllMaterialHandling;

  return fitterFunction;
#endif
  throw std::runtime_error("Direct fitting now not implemented");
}
