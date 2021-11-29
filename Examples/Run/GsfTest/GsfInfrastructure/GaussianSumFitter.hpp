// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/StandardAborters.hpp"

#include <fstream>

#include "GsfActor.hpp"
#include "MultiStepperAborters.hpp"
#include "MultiSteppingLogger.hpp"

#define RETURN_ERROR_OR_ABORT_FIT(error) \
  if (options.abortOnError) {            \
    std::abort();                        \
  } else {                               \
    return error;                        \
  }

using namespace std::string_literals;

constexpr static auto myNAN =
    std::numeric_limits<Acts::ActsScalar>::quiet_NaN();

template <std::size_t N>
class SimpleCsvWriter {
  std::ofstream m_file;

 public:
  SimpleCsvWriter(const std::string& filename,
                  const std::array<std::string, N>& headers)
      : m_file(filename, std::ios::app) {
    for (auto header : headers) {
      m_file << header << ",";
    }
    m_file << "\n";
  }

  template <typename... Args>
  void write(const Args&... args) {
    static_assert(sizeof...(Args) == N);
    ((m_file << args << ","), ...);
    m_file << "\n";
  }
};

namespace Acts {

template <typename calibrator_t, typename outlier_finder_t>
struct GsfOptions {
  using Calibrator = calibrator_t;
  using OutlierFinder = outlier_finder_t;

  Calibrator calibrator;
  OutlierFinder outlierFinder;

  std::reference_wrapper<const GeometryContext> geoContext;
  std::reference_wrapper<const MagneticFieldContext> magFieldContext;

  const Surface* referenceSurface = nullptr;

  LoggerWrapper logger;

  bool abortOnError = true;

  std::size_t maxComponents = 4;
  std::size_t maxSteps = 1000;
  bool loopProtection = true;

  bool applyMaterialEffects = true;
};

template <typename propagator_t>
struct GaussianSumFitter {
  GaussianSumFitter(propagator_t propagator)
      : m_propagator(std::move(propagator)),
        m_failStatistics("gsf-fail-statistics.csv",
                         {"errorMsg", "fwdSteps", "fwdPathlength", "bwdSteps",
                          "bwdPathLenght", "absoluteMomentum",
                          "transverseMomentum", "theta"}),
        m_successStatistics(
            "gsf-success-statistics.csv",
            {"fwdSteps", "fwdPathlength", "bwdSteps", "bwdPathLenght",
             "absoluteMomentum", "transverseMomentum", "theta"}) {}

  /// The navigator type
  using GsfNavigator = typename propagator_t::Navigator;

  /// The propagator instance used by the fit function
  propagator_t m_propagator;

  /// Some output files for debugging
  mutable SimpleCsvWriter<8> m_failStatistics;
  mutable SimpleCsvWriter<7> m_successStatistics;

  /// @brief The fit function for the Direct navigator
  template <typename source_link_it_t, typename start_parameters_t,
            typename calibrator_t, typename outlier_finder_t>
  Acts::Result<Acts::KalmanFitterResult> fit(
      source_link_it_t begin, source_link_it_t end,
      const start_parameters_t& sParameters,
      const GsfOptions<calibrator_t, outlier_finder_t>& options,
      const std::vector<const Surface*>& sSequence) const {
    // Check if we have the correct navigator
    static_assert(
        std::is_same_v<DirectNavigator, typename propagator_t::Navigator>);

    using ThisGsfActor = GsfActor<GainMatrixUpdater, outlier_finder_t,
                                  calibrator_t, GainMatrixSmoother>;

    // Initialize the forward propagation with the DirectNavigator
    auto fwdPropInitializer = [&sSequence](const auto& opts,
                                           const auto& logger) {
      using Actors = ActionList<ThisGsfActor, DirectNavigator::Initializer>;
      using Aborters = AbortList<>;

      PropagatorOptions<Actors, Aborters> propOptions(
          opts.geoContext, opts.magFieldContext, logger);

      propOptions.loopProtection = opts.loopProtection;
      propOptions.actionList.template get<DirectNavigator::Initializer>()
          .navSurfaces = sSequence;

      return propOptions;
    };

    // Initialize the backward propagation with the DirectNavigator
    auto bwdPropInitializer = [&sSequence](const auto& opts,
                                           const auto& logger) {
      using Actors = ActionList<ThisGsfActor, DirectNavigator::Initializer>;
      using Aborters = AbortList<>;

      PropagatorOptions<Actors, Aborters> propOptions(
          opts.geoContext, opts.magFieldContext, logger);

      std::vector<const Surface*> backwardSequence(
          std::next(sSequence.rbegin()), sSequence.rend());
      backwardSequence.push_back(opts.referenceSurface);

      propOptions.loopProtection = opts.loopProtection;
      propOptions.actionList.template get<DirectNavigator::Initializer>()
          .navSurfaces = std::move(backwardSequence);

      return propOptions;
    };

    return fit_impl(begin, end, sParameters, options, fwdPropInitializer,
                    bwdPropInitializer);
  }

  /// @brief The fit function for the standard navigator
  template <typename source_link_it_t, typename start_parameters_t,
            typename calibrator_t, typename outlier_finder_t>
  Acts::Result<Acts::KalmanFitterResult> fit(
      source_link_it_t begin, source_link_it_t end,
      const start_parameters_t& sParameters,
      const GsfOptions<calibrator_t, outlier_finder_t>& options) const {
    // Check if we have the correct navigator
    static_assert(std::is_same_v<Navigator, typename propagator_t::Navigator>);

    // Create the ActionList and AbortList
    using ThisGsfActor = GsfActor<GainMatrixUpdater, outlier_finder_t,
                                  calibrator_t, GainMatrixSmoother>;

    // Initialize the forward propagation with the DirectNavigator
    auto fwdPropInitializer = [](const auto& opts, const auto& logger) {
      using Actors = ActionList<ThisGsfActor>;
      using Aborters = AbortList<EndOfWorldReached>;

      PropagatorOptions<Actors, Aborters> propOptions(
          opts.geoContext, opts.magFieldContext, logger);
      propOptions.maxSteps = opts.maxSteps;
      propOptions.loopProtection = opts.loopProtection;

      return propOptions;
    };

    // Initialize the backward propagation with the DirectNavigator
    auto bwdPropInitializer = [](const auto& opts, const auto& logger) {
      using Actors = ActionList<ThisGsfActor>;
      using Aborters = AbortList<EndOfWorldReached>;

      PropagatorOptions<Actors, Aborters> propOptions(
          opts.geoContext, opts.magFieldContext, logger);
      propOptions.maxSteps = opts.maxSteps;
      propOptions.loopProtection = opts.loopProtection;

      return propOptions;
    };

    return fit_impl(begin, end, sParameters, options, fwdPropInitializer,
                    bwdPropInitializer);
  }

  template <typename source_link_it_t, typename start_parameters_t,
            typename calibrator_t, typename outlier_finder_t,
            typename fwd_prop_initializer_t, typename bwd_prop_initializer_t>
  Acts::Result<Acts::KalmanFitterResult> fit_impl(
      source_link_it_t begin, source_link_it_t end,
      const start_parameters_t& sParameters,
      const GsfOptions<calibrator_t, outlier_finder_t>& options,
      const fwd_prop_initializer_t& fwdPropInitializer,
      const bwd_prop_initializer_t& bwdPropInitializer) const {
    // The logger
    const auto& logger = options.logger;

    // Print some infos about the start parameters
    ACTS_VERBOSE("Run Gsf with start parameters: \n" << sParameters);

    auto intersectionStatusStartSurface =
        sParameters.referenceSurface()
            .intersect(GeometryContext{},
                       sParameters.position(GeometryContext{}),
                       sParameters.unitDirection(), true)
            .intersection.status;

    if (intersectionStatusStartSurface != Intersection3D::Status::onSurface) {
      ACTS_ERROR(
          "Surface intersection of start parameters with bound-check failed");
      m_failStatistics.write("StartParametersNotOnStartSurface"s, myNAN, myNAN,
                             myNAN, myNAN, sParameters.absoluteMomentum(),
                             sParameters.transverseMomentum(),
                             sParameters.template get<eBoundTheta>());
      return GsfError::StartParametersNotOnStartSurface;
    }

    // To be able to find measurements later, we put them into a map
    // We need to copy input SourceLinks anyways, so the map can own them.
    ACTS_VERBOSE("Preparing " << std::distance(begin, end)
                              << " input measurements");
    std::map<GeometryIdentifier, std::reference_wrapper<const SourceLink>>
        inputMeasurements;
    for (auto it = begin; it != end; ++it) {
      inputMeasurements.emplace(it->get().geometryId(), *it);
    }

    ACTS_VERBOSE(
        "Gsf: Final measuerement map size: " << inputMeasurements.size());
    throw_assert(sParameters.covariance() != std::nullopt,
                 "we need a covariance here...");

    using ThisGsfActor = GsfActor<GainMatrixUpdater, outlier_finder_t,
                                  calibrator_t, GainMatrixSmoother>;

    /////////////////
    // Forward pass
    /////////////////
    ACTS_VERBOSE("+-----------------------------+");
    ACTS_VERBOSE("| Gsf: Do forward propagation |");
    ACTS_VERBOSE("+-----------------------------+");

    auto fwdResult = [&]() {
      MultiComponentBoundTrackParameters<SinglyCharged> params(
          sParameters.referenceSurface().getSharedPtr(),
          sParameters.parameters(), sParameters.covariance());

      auto fwdPropOptions = fwdPropInitializer(options, logger);

      // Catch the actor and set the measurements
      auto& actor = fwdPropOptions.actionList.template get<ThisGsfActor>();
      actor.m_cfg.inputMeasurements = inputMeasurements;
      actor.m_cfg.maxComponents = options.maxComponents;
      actor.m_calibrator = options.calibrator;
      actor.m_outlierFinder = options.outlierFinder;
      actor.m_cfg.abortOnError = options.abortOnError;
      actor.m_cfg.applyMaterialEffects = options.applyMaterialEffects;

      fwdPropOptions.direction = Acts::forward;

      return m_propagator.propagate(params, fwdPropOptions);
    }();

    if (!fwdResult.ok()) {
      m_failStatistics.write(fwdResult.error().message(), myNAN, myNAN, myNAN,
                             myNAN, sParameters.absoluteMomentum(),
                             sParameters.transverseMomentum(),
                             sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(fwdResult.error());
    }

    auto& fwdGsfResult = (*fwdResult).template get<GsfResult>();

    if (!fwdGsfResult.result.ok()) {
      m_failStatistics.write(fwdGsfResult.result.error().message(),
                             (*fwdResult).steps, (*fwdResult).pathLength, myNAN,
                             myNAN, sParameters.absoluteMomentum(),
                             sParameters.transverseMomentum(),
                             sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(fwdGsfResult.result.error());
    }

    if (fwdGsfResult.processedStates == 0) {
      m_failStatistics.write("noProcessedStates"s, (*fwdResult).steps,
                             (*fwdResult).pathLength, myNAN, myNAN,
                             sParameters.absoluteMomentum(),
                             sParameters.transverseMomentum(),
                             sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(GsfError::NoStatesCreated);
    }

    ACTS_VERBOSE("Finished forward propagation");
    ACTS_VERBOSE("- visited surfaces: " << fwdGsfResult.visitedSurfaces.size());
    ACTS_VERBOSE("- processed states: " << fwdGsfResult.processedStates);
    ACTS_VERBOSE("- measuerement states: " << fwdGsfResult.measurementStates);

    //////////////////
    // Backward pass
    //////////////////
    ACTS_VERBOSE("+------------------------------+");
    ACTS_VERBOSE("| Gsf: Do backward propagation |");
    ACTS_VERBOSE("+------------------------------+");

    auto bwdResult = [&]() {
      // Use last forward state as start parameters for backward propagation
      const auto params = detail::extractMultiComponentState(
          fwdGsfResult.fittedStates, fwdGsfResult.currentTips,
          fwdGsfResult.weightsOfStates, detail::StatesType::eFiltered);

      auto bwdPropOptions = bwdPropInitializer(options, logger);

      auto& actor = bwdPropOptions.actionList.template get<ThisGsfActor>();
      actor.m_cfg.inputMeasurements = inputMeasurements;
      actor.m_cfg.maxComponents = options.maxComponents;
      actor.m_calibrator = options.calibrator;
      actor.m_outlierFinder = options.outlierFinder;
      actor.m_cfg.abortOnError = options.abortOnError;
      actor.m_cfg.applyMaterialEffects = options.applyMaterialEffects;

      // Workaround to get the first state into the MultiTrajectory seems also
      // to be necessary for standard navigator to prevent double kalman
      // update on the last surface
      actor.m_cfg.multiTrajectoryInitializer = [&fwdGsfResult](
                                                   auto& result,
                                                   const auto& logger) {
        result.currentTips.clear();

        ACTS_VERBOSE(
            "Initialize the MultiTrajectory with information provided to the "
            "Actor");

        for (const auto idx : fwdGsfResult.currentTips) {
          result.currentTips.push_back(
              result.fittedStates.addTrackState(TrackStatePropMask::All));
          auto proxy =
              result.fittedStates.getTrackState(result.currentTips.back());
          proxy.copyFrom(fwdGsfResult.fittedStates.getTrackState(idx));
          result.weightsOfStates[result.currentTips.back()] =
              fwdGsfResult.weightsOfStates.at(idx);

          // Because we are backwards, we use forward filtered as predicted
          proxy.predicted() = proxy.filtered();
          proxy.predictedCovariance() = proxy.filteredCovariance();

          // Mark surface as visited
          result.visitedSurfaces.insert(proxy.referenceSurface().geometryId());
        }
      };

      bwdPropOptions.direction = Acts::backward;

      // const auto targetSurface = [&]() {
      //   const Surface* ts = nullptr;
      //   fwdGsfResult.fittedStates.visitBackwards(
      //       fwdGsfResult.currentTips.front(),
      //       [&ts](const auto& state) { ts = &state.referenceSurface();
      //       });
      //   throw_assert(ts, "target surface must not be nullptr");
      //   return ts;
      // }();

      // TODO somehow this proagation fails if we target the first
      // measuerement surface, go instead back to beamline for now
      return m_propagator
          .template propagate<decltype(params), decltype(bwdPropOptions),
                              MultiStepperSurfaceReached>(
              params, *options.referenceSurface, bwdPropOptions);
    }();

    if (!bwdResult.ok()) {
      m_failStatistics.write(bwdResult.error().message(), (*fwdResult).steps,
                             (*fwdResult).pathLength, myNAN, myNAN,
                             sParameters.absoluteMomentum(),
                             sParameters.transverseMomentum(),
                             sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(bwdResult.error());
    }

    auto& bwdGsfResult = (*bwdResult).template get<GsfResult>();

    if (!bwdGsfResult.result.ok()) {
      m_failStatistics.write(
          bwdGsfResult.result.error().message(), (*fwdResult).steps,
          (*fwdResult).pathLength, (*bwdResult).steps, (*bwdResult).pathLength,
          sParameters.absoluteMomentum(), sParameters.transverseMomentum(),
          sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(bwdGsfResult.result.error());
    }

    if (bwdGsfResult.processedStates == 0) {
      m_failStatistics.write(
          "noProcessedStates"s, (*fwdResult).steps, (*fwdResult).pathLength,
          (*bwdResult).steps, (*bwdResult).pathLength,
          sParameters.absoluteMomentum(), sParameters.transverseMomentum(),
          sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(GsfError::NoStatesCreated);
    }

    ////////////////////////////////////
    // Smooth and create Kalman Result
    ////////////////////////////////////
    ACTS_VERBOSE("Gsf: Do smoothing");
    ACTS_VERBOSE(
        "- Fwd measurement states: " << fwdGsfResult.measurementStates);
    ACTS_VERBOSE(
        "- Bwd measurement states: " << bwdGsfResult.measurementStates);

    const auto smoothResult = detail::smoothAndCombineTrajectories<true>(
        fwdGsfResult.fittedStates, fwdGsfResult.currentTips,
        fwdGsfResult.weightsOfStates, bwdGsfResult.fittedStates,
        bwdGsfResult.currentTips, bwdGsfResult.weightsOfStates, logger);

    // Cannot use structured binding since they cannot be captured in lambda
    const auto& combinedTraj = std::get<0>(smoothResult);
    const auto lastTip = std::get<1>(smoothResult);

    // Some test
    if (lastTip == SIZE_MAX) {
      m_failStatistics.write(
          "NoStatesCreated", (*fwdResult).steps, (*fwdResult).pathLength,
          (*bwdResult).steps, (*bwdResult).pathLength,
          sParameters.absoluteMomentum(), sParameters.transverseMomentum(),
          sParameters.template get<eBoundTheta>());
      return GsfError::NoStatesCreated;
    }

    Acts::KalmanFitterResult kalmanResult;
    kalmanResult.lastTrackIndex = lastTip;
    kalmanResult.fittedStates = std::move(combinedTraj);
    kalmanResult.smoothed = true;
    kalmanResult.reversed = true;
    kalmanResult.finished = true;
    kalmanResult.lastMeasurementIndex = lastTip;

    ///////////////////////////////////////////////////////
    // Propagate back to origin with smoothed parameters //
    ///////////////////////////////////////////////////////
    ACTS_VERBOSE("+--------------------------------------+");
    ACTS_VERBOSE("| Gsf: Do propagation back to beamline |");
    ACTS_VERBOSE("+--------------------------------------+");
    auto lastResult = [&]() -> Result<std::unique_ptr<BoundTrackParameters>> {
      const auto& [surface, lastSmoothedState] =
          std::get<2>(smoothResult).front();

      throw_assert(
          detail::weightsAreNormalized(
              lastSmoothedState,
              [](const auto& tuple) { return std::get<double>(tuple); }),
          "");

      const MultiComponentBoundTrackParameters<SinglyCharged> params(
          surface->getSharedPtr(), lastSmoothedState);

      auto lastPropOptions = bwdPropInitializer(options, logger);

      auto& actor = lastPropOptions.actionList.template get<ThisGsfActor>();
      actor.m_cfg.maxComponents = options.maxComponents;
      actor.m_cfg.abortOnError = options.abortOnError;
      actor.m_cfg.applyMaterialEffects = options.applyMaterialEffects;
      actor.m_cfg.surfacesToSkip.insert(surface->geometryId());

      lastPropOptions.direction = Acts::backward;

      auto result =
          m_propagator
              .template propagate<decltype(params), decltype(lastPropOptions),
                                  MultiStepperSurfaceReached>(
                  params, *options.referenceSurface, lastPropOptions);

      if (!result.ok()) {
        return result.error();
      } else {
        return std::move((*result).endParameters);
      }
    }();

    if (!lastResult.ok()) {
      RETURN_ERROR_OR_ABORT_FIT(lastResult.error());
    }

    kalmanResult.fittedParameters = **lastResult;

    m_successStatistics.write((*fwdResult).steps, (*fwdResult).pathLength,
                              (*bwdResult).steps, (*bwdResult).pathLength,
                              sParameters.absoluteMomentum(),
                              sParameters.transverseMomentum(),
                              sParameters.template get<eBoundTheta>());

    return kalmanResult;
  }
};

}  // namespace Acts
