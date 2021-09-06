// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/MultiComponentBoundTrackParameters.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/MultiTrajectoryHelpers.hpp"
#include "Acts/EventData/SourceLinkConcept.hpp"
#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/Material/ISurfaceMaterial.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/TrackFitting/GainMatrixSmoother.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/TrackFitting/KalmanFitter.hpp"
#include "Acts/TrackFitting/KalmanFitterError.hpp"

#include <map>
#include <numeric>

#include "BetheHeitlerApprox.hpp"
#include "GsfError.hpp"
#include "GsfUtils.hpp"
#include "KLMixtureReduction.hpp"
#include "MultiSteppingLogger.hpp"

#define RETURN_ERROR_OR_ABORT(error) \
  if (m_config.abortOnError) {       \
    std::abort();                    \
  } else {                           \
    return error;                    \
  }

#define RETURN_OR_ABORT(error) \
  if (m_config.abortOnError) { \
    std::abort();              \
  } else {                     \
    result.result = error;     \
    return;                    \
  }

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

  bool throwOnError = true;

  std::size_t maxComponents = 4;

  bool multiComponentPropagationToPerigee = true;
};

template <typename source_link_t>
struct GsfResult {
  /// The multi-trajectory which stores the graph of components
  MultiTrajectory<source_link_t> fittedStates;
  std::map<size_t, ActsScalar> weightsOfStates;

  /// The current indexes for the newest components in the multi trajectory
  std::vector<std::size_t> currentTips;

  /// The indices of the parent states in the multi trajectory. This is needed
  /// since there can be cases with splitting but no Kalman-Update, so we need
  /// to preserve this information
  std::vector<std::size_t> parentTips;

  /// The number of measurement states created
  std::size_t measurementStates = 0;
  std::size_t processedStates = 0;

  // Propagate potential errors to the outside
  Result<void> result{Result<void>::success()};

  // Used for workaround to initialize MT correctly
  bool haveInitializedMT = false;
};

template <typename propagator_t>
struct GaussianSumFitter {
  GaussianSumFitter(propagator_t propagator)
      : m_propagator(std::move(propagator)) {}

  /// The navigator type
  using GsfNavigator = typename propagator_t::Navigator;

  template <typename source_link_t, typename updater_t,
            typename outlier_finder_t, typename calibrator_t,
            typename smoother_t>
  struct Actor {
    /// Enforce default construction
    Actor() = default;

    /// Broadcast the result_type
    using result_type = GsfResult<source_link_t>;

    // Actor configuration
    struct Config {
      /// Maximum number of components which the GSF should handle
      std::size_t maxComponents = 16;

      /// Input measurements
      std::map<GeometryIdentifier, source_link_t> inputMeasurements;

      /// Bethe Heitler Approximator
      detail::BHApprox bethe_heitler_approx =
          detail::BHApprox(detail::bh_cmps6_order5_data);

      /// Number of processed states
      std::size_t processedStates = 0;

      /// Wether to transport covariance
      bool doCovTransport = true;

      /// Whether to consider multiple scattering.
      bool multipleScattering = true;

      /// Whether to consider energy loss.
      bool energyLoss = true;

      /// Whether to abort immediately when an error occurs
      bool abortOnError = false;

      /// A not so nice workaround to get the first backward state in the
      /// MultiTrajectory for the DirectNavigator
      std::function<void(result_type&, const LoggerWrapper&)>
          multiTrajectoryInitializer;
    } m_config;

    /// Configurable components:
    updater_t m_updater;
    outlier_finder_t m_outlierFinder;
    calibrator_t m_calibrator;
    smoother_t m_smoother;

    SurfaceReached m_targetReachedAborter;

    /// Broadcast Cache Type
    using TrackProxy = typename MultiTrajectory<source_link_t>::TrackStateProxy;
    using ComponentCache =
        std::tuple<std::variant<detail::GsfComponentParameterCache, TrackProxy>,
                   detail::GsfComponentMetaCache>;

    struct ParametersCacheProjector {
      auto& operator()(ComponentCache& cache) const {
        return std::get<detail::GsfComponentParameterCache>(std::get<0>(cache));
      }
      const auto& operator()(const ComponentCache& cache) const {
        return std::get<detail::GsfComponentParameterCache>(std::get<0>(cache));
      }
    };

    /// @brief GSF actor operation
    ///
    /// @tparam propagator_state_t is the type of Propagagor state
    /// @tparam stepper_t Type of the stepper
    ///
    /// @param state is the mutable propagator state object
    /// @param stepper The stepper in use
    /// @param result is the mutable result state object
    template <typename propagator_state_t, typename stepper_t>
    void operator()(propagator_state_t& state, const stepper_t& stepper,
                    result_type& result) const {
      const auto& logger = state.options.logger;

      // Some initial printing
      ACTS_VERBOSE("Gsf step at mean position "
                   << stepper.position(state.stepping).transpose()
                   << " with direction "
                   << stepper.direction(state.stepping).transpose());
      ACTS_VERBOSE(
          "Propagation is in "
          << (state.stepping.navDir == forward ? "forward" : "backward")
          << " mode");

      for (auto i = 0ul; i < stepper.numberComponents(state.stepping); ++i) {
        typename stepper_t::ComponentProxy cmp(state.stepping, i);

        ACTS_VERBOSE(
            "  #" << i << " pos: "
                  << cmp.pars().template segment<3>(eFreePos0).transpose()
                  << ", dir: "
                  << cmp.pars().template segment<3>(eFreeDir0).transpose());
      }

      // Workaround to initialize MT in backward mode
      if (!result.haveInitializedMT && m_config.multiTrajectoryInitializer) {
        m_config.multiTrajectoryInitializer(result, logger);
      }

      // Initialize current tips on first pass
      if (result.parentTips.empty()) {
        result.parentTips.resize(stepper.numberComponents(state.stepping),
                                 SIZE_MAX);
      }

      // Check if we have the right number of components
      if (result.parentTips.size() !=
          stepper.numberComponents(state.stepping)) {
        ACTS_ERROR("component number mismatch:"
                   << result.parentTips.size() << " vs "
                   << stepper.numberComponents(state.stepping));

        RETURN_OR_ABORT(GsfError::ComponentNumberMismatch);
      }

      // We only need to do something if we are on a surface
      if (state.navigation.currentSurface) {
        const auto& surface = *state.navigation.currentSurface;
        ACTS_VERBOSE("Step is at surface " << surface.geometryId());

        // Check what we have on this surface
        const auto found_source_link =
            m_config.inputMeasurements.find(surface.geometryId());
        const bool haveMaterial =
            state.navigation.currentSurface->surfaceMaterial();
        const bool haveMeasurement =
            found_source_link != m_config.inputMeasurements.end();

        // Early return if nothing happens
        if (not haveMaterial && not haveMeasurement) {
          return;
        }

        // Create Cache
        thread_local std::vector<ComponentCache> componentCache;
        componentCache.clear();

        // Projectors
        auto mapToProxyAndWeight = [&](auto& cmp) {
          auto& proxy = std::get<TrackProxy>(std::get<0>(cmp));
          return std::tie(proxy, result.weightsOfStates.at(proxy.index()));
        };

        auto mapProxyToWeightParsCov = [&](const auto& variant) {
          auto& proxy = std::get<TrackProxy>(variant);
          return std::make_tuple(result.weightsOfStates.at(proxy.index()),
                                 proxy.filtered(), proxy.filteredCovariance());
        };

        ///////////////////////////////////////////
        // Component Splitting AND Kalman Update
        ///////////////////////////////////////////
        if (haveMaterial && haveMeasurement) {
          ACTS_VERBOSE("Material and measurement");
          detail::extractComponents(
              state, stepper, result.parentTips,
              detail::ComponentSplitter{m_config.bethe_heitler_approx},
              m_config.doCovTransport, componentCache);

          ACTS_VERBOSE("reduce component number...");
          detail::reduceWithKLDistance(
              componentCache,
              std::min(static_cast<std::size_t>(stepper.maxComponents),
                       m_config.maxComponents),
              ParametersCacheProjector{});

          ACTS_VERBOSE("kalman update...");
          kalmanUpdate(state, found_source_link->second, result,
                       componentCache);
          result.parentTips = result.currentTips;

          ACTS_VERBOSE("reweight components...");
          detail::reweightComponents(componentCache, mapToProxyAndWeight);

          ACTS_VERBOSE("update stepper...");
          detail::updateStepper(state, stepper, componentCache,
                                mapProxyToWeightParsCov);
        }
        /////////////////////////////////////////////
        // Component Splitting BUT NO Kalman Update
        /////////////////////////////////////////////
        else if (haveMaterial && not haveMeasurement) {
          ACTS_VERBOSE("Only Material");
          detail::extractComponents(
              state, stepper, result.parentTips,
              detail::ComponentSplitter{m_config.bethe_heitler_approx},
              m_config.doCovTransport, componentCache);

          detail::reduceWithKLDistance(
              componentCache,
              std::min(static_cast<std::size_t>(stepper.maxComponents),
                       m_config.maxComponents),
              ParametersCacheProjector{});

          result.parentTips.clear();
          for (const auto& [variant, meta] : componentCache) {
            result.parentTips.push_back(meta.parentIndex);
          }

          detail::updateStepper(
              state, stepper, componentCache, [&](const auto& variant) {
                return std::get<detail::GsfComponentParameterCache>(variant);
              });
        }
        /////////////////////////////////////////////
        // Kalman Update BUT NO Component Splitting
        /////////////////////////////////////////////
        else if (not haveMaterial && haveMeasurement) {
          ACTS_VERBOSE("Only measurement");
          detail::extractComponents(state, stepper, result.parentTips,
                                    detail::ComponentForwarder{},
                                    m_config.doCovTransport, componentCache);

          kalmanUpdate(state, found_source_link->second, result,
                       componentCache);
          result.parentTips = result.currentTips;

          detail::reweightComponents(componentCache, mapToProxyAndWeight);

          detail::updateStepper(state, stepper, componentCache,
                                mapProxyToWeightParsCov);
        }
      }
    }

    template <typename propagator_state_t>
    Result<void> kalmanUpdate(const propagator_state_t& state,
                              const source_link_t& source_link,
                              result_type& result,
                              std::vector<ComponentCache>& components) const {
      const auto& logger = state.options.logger;
      const auto& surface = *state.navigation.currentSurface;
      result.currentTips.clear();

      for (auto& [variant, meta] : components) {
        // Create new track state
        result.currentTips.push_back(result.fittedStates.addTrackState(
            TrackStatePropMask::All, meta.parentIndex));

        const auto [weight, pars, cov] =
            std::get<detail::GsfComponentParameterCache>(variant);

        variant = result.fittedStates.getTrackState(result.currentTips.back());
        auto& trackProxy = std::get<TrackProxy>(variant);

        // Set track parameters
        trackProxy.predicted() = std::move(pars);
        if (cov) {
          trackProxy.predictedCovariance() = std::move(*cov);
        }
        result.weightsOfStates[result.currentTips.back()] = weight;

        // Set surface
        trackProxy.setReferenceSurface(surface.getSharedPtr());

        // assign the source link to the track state
        trackProxy.uncalibrated() = source_link;

        trackProxy.jacobian() = std::move(meta.jacobian);
        trackProxy.pathLength() = std::move(meta.pathLength);

        // We have predicted parameters, so calibrate the uncalibrated
        std::visit(
            [&](const auto& calibrated) {
              trackProxy.setCalibrated(calibrated);
            },
            m_calibrator(trackProxy.uncalibrated(), trackProxy.predicted()));

        // Get and set the type flags
        trackProxy.typeFlags().set(TrackStateFlag::ParameterFlag);
        if (surface.surfaceMaterial() != nullptr) {
          trackProxy.typeFlags().set(TrackStateFlag::MaterialFlag);
        }

        // Do Kalman update
        if (not m_outlierFinder(trackProxy)) {
          // Perform update
          auto updateRes = m_updater(state.geoContext, trackProxy,
                                     state.stepping.navDir, logger);

          if (!updateRes.ok()) {
            ACTS_ERROR("Update step failed: " << updateRes.error());
            RETURN_ERROR_OR_ABORT(updateRes.error());
          }

          trackProxy.typeFlags().set(TrackStateFlag::MeasurementFlag);

          // We count the state with measurement
          ++result.measurementStates;
        } else {
          // TODO ACTS_VERBOSE message
          throw std::runtime_error(
              "outlier handling not yet implemented fully");

          // Set the outlier type flag
          trackProxy.typeFlags().set(TrackStateFlag::OutlierFlag);
          trackProxy.data().ifiltered = trackProxy.data().ipredicted;
        }
      }

      ++result.processedStates;

      return Acts::Result<void>::success();
    }
  };

  struct FinalizePositionPrinter {
    using result_type = char;

    template <typename propagator_state_t, typename stepper_t>
    void operator()(propagator_state_t& state, const stepper_t& stepper,
                    result_type&) const {
      const auto& logger = state.options.logger;

      ACTS_VERBOSE("Finalizer step at mean position "
                   << stepper.position(state.stepping).transpose()
                   << " with direction "
                   << stepper.direction(state.stepping).transpose());
    }
  };

  /// The propagator instance used by the fit function
  propagator_t m_propagator;

  /// @brief The fit function for the Direct navigator
  template <typename source_link_t, typename start_parameters_t,
            typename calibrator_t, typename outlier_finder_t>
  Acts::Result<Acts::KalmanFitterResult<source_link_t>> fit(
      const std::vector<source_link_t>& sourcelinks,
      const start_parameters_t& sParameters,
      const GsfOptions<calibrator_t, outlier_finder_t>& options,
      const std::vector<const Surface*>& sSequence) const {
    static_assert(SourceLinkConcept<source_link_t>,
                  "Source link does not fulfill SourceLinkConcept");
    static_assert(
        std::is_same_v<DirectNavigator, typename propagator_t::Navigator>);

    // The logger
    const auto& logger = options.logger;

    // To be able to find measurements later, we put them into a map
    // We need to copy input SourceLinks anyways, so the map can own them.
    ACTS_VERBOSE("Preparing " << sourcelinks.size() << " input measurements");
    std::map<GeometryIdentifier, source_link_t> inputMeasurements;
    for (const auto& sl : sourcelinks) {
      inputMeasurements.emplace(sl.geometryId(), sl);
    }

    ACTS_VERBOSE("Run Gsf with start parameters: \n" << sParameters);
    ACTS_VERBOSE("Gsf: Number of surfaces in sSequence: " << sSequence.size());
    ACTS_VERBOSE(
        "Gsf: Final measuerement map size: " << inputMeasurements.size());
    throw_assert(sParameters.covariance() != std::nullopt,
                 "we need a covariance here...");

    ////////////////////
    // Common Settings
    ////////////////////

    // Create the ActionList and AbortList
    using GSFActor = Actor<source_link_t, GainMatrixUpdater, outlier_finder_t,
                           calibrator_t, GainMatrixSmoother>;

    using Actors = ActionList<DirectNavigator::Initializer, GSFActor>;
    using Aborters = AbortList<>;

    // Create relevant options for the propagation options
    PropagatorOptions<Actors, Aborters> propOptions(
        options.geoContext, options.magFieldContext, logger);

    // Catch the actor and set the measurements
    auto& actor = propOptions.actionList.template get<GSFActor>();
    actor.m_config.inputMeasurements = inputMeasurements;
    actor.m_config.maxComponents = options.maxComponents;
    actor.m_calibrator = options.calibrator;
    actor.m_outlierFinder = options.outlierFinder;
    actor.m_config.abortOnError = options.throwOnError;

    // TODO This seems so solve some issues in the navigation
    //     propOptions.tolerance = 1e-5;

    /////////////////
    // Forward pass
    /////////////////
    ACTS_VERBOSE("+-----------------------------+");
    ACTS_VERBOSE("| Gsf: Do forward propagation |");
    ACTS_VERBOSE("+-----------------------------+");
    auto fwdResult = [&]() -> Result<GsfResult<source_link_t>> {
      MultiComponentBoundTrackParameters<SinglyCharged> params(
          sParameters.referenceSurface().getSharedPtr(),
          sParameters.parameters(), sParameters.covariance());

      propOptions.direction = Acts::forward;
      propOptions.actionList.template get<DirectNavigator::Initializer>()
          .navSurfaces = sSequence;

      auto propResult = m_propagator.propagate(params, propOptions);

      if (!propResult.ok()) {
        return propResult.error();
      }

      GsfResult gsfResult =
          (*propResult).template get<GsfResult<source_link_t>>();

      if (!gsfResult.result.ok()) {
        return gsfResult.result.error();
      }

      if (gsfResult.processedStates == 0) {
        return GsfError::NoStatesCreated;
      }

      return gsfResult;
    }();

    if (!fwdResult.ok()) {
      return fwdResult.error();
    }

    const auto& fwdGsfResult = *fwdResult;

    //////////////////
    // Backward pass
    //////////////////

    ACTS_VERBOSE("+------------------------------+");
    ACTS_VERBOSE("| Gsf: Do backward propagation |");
    ACTS_VERBOSE("+------------------------------+");
    auto bwdResult = [&]() -> Result<GsfResult<source_link_t>> {
      const auto params = detail::extractMultiComponentState(
          fwdGsfResult.fittedStates, fwdGsfResult.currentTips,
          fwdGsfResult.weightsOfStates, detail::StatesType::eFiltered);

      propOptions.direction = Acts::backward;

      // TODO here we start with the second last surface, so we don't get
      // navigation problems. this should get fixed
      std::vector<const Surface*> backwardSequence(
          std::next(sSequence.rbegin()), sSequence.rend());
      propOptions.actionList.template get<DirectNavigator::Initializer>()
          .navSurfaces = std::move(backwardSequence);

      // Workaround to get the first state into the MultiTrajectory
      actor.m_config.multiTrajectoryInitializer = [&](auto& result,
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
        }

        result.haveInitializedMT = true;
      };

      auto propResult = m_propagator.propagate(params, propOptions);

      if (!propResult.ok()) {
        return propResult.error();
      }

      GsfResult gsfResult =
          (*propResult).template get<GsfResult<source_link_t>>();

      if (!gsfResult.result.ok()) {
        return gsfResult.result.error();
      }

      if (gsfResult.processedStates == 0) {
        return GsfError::NoStatesCreated;
      }

      return gsfResult;
    }();

    if (!bwdResult.ok()) {
      return bwdResult.error();
    }

    const auto& bwdGsfResult = *bwdResult;

    //////////////////////////////
    // Last part towards perigee
    //////////////////////////////
    ACTS_VERBOSE("+-------------------+");
    ACTS_VERBOSE("| Gsf: Do Last Part |");
    ACTS_VERBOSE("+-------------------+");

    const auto lastMultiPars = [&]() {
      if (options.multiComponentPropagationToPerigee) {
        return detail::multiTrajectoryToMultiComponentParameters(
            bwdGsfResult.currentTips, bwdGsfResult.fittedStates,
            bwdGsfResult.weightsOfStates, detail::StatesType::ePredicted);
      } else {
        using Projector =
            detail::MultiTrajectoryProjector<detail::StatesType::ePredicted,
                                             source_link_t>;
        const auto [lastPars, lastCov] = detail::combineComponentRange(
            bwdGsfResult.currentTips.begin(), bwdGsfResult.currentTips.end(),
            Projector{bwdGsfResult.fittedStates, bwdGsfResult.weightsOfStates});

        const auto& surface =
            bwdGsfResult.fittedStates
                .getTrackState(bwdGsfResult.currentTips.front())
                .referenceSurface();

        return MultiComponentBoundTrackParameters<SinglyCharged>(
            surface.getSharedPtr(), lastPars, lastCov);
      }
    }();

    PropagatorOptions<
        Acts::ActionList<DirectNavigator::Initializer, FinalizePositionPrinter>,
        Aborters>
        lastPropOptions(options.geoContext, options.magFieldContext, logger);

    lastPropOptions.actionList.template get<DirectNavigator::Initializer>()
        .navSurfaces = {options.referenceSurface};
    lastPropOptions.direction = NavigationDirection::backward;
    lastPropOptions.targetTolerance *= 1000.0;

    auto lastPropRes = m_propagator.propagate(
        lastMultiPars, *options.referenceSurface, lastPropOptions);

    if (!lastPropRes.ok()) {
      return lastPropRes.error();
    }

    ////////////////////////////////////
    // Smooth and create Kalman Result
    ////////////////////////////////////

    ACTS_VERBOSE("Gsf: Do smoothing");

    const auto [combinedTraj, lastTip] = detail::smoothAndCombineTrajectories(
        fwdGsfResult.fittedStates, fwdGsfResult.currentTips,
        fwdGsfResult.weightsOfStates, bwdGsfResult.fittedStates,
        bwdGsfResult.currentTips, bwdGsfResult.weightsOfStates);

    Acts::KalmanFitterResult<source_link_t> kalmanResult;
    kalmanResult.lastTrackIndex = lastTip;
    kalmanResult.fittedStates = std::move(combinedTraj);
    kalmanResult.smoothed = true;
    kalmanResult.reversed = true;
    kalmanResult.finished = true;
    kalmanResult.lastMeasurementIndex = lastTip;
    kalmanResult.fittedParameters = *((*lastPropRes).endParameters);

    return kalmanResult;
  }

  /// Workaround to let navigation abort on surface
  struct SurfaceAborter {
    const Surface* target;
    template <typename propagator_state_t, typename stepper_t>
    bool operator()(propagator_state_t& state, const stepper_t&) const {
      const auto& logger = state.options.logger;
      if (state.navigation.currentSurface == target) {
        ACTS_VERBOSE("Surface aborter does its job");
        return true;
      }
      return false;
    }
  };

  /// @brief The fit function for the standard navigator
  template <typename source_link_t, typename start_parameters_t,
            typename calibrator_t, typename outlier_finder_t>
  Acts::Result<Acts::KalmanFitterResult<source_link_t>> fit(
      const std::vector<source_link_t>& sourcelinks,
      const start_parameters_t& sParameters,
      const GsfOptions<calibrator_t, outlier_finder_t>& options) const {
    static_assert(SourceLinkConcept<source_link_t>,
                  "Source link does not fulfill SourceLinkConcept");
    static_assert(std::is_same_v<Navigator, typename propagator_t::Navigator>);

    // The logger
    const auto& logger = options.logger;

    // To be able to find measurements later, we put them into a map
    // We need to copy input SourceLinks anyways, so the map can own them.
    ACTS_VERBOSE("Preparing " << sourcelinks.size() << " input measurements");
    std::map<GeometryIdentifier, source_link_t> inputMeasurements;
    for (const auto& sl : sourcelinks) {
      inputMeasurements.emplace(sl.geometryId(), sl);
    }

    ACTS_VERBOSE("Run Gsf with start parameters: \n" << sParameters);
    ACTS_VERBOSE(
        "Gsf: Final measuerement map size: " << inputMeasurements.size());
    throw_assert(sParameters.covariance() != std::nullopt,
                 "we need a covariance here...");

    ////////////////////
    // Common Settings
    ////////////////////

    // Create the ActionList and AbortList
    using GSFActor = Actor<source_link_t, GainMatrixUpdater, outlier_finder_t,
                           calibrator_t, GainMatrixSmoother>;

    using Actors = ActionList<GSFActor>;
    using Aborters = AbortList<EndOfWorldReached>;

    // Create relevant options for the propagation options
    PropagatorOptions<Actors, Aborters> propOptions(
        options.geoContext, options.magFieldContext, logger);

    // Catch the actor and set the measurements
    auto& actor = propOptions.actionList.template get<GSFActor>();
    actor.m_config.inputMeasurements = inputMeasurements;
    actor.m_config.maxComponents = options.maxComponents;
    actor.m_calibrator = options.calibrator;
    actor.m_outlierFinder = options.outlierFinder;
    actor.m_config.abortOnError = options.throwOnError;

    // TODO This seems so solve some issues in the navigation
    //     propOptions.tolerance = 1e-5;

    /////////////////
    // Forward pass
    /////////////////
    ACTS_VERBOSE("+-----------------------------+");
    ACTS_VERBOSE("| Gsf: Do forward propagation |");
    ACTS_VERBOSE("+-----------------------------+");
    auto fwdResult = [&]() -> Result<GsfResult<source_link_t>> {
      MultiComponentBoundTrackParameters<SinglyCharged> params(
          sParameters.referenceSurface().getSharedPtr(),
          sParameters.parameters(), sParameters.covariance());

      propOptions.direction = Acts::forward;

      auto propResult = m_propagator.propagate(params, propOptions);

      if (!propResult.ok()) {
        return propResult.error();
      }

      GsfResult gsfResult =
          (*propResult).template get<GsfResult<source_link_t>>();

      if (!gsfResult.result.ok()) {
        return gsfResult.result.error();
      }

      if (gsfResult.processedStates == 0) {
        return GsfError::NoStatesCreated;
      }

      return gsfResult;
    }();

    if (!fwdResult.ok()) {
      return fwdResult.error();
    }

    const auto& fwdGsfResult = *fwdResult;

    //////////////////
    // Backward pass
    //////////////////

    ACTS_VERBOSE("+------------------------------+");
    ACTS_VERBOSE("| Gsf: Do backward propagation |");
    ACTS_VERBOSE("+------------------------------+");
    auto bwdResult = [&]() -> Result<GsfResult<source_link_t>> {
      const auto params = detail::extractMultiComponentState(
          fwdGsfResult.fittedStates, fwdGsfResult.currentTips,
          fwdGsfResult.weightsOfStates, detail::StatesType::eFiltered);

      const Surface* backwardTarget = nullptr;
      fwdGsfResult.fittedStates.visitBackwards(
          fwdGsfResult.currentTips.front(), [&](const auto& state) {
            backwardTarget = &state.referenceSurface();
          });

      using BwdActors = ActionList<GSFActor>;
      using BwdAborters = AbortList<SurfaceAborter>;

      PropagatorOptions<BwdActors, BwdAborters> bwdPropOptions(
          options.geoContext, options.magFieldContext, logger);

      bwdPropOptions.abortList.template get<SurfaceAborter>().target = backwardTarget;
      
      auto& actor = bwdPropOptions.actionList.template get<GSFActor>();
      actor.m_config.inputMeasurements = inputMeasurements;
      actor.m_config.maxComponents = options.maxComponents;
      actor.m_calibrator = options.calibrator;
      actor.m_outlierFinder = options.outlierFinder;
      actor.m_config.abortOnError = options.throwOnError;

      bwdPropOptions.direction = Acts::backward;

      ACTS_VERBOSE("Backward propagation with target surface "
                   << backwardTarget->geometryId());
      auto propResult = m_propagator.propagate(params, bwdPropOptions);

      if (!propResult.ok()) {
        return propResult.error();
      }

      GsfResult gsfResult =
          (*propResult).template get<GsfResult<source_link_t>>();

      if (!gsfResult.result.ok()) {
        return gsfResult.result.error();
      }

      if (gsfResult.processedStates == 0) {
        return GsfError::NoStatesCreated;
      }

      return gsfResult;
    }();

    if (!bwdResult.ok()) {
      return bwdResult.error();
    }

    const auto& bwdGsfResult = *bwdResult;

    //////////////////////////////
    // Last part towards perigee
    //////////////////////////////

    const auto lastMultiPars = [&]() {
      if (options.multiComponentPropagationToPerigee) {
        return detail::multiTrajectoryToMultiComponentParameters(
            bwdGsfResult.currentTips, bwdGsfResult.fittedStates,
            bwdGsfResult.weightsOfStates, detail::StatesType::ePredicted);
      } else {
        using Projector =
            detail::MultiTrajectoryProjector<detail::StatesType::ePredicted,
                                             source_link_t>;
        const auto [lastPars, lastCov] = detail::combineComponentRange(
            bwdGsfResult.currentTips.begin(), bwdGsfResult.currentTips.end(),
            Projector{bwdGsfResult.fittedStates, bwdGsfResult.weightsOfStates});

        const auto& surface =
            bwdGsfResult.fittedStates
                .getTrackState(bwdGsfResult.currentTips.front())
                .referenceSurface();

        return MultiComponentBoundTrackParameters<SinglyCharged>(
            surface.getSharedPtr(), lastPars, lastCov);
      }
    }();

    PropagatorOptions<Acts::ActionList<FinalizePositionPrinter>, Aborters>
        lastPropOptions(options.geoContext, options.magFieldContext, logger);

    lastPropOptions.direction = NavigationDirection::backward;
    lastPropOptions.targetTolerance *= 1000.0;

    ACTS_VERBOSE("+-------------------+");
    ACTS_VERBOSE("| Gsf: Do Last Part |");
    ACTS_VERBOSE("+-------------------+");
    auto lastPropRes = m_propagator.propagate(
        lastMultiPars, *options.referenceSurface, lastPropOptions);

    if (!lastPropRes.ok()) {
      return lastPropRes.error();
    }

    ////////////////////////////////////
    // Smooth and create Kalman Result
    ////////////////////////////////////

    ACTS_VERBOSE("Gsf: Do smoothing");

    const auto [combinedTraj, lastTip] = detail::smoothAndCombineTrajectories(
        fwdGsfResult.fittedStates, fwdGsfResult.currentTips,
        fwdGsfResult.weightsOfStates, bwdGsfResult.fittedStates,
        bwdGsfResult.currentTips, bwdGsfResult.weightsOfStates);

    Acts::KalmanFitterResult<source_link_t> kalmanResult;
    kalmanResult.lastTrackIndex = lastTip;
    kalmanResult.fittedStates = std::move(combinedTraj);
    kalmanResult.smoothed = true;
    kalmanResult.reversed = true;
    kalmanResult.finished = true;
    kalmanResult.lastMeasurementIndex = lastTip;
    kalmanResult.fittedParameters = *((*lastPropRes).endParameters);

    return kalmanResult;
  }
};

}  // namespace Acts
