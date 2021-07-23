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

#include <map>
#include <numeric>

#include "BetheHeitlerApprox.hpp"
#include "GsfUtils.hpp"
#include "KLMixtureReduction.hpp"
#include "MultiSteppingLogger.hpp"

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
};

template <typename source_link_t>
struct GsfResult {
  /// The multi-trajectory which stores the graph of components
  MultiTrajectory<source_link_t> fittedStates;
  std::map<size_t, ActsScalar> weightsOfStates;

  /// The current indexes for the active components in the multi trajectory
  std::vector<std::size_t> currentTips;

  /// The number of measurement states created
  std::size_t measurementStates = 0;
  std::size_t processedStates = 0;
};

template <typename propagator_t>
struct GaussianSumFitter {
  GaussianSumFitter(propagator_t propagator)
      : m_propagator(std::move(propagator)) {}

  /// The navigator type
  using GsfNavigator = typename propagator_t::Navigator;

  template <typename source_link_t, typename updater_t,
            typename outlier_finder_t, typename calibrator_t,
            typename smoother_t /*, typename parameters_t*/>
  struct Actor {
    /// Enforce default construction
    Actor() = default;

    /// Broadcast the result_type
    using result_type = GsfResult<source_link_t>;

    /// Broadcast the componentCache type
    using ComponentCache = detail::GsfComponentCache<source_link_t>;

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
    } m_config;

    /// Configurable components:
    updater_t m_updater;
    outlier_finder_t m_outlierFinder;
    calibrator_t m_calibrator;
    smoother_t m_smoother;
    SurfaceReached m_targetReachedAborter;

    /// @brief Kalman actor operation
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

      ACTS_VERBOSE("Gsf step at mean position "
                   << stepper.position(state.stepping).transpose()
                   << " with direction "
                   << stepper.direction(state.stepping).transpose());

      // TODO handle surfaces without material
      if (state.navigation.currentSurface &&
          state.navigation.currentSurface->surfaceMaterial()) {
        const auto& surface = *state.navigation.currentSurface;

        const auto source_link =
            m_config.inputMeasurements.find(surface.geometryId())->second;

        if (result.currentTips.empty()) {
          result.currentTips.resize(stepper.numberComponents(state.stepping),
                                    SIZE_MAX);
        }

        std::string_view direction =
            (state.stepping.navDir == forward) ? "forward" : "backward";
        ACTS_VERBOSE("Perform " << direction << " filter step");
        filter(state, stepper, result, source_link);
      }
    }

    /// @brief A filtering step
    template <typename propagator_state_t, typename stepper_t>
    Result<void> filter(propagator_state_t& state, const stepper_t& stepper,
                        result_type& result,
                        const source_link_t& measurement) const {
      const auto& logger = state.options.logger;
      const auto& surface = *state.navigation.currentSurface;

      throw_assert(
          result.currentTips.size() == stepper.numberComponents(state.stepping),
          "component number mismatch:"
              << result.currentTips.size() << " vs "
              << stepper.numberComponents(state.stepping));

      // Static std::vector to avoid reallocation every pass. Reserve enough
      // space to allow all possible components to be stored
      thread_local std::vector<ComponentCache> componentCache;
      componentCache.reserve(m_config.maxComponents *
                             m_config.bethe_heitler_approx.numComponents());
      componentCache.clear();

      ACTS_VERBOSE("GSF actor: Start filtering");

      // Apply material effect by creating new components
      create_new_components(state, stepper, componentCache, result.currentTips,
                            surface);

      throw_assert(!componentCache.empty(),
                   "components are empty after creation");
      ACTS_VERBOSE("GSF actor: " << componentCache.size()
                                 << " components candidates");

      // Reduce the number of components to a managable degree
      if (componentCache.size() > m_config.maxComponents ||
          componentCache.size() > stepper.maxComponents) {
        detail::reduceWithKLDistance(
            componentCache, std::min(static_cast<int>(m_config.maxComponents),
                                     stepper.maxComponents));
        //         detail::reduceNumberOfComponents(componentCache,
        //                   std::min(static_cast<int>(m_config.maxComponents),
        //                            stepper.maxComponents));
      }

      throw_assert(!componentCache.empty(),
                   "components are empty after reduction");
      ACTS_VERBOSE("GSF actor: " << componentCache.size()
                                 << " after component reduction");

      // Update the Multi trajectory with the new components
      auto res =
          kalmanUpdateForward(state, result, measurement, componentCache);

      if (!res.ok()) {
        throw_assert(false, "no error should occur here");
        return res.error();
      }

      // Reweigth components according to measurement
      detail::reweightComponents(componentCache);

      // Store the weights TODO can we incorporate weights in MultiTrajectory?
      for (const auto& cmp : componentCache) {
        const auto [it, success] = result.weightsOfStates.insert(
            {cmp.trackStateProxy->index(), cmp.weight});
        throw_assert(success, "Could not insert weight");
      }

      // update stepper state with the collected component caches
      stepper.updateComponents(state.stepping, componentCache, surface);

      // Add the combined track state to results
      //       result.combinedPars.push_back(
      //           detail::combineMultiComponentState(componentCache, surface));

      return Result<void>::success();
    }

    /// @brief Expands all existing components to new components by using a
    /// gaussian-mixture approximation for the Bethe-Heitler distribution.
    ///
    /// @return a std::vector with all new components (parent tip, weight,
    /// parameters, covariance)
    template <typename propagator_state_t, typename stepper_t>
    void create_new_components(
        propagator_state_t& state, const stepper_t& stepper,
        std::vector<ComponentCache>& new_components,
        const std::vector<std::size_t>& parentTrajectoryIdxs,
        const Surface& surface) const {
      // Some shortcuts
      auto& stepping = state.stepping;
      const auto& logger = state.options.logger;
      const auto& gctx = stepping.geoContext;
      const auto navDir = stepping.navDir;
      const auto surfaceMaterial = surface.surfaceMaterial();

      // Remove old components and reserve space for the number of possible
      // components
      new_components.clear();
      new_components.reserve(parentTrajectoryIdxs.size() *
                             m_config.bethe_heitler_approx.numComponents());

      // Approximate bethe-heitler distribution as gaussian mixture
      for (auto i = 0ul; i < stepper.numberComponents(stepping); ++i) {
        typename stepper_t::ComponentProxy old_cmp(stepping, i);

        auto boundState = old_cmp.boundState(surface, m_config.doCovTransport);

        if (!boundState.ok()) {
          ACTS_ERROR("Error with boundstate: " << boundState.error());
          continue;
        }

        const auto& [old_bound, jac, pathLength] = boundState.value();
        const auto p_prev = old_bound.absoluteMomentum();
        const auto slab = surfaceMaterial->materialSlab(
            old_bound.position(gctx), navDir, MaterialUpdateStage::fullUpdate);

        const auto mixture =
            m_config.bethe_heitler_approx.mixture(slab.thicknessInX0());

        // Create all possible new components
        for (const auto& gaussian : mixture) {
          ComponentCache new_cmp;

          // compute delta p from mixture and update parameters
          const auto delta_p = [&]() {
            if (navDir == NavigationDirection::forward)
              return p_prev * (gaussian.mean - 1.);
            else
              return p_prev * (1. / gaussian.mean - 1.);
          }();

          throw_assert(p_prev + delta_p > 0.,
                       "new momentum after bethe-heitler must be > 0");

          new_cmp.predictedPars = old_bound.parameters();
          new_cmp.predictedPars[eBoundQOverP] =
              old_bound.charge() / (p_prev + delta_p);

          // compute inverse variance of p from mixture and update covariance
          if (old_bound.covariance()) {
            const auto varInvP = [&]() {
              if (navDir == NavigationDirection::forward) {
                const auto f = 1. / (p_prev * gaussian.mean);
                return f * f * gaussian.var;
              } else {
                return gaussian.var / (p_prev * p_prev);
              }
            }();

            new_cmp.predictedCov = old_bound.covariance();
            (*new_cmp.predictedCov)(eBoundQOverP, eBoundQOverP) += varInvP;
          }

          // Here we combine the new child weight with the parent weight.
          // However, this must be later re-adjusted
          new_cmp.weight = gaussian.weight * old_cmp.weight();

          // Set the remaining things and push to vector
          new_cmp.jacobian = jac;
          new_cmp.pathLength = pathLength;
          new_cmp.jacToGlobal = old_cmp.jacToGlobal();
          new_cmp.derivative = old_cmp.derivative();
          new_cmp.jacTransport = old_cmp.jacTransport();
          new_cmp.parentIndex = parentTrajectoryIdxs[i];

          new_components.push_back(new_cmp);
        }
      }
    }

    template <typename propagator_state_t>
    Acts::Result<void> kalmanUpdateForward(
        propagator_state_t& state, result_type& result,
        const source_link_t measurement,
        std::vector<ComponentCache>& new_components) const {
      // Some shortcut references
      const auto& logger = state.options.logger;
      const auto& surface = *state.navigation.currentSurface;

      // Clear the current tips, to set the new ones
      result.currentTips.clear();

      // Loop over new components, add to MultiTrajectory and do kalman update
      for (auto i = 0ul; i < new_components.size(); ++i) {
        auto& cmp = new_components[i];

        // Create new track state
        result.currentTips.push_back(result.fittedStates.addTrackState(
            TrackStatePropMask::All, cmp.parentIndex));

        cmp.trackStateProxy =
            result.fittedStates.getTrackState(result.currentTips.back());

        auto& trackProxy = *cmp.trackStateProxy;

        // Set surface
        trackProxy.setReferenceSurface(surface.getSharedPtr());

        // assign the source link to the track state
        trackProxy.uncalibrated() = measurement;

        // Set track parameters
        trackProxy.predicted() = std::move(cmp.predictedPars);
        if (cmp.predictedCov) {
          trackProxy.predictedCovariance() = std::move(*cmp.predictedCov);
        }

        trackProxy.jacobian() = std::move(cmp.jacobian);
        trackProxy.pathLength() = std::move(cmp.pathLength);

        // We have predicted parameters, so calibrate the uncalibrated input
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
            return updateRes.error();
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

    /// @brief Compute the combined bound track parameters at the current surface and mark result as finished
    template <typename propagator_state_t, typename stepper_t>
    void finalize(propagator_state_t& state, const stepper_t& stepper,
                  result_type& result) const {
      const auto& logger = state.options.logger;
      const auto& surface = *state.navigation.currentSurface;

      ACTS_VERBOSE("finalize");

      std::vector<
          std::tuple<ActsScalar, BoundVector, std::optional<BoundSymMatrix>>>
          components;
      components.reserve(stepper.numberComponents(state.stepping));

      bool error = false;

      for (auto i = 0ul; i < stepper.numberComponents(state.stepping); ++i) {
        typename stepper_t::ComponentProxy cmp(state.stepping, i);

        auto bs = cmp.boundState(surface, true);

        if (not bs.ok()) {
          error = true;
          ACTS_ERROR("Error in boundstate conversion: " << bs.error());
          ACTS_INFO("Component is at global position "
                    << cmp.pars().template segment<3>(eFreePos0).transpose());
          continue;
        }

        components.push_back(std::tuple{cmp.weight(),
                                        (std::get<0>(*bs)).parameters(),
                                        (std::get<0>(*bs)).covariance()});
      }

      throw_assert(!error, "Error occured in the bound state conversions");

      const auto [finalPars, finalCov] = detail::combineComponentRange(
          components.begin(), components.end(), detail::Identity{});

      result.finalParameters =
          BoundTrackParameters(surface.getSharedPtr(), finalPars, finalCov);
      result.isFinished = true;
    }
  };

  struct GsfFinalizerResult {};

  struct GsfFinalizer {
    /// Enforce default construction
    GsfFinalizer() = default;

    /// Broadcast the result_type
    //     using result_type =
    //     std::optional<Acts::Result<BoundTrackParameters>>;
    using result_type = std::optional<BoundTrackParameters>;

    const Surface* targetSurface = nullptr;

    /// @brief Only acts if we are on the last surface
    template <typename propagator_state_t, typename stepper_t>
    void operator()(propagator_state_t& state, const stepper_t& stepper,
                    result_type& result) const {
      throw_assert(targetSurface, "we need a target surface");
      const auto& logger = state.options.logger;

      ACTS_VERBOSE("Finalizer step at mean position "
                   << stepper.position(state.stepping).transpose()
                   << " with direction "
                   << stepper.direction(state.stepping).transpose());

      if (state.navigation.currentSurface == targetSurface) {
        throw_assert(stepper.numberComponents(state.stepping) == 1,
                     "we expect single component state here");
        typename stepper_t::ComponentProxy cmp(state.stepping, 0);

        // TODO get the correct bool value here later...
        ACTS_VERBOSE("Evaluate final track parameters");
        auto boundRes = cmp.boundState(*targetSurface, true);

        // Propagate result error to the outside
        throw_assert(boundRes.ok(), "bound result must be ok");

        result = std::get<BoundTrackParameters>(*boundRes);
      }
    }
  };

  template <typename gsf_actor_t>
  class Aborter {
   public:
    /// Broadcast the result_type
    using action_type = gsf_actor_t;

    /// The call operator
    template <typename propagator_state_t, typename stepper_t,
              typename result_t>
    bool operator()(propagator_state_t& /*state*/, const stepper_t& /*stepper*/,
                    const result_t& /*result*/) const {
      //       const auto& logger = state.options.logger;

      //       if (result.isFinished) {
      //         ACTS_VERBOSE("Terminated by GSF Aborter");
      //         return true;
      //       }
      return false;
    }
  };

  /// The propagator instance used by the fit function
  propagator_t m_propagator;

  /// @brief Function which does the forward pass
  template <typename source_link_t, typename propagator_options_t>
  Acts::Result<std::vector<detail::MultiComponentState>> gsfPropagationPass(
      propagator_options_t& propOptions,
      const MultiComponentBoundTrackParameters<SinglyCharged>& params,
      const std::vector<const Surface*>& surfaceSequence,
      Acts::NavigationDirection navDir) const {
    propOptions.direction = navDir;

    if (navDir == Acts::forward) {
      propOptions.actionList.template get<DirectNavigator::Initializer>()
          .navSurfaces = surfaceSequence;
    } else {
      std::vector<const Surface *> backwardSequence(std::next(surfaceSequence.rbegin()), surfaceSequence.rend());
      std::cout << "size of backward Sequence " << backwardSequence.size() << std::endl;
      propOptions.actionList.template get<DirectNavigator::Initializer>()
          .navSurfaces = std::move(backwardSequence);
    }

    auto propResult = m_propagator.propagate(params, propOptions);

    if (!propResult.ok()) {
      return propResult.error();
    }

    GsfResult gsfForwardResult =
        (*propResult).template get<GsfResult<source_link_t>>();

    auto states = detail::extractMultiComponentStates(
        gsfForwardResult.fittedStates, gsfForwardResult.currentTips,
        gsfForwardResult.weightsOfStates, false);

    if (navDir == Acts::NavigationDirection::backward)
      std::reverse(states.begin(), states.end());

    return states;
  }

  /// @brief The fit function
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

    ACTS_VERBOSE("Gsf: Number of surfaces in sSequence: " << sSequence.size());
    ACTS_VERBOSE(
        "Gsf: Final measuerement map size: " << inputMeasurements.size());
    throw_assert(sParameters.covariance() != std::nullopt,
                 "we need a covariance here...");
    ACTS_VERBOSE("Gsf: Start parameters: \n" << sParameters);

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
    actor.m_config.maxComponents = 4;  // for easier debugging just 4 components
    actor.m_calibrator = options.calibrator;
    actor.m_outlierFinder = options.outlierFinder;

    // Run the fitter forward
    MultiComponentBoundTrackParameters<SinglyCharged> sMultiParsFwd(
        sParameters.referenceSurface().getSharedPtr(), sParameters.parameters(),
        sParameters.covariance());

    ACTS_VERBOSE("Gsf: Do forward propagation");
    auto fwdResult = gsfPropagationPass<source_link_t>(
        propOptions, sMultiParsFwd, sSequence, Acts::forward);

    if (!fwdResult.ok()) {
      return fwdResult.error();
    }

    const auto& fwdStates = *fwdResult;

    // Run backward
    MultiComponentBoundTrackParameters<SinglyCharged> sMultiParsBwd(
        fwdStates.front().first, fwdStates.front().second);

    ACTS_VERBOSE("Gsf: Do backward propagation");
    auto bwdResult = gsfPropagationPass<source_link_t>(
        propOptions, sMultiParsBwd, sSequence, Acts::backward);

    if (!bwdResult.ok()) {
      return bwdResult.error();
    }

    const auto& bwdStates = *bwdResult;

    // Do the last part to the reference surface
    const auto [lastPars, lastCov] = detail::combineComponentRange(
        bwdStates.back().second.begin(), bwdStates.back().second.end());
    MultiComponentBoundTrackParameters<SinglyCharged> lastMultiPars(
        bwdStates.back().first, lastPars, lastCov);

    PropagatorOptions<
        Acts::ActionList<DirectNavigator::Initializer, GsfFinalizer>, Aborters>
        lastPropOptions(options.geoContext, options.magFieldContext, logger);

    lastPropOptions.actionList.template get<DirectNavigator::Initializer>()
        .navSurfaces = {options.referenceSurface};
    lastPropOptions.actionList.template get<GsfFinalizer>()
        .targetSurface = options.referenceSurface;
    lastPropOptions.direction = NavigationDirection::backward;

    ACTS_VERBOSE("Gsf: Do last steps back to reference surface");
    std::cout << "surface sequence size = " << lastPropOptions.actionList.template get<DirectNavigator::Initializer>()
        .navSurfaces.size() << "\n";
    auto propLastResult = m_propagator.propagate(
        lastMultiPars, *options.referenceSurface, lastPropOptions);

    if (!propLastResult.ok()) {
      return propLastResult.error();
    }

    // Do the smoothing
    ACTS_VERBOSE("Gsf: Start smoothing");
    std::vector<detail::MultiComponentState> smoothed;
    throw_assert(fwdStates.size() == bwdStates.size(), "size mismatch");
    for (auto i = 0ul; i < fwdStates.size(); ++i) {
      smoothed.push_back(detail::bayesianSmoothing(fwdStates[i], bwdStates[i]));
    }

    ACTS_VERBOSE("Gsf: Combine smoothed states");
    std::vector<BoundTrackParameters> combinedSmoothed;
    for (auto& state : smoothed) {
      detail::normalizeMultiComponentState(state);
      const auto [smoothedPars, smoothedCov] = detail::combineComponentRange(
          state.second.begin(), state.second.end(), detail::Identity{});
      combinedSmoothed.push_back(
          BoundTrackParameters(state.first, smoothedPars, smoothedCov));
    }

    // Create Kalman result for return
    ACTS_VERBOSE("Gsf: Create Kalman Result");
    Acts::KalmanFitterResult<source_link_t> kalmanResult;
    kalmanResult.lastTrackIndex = SIZE_MAX;

    // Reverse combined smoothed state again to match the output scheme of
    // KalmanFitter
    std::reverse(combinedSmoothed.begin(), combinedSmoothed.end());

    for (const auto& stage : combinedSmoothed) {
      kalmanResult.lastTrackIndex = kalmanResult.fittedStates.addTrackState(
          TrackStatePropMask::All, kalmanResult.lastTrackIndex);

      auto proxy =
          kalmanResult.fittedStates.getTrackState(kalmanResult.lastTrackIndex);

      proxy.smoothed() = stage.parameters();
      if (stage.covariance()) {
        proxy.smoothedCovariance() = *stage.covariance();
      }
      proxy.setReferenceSurface(stage.referenceSurface().getSharedPtr());
    }

    kalmanResult.lastMeasurementIndex = kalmanResult.lastTrackIndex;
    kalmanResult.fittedParameters = (*propLastResult).template get<typename GsfFinalizer::result_type>().value();

    return kalmanResult;
  }
};

}  // namespace Acts
