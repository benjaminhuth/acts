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

  /// The collected BoundStates along the trajectory
  std::vector<BoundTrackParameters> combinedPars;

  // Measurement surfaces handled in both forward and backward filtering
  std::vector<const Surface*> passedAgainSurfaces;

  // Flag if we are isFinished
  bool isFinished = false;
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
      
      /// Target surface for backwards
      const Surface *targetSurface = nullptr;
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

      if (state.navigation.targetSurface &&
          m_targetReachedAborter(state, stepper)) {
        ACTS_VERBOSE("Gsf Actor reached target surface");
        result.isFinished = true;
        return;
      }

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

    //     /// @brief Kalman actor operation
    //     ///
    //     /// @tparam propagator_state_t is the type of Propagagor state
    //     /// @tparam stepper_t Type of the stepper
    //     ///
    //     /// @param state is the mutable propagator state object
    //     /// @param stepper The stepper in use
    //     /// @param result is the mutable result state object
    //     template <typename propagator_state_t, typename stepper_t>
    //     void operator()(propagator_state_t& state, const stepper_t& stepper,
    //                     result_type& result) const {
    //       const auto& logger = state.options.logger;
    //
    //       ACTS_VERBOSE("Gsf step");
    //
    //       if (result.isFinished) {
    //         ACTS_VERBOSE("Gsf Actor: Fast return because of isFinished");
    //         return;
    //       }
    //
    //       throw_assert(m_config.targetSurface, "we must have a target
    //       surface");
    //
    //       ////////////////
    //       // Pre-Config //
    //       ////////////////
    //
    //       // This is because we do not have a DirectNavigator option for now.
    //       Should
    //       // also be resolved by #846
    //       {
    //         if (result.processedStates == 0) {
    //           for (auto measurementIt = m_config.inputMeasurements.begin();
    //                measurementIt != m_config.inputMeasurements.end();
    //                measurementIt++) {
    //             state.navigation.externalSurfaces.insert(
    //                 std::pair<uint64_t, GeometryIdentifier>(
    //                     measurementIt->first.layer(), measurementIt->first));
    //           }
    //         }
    //
    //         if (result.doReset and state.navigation.navSurfaceIter ==
    //                                    state.navigation.navSurfaces.end()) {
    //           // So the navigator target call will target layers
    //           state.navigation.navigationStage =
    //           GsfNavigator::Stage::layerTarget;
    //           // We only do this after the backward-propagation
    //           // starting layer has been processed
    //           result.doReset = false;
    //         }
    //       }
    //
    //       ////////////
    //       // Update //
    //       ////////////
    //       //
    //       std::string direction =
    //           (state.stepping.navDir == forward) ? "forward" : "backward";
    //
    //       // TODO handle surfaces without material
    //       if (state.navigation.currentSurface &&
    //           state.navigation.currentSurface->surfaceMaterial()) {
    //         const auto& surface = *state.navigation.currentSurface;
    //
    //         const auto source_link =
    //             m_config.inputMeasurements.find(surface.geometryId())->second;
    //
    //         if(result.currentTips.empty()) {
    //             result.currentTips.resize(stepper.numberComponents(state.stepping),
    //             SIZE_MAX);
    //         }
    //
    //         auto res = [&]() {
    //           if (not result.isReversed) {
    //             ACTS_VERBOSE("Perform " << direction << " filter step");
    //             return filter(state, stepper, result, source_link);
    //           } else {
    //             ACTS_VERBOSE("Perform " << direction << " filter step");
    //             return filter(state, stepper, result, source_link);
    //           }
    //         }();
    //
    //         if (!res.ok()) {
    //           ACTS_ERROR("Error in " << direction << " filter: " <<
    //           res.error());
    //           //           result.result = res.error();
    //         }
    //       }
    //
    //       //////////////////
    //       // Finalization //
    //       //////////////////
    //       bool canReverse =
    //           result.measurementStates == m_config.inputMeasurements.size()
    //           || (result.measurementStates > 0 &&
    //           state.navigation.navigationBreak);
    //
    //       if (not result.isReversed && canReverse) {
    //         throw_assert(false, "break here at reserve");
    //         reverse(state, stepper, result);
    //       }
    //
    //       ///////////////////////
    //       // Post-finalization //
    //       ///////////////////////
    //
    //       if (result.isReversed &&
    //           m_targetReachedAborter(state, stepper,
    //           *m_config.targetSurface)) {
    //         ACTS_VERBOSE("Completing with fitted track parameter");
    //
    //         throw_assert(
    //             result.combinedParsForward.size() ==
    //                 result.combinedParsReverse.size(),
    //             "both reverse and forward pass results must match in size");
    //
    //         std::reverse(result.combinedParsReverse.begin(),
    //                      result.combinedParsReverse.end());
    //
    //         result.isFinished = true;
    //       }
    //     }

    /// @brief A filtering step
    template <typename propagator_state_t, typename stepper_t>
    Result<void> filter(propagator_state_t& state, const stepper_t& stepper,
                        result_type& result,
                        const source_link_t& measurement) const {
      const auto& logger = state.options.logger;
      const auto& surface = *state.navigation.currentSurface;

      throw_assert(
          result.currentTips.size() == stepper.numberComponents(state.stepping),
          "component number mismatch");

      // Static std::vector to avoid reallocation every pass. Reserve enough
      // space to allow all possible components to be stored
      static std::vector<ComponentCache> componentCache;
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
        reduceNumberOfComponents(
            componentCache, std::min(static_cast<int>(m_config.maxComponents),
                                     stepper.maxComponents));
      }

      throw_assert(!componentCache.empty(),
                   "components are empty after reduction");
      ACTS_VERBOSE("GSF actor: " << componentCache.size()
                                 << " after component reduction");

      // Update the Multi trajectory with the new components
      auto res =
          kalmanUpdateForward(state, result, measurement, componentCache);

      if (!res.ok()) {
        return res.error();
      }

      // Reweigth components according to measurement
      detail::reweightComponents(componentCache);

      // Store the weights
      for (const auto& cmp : componentCache) {
        result.weightsOfStates[cmp.trackStateProxy->index()] = cmp.weight;
      }

      // update stepper state with the collected component caches
      stepper.updateComponents(state.stepping, componentCache, surface);

      // Add the combined track state to results
      result.combinedPars.push_back(
          detail::combineMultiComponentState(componentCache, surface));

      ACTS_VERBOSE("GSF actor: " << stepper.numberComponents(state.stepping)
                                 << " at the end");

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

      for (auto i = 0ul; i < stepper.numberComponents(stepping); ++i) {
        // Approximate bethe-heitler distribution as gaussian mixture
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

    /// @brief Function that reduces the number of components. at the moment,
    /// this is just by erasing the components with the lowest weights.
    /// Finally, the components are reweighted so the sum of the weights is
    /// still 1
    void reduceNumberOfComponents(
        std::vector<ComponentCache>& components,
        std::size_t numberOfRemainingCompoents) const {
      // The elements should be sorted by weight (high -> low)
      std::sort(
          begin(components), end(components),
          [](const auto& a, const auto& b) { return a.weight > b.weight; });

      // Remove elements by resize
      components.erase(begin(components) + numberOfRemainingCompoents,
                       end(components));

      // Reweight after removal
      const auto sum_of_weights = std::accumulate(
          begin(components), end(components), 0.0,
          [](auto sum, const auto& cmp) { return sum + cmp.weight; });

      for (auto& cmp : components) {
        cmp.weight /= sum_of_weights;
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

    //     template <typename propagator_state_t>
    //     Acts::Result<void> kalmanUpdateReverse(
    //         propagator_state_t& state, result_type& result,
    //         const source_link_t measurement,
    //         std::vector<ComponentCache>& new_components) const {
    //       const auto& logger = state.options.logger;
    //       const auto& surface = *state.navigation.currentSurface;
    //
    //       // Screen output message
    //       ACTS_VERBOSE("Measurement surface "
    //                    << surface->geometryId()
    //                    << " detected in reversed propagation.");
    //
    //       // No reversed filtering for last measurement state, but still
    //       update
    //       // with material effects
    //       if (result.reversed and surface == state.navigation.startSurface) {
    //         //           materialInteractor(surface, state, stepper);
    //         return Result<void>::success();
    //       }
    //
    //       // Create a detached track state proxy since we add everything to
    //       the
    //       // forward track in the end
    //
    //       // Loop over new components, add to MultiTrajectory and do kalman
    //       update for (auto i = 0ul; i < new_components.size(); ++i) {
    //         auto& cmp = new_components[i];
    //
    //         auto tempTrackTip =
    //             result.fittedStates.addTrackState(TrackStatePropMask::All);
    //
    //         cmp.trackStateProxy =
    //         result.fittedStates.getTrackState(tempTrackTip);
    //
    //         auto& trackProxy = *cmp.trackStateProxy;
    //
    //         // Set surface
    //         trackProxy.setReferenceSurface(surface.getSharedPtr());
    //
    //         // assign the source link to the track state
    //         trackProxy.uncalibrated() = measurement;
    //
    //         // Set track parameters
    //         trackProxy.predicted() = std::move(cmp.predictedPars);
    //         if (cmp.predictedCov) {
    //           trackProxy.predictedCovariance() =
    //           std::move(*cmp.predictedCov);
    //         }
    //
    //         trackProxy.jacobian() = std::move(cmp.jacobian);
    //         trackProxy.pathLength() = std::move(cmp.pathLength);
    //
    //         // We have predicted parameters, so calibrate the uncalibrated
    //         input std::visit(
    //             [&](const auto& calibrated) {
    //               trackProxy.setCalibrated(calibrated);
    //             },
    //             m_calibrator(trackProxy.uncalibrated(),
    //             trackProxy.predicted()));
    //
    //         // Get and set the type flags
    //         trackProxy.typeFlags().set(TrackStateFlag::ParameterFlag);
    //         if (surface.surfaceMaterial() != nullptr) {
    //           trackProxy.typeFlags().set(TrackStateFlag::MaterialFlag);
    //         }
    //
    //         // Perform update
    //         auto updateRes = m_updater(state.geoContext, trackProxy,
    //                                    state.stepping.navDir, logger);
    //
    //         // Check update result
    //         if (!updateRes.ok()) {
    //           ACTS_ERROR("Backward update step failed: " <<
    //           updateRes.error()); return updateRes.error();
    //         } else {
    //           // Update the stepping state with filtered parameters
    //           ACTS_VERBOSE(
    //               "Backward Filtering step successful, updated parameters are
    //               :\n"
    //               << trackProxy.filtered().transpose());
    //
    //           // Fill the smoothed parameter for the existing track state
    //           result.fittedStates.applyBackwards(
    //               result.lastMeasurementIndex, [&](auto trackState) {
    //                 auto fSurface = &trackState.referenceSurface();
    //                 if (fSurface == surface) {
    //                   result.passedAgainSurfaces.push_back(surface);
    //                   trackState.smoothed() = trackProxy.filtered();
    //                   trackState.smoothedCovariance() =
    //                       trackProxy.filteredCovariance();
    //                   return false;
    //                 }
    //                 return true;
    //               });
    //         }
    //       }
    //
    //       return Result<void>::success();
    //     }

    //     /// @brief GSF actor operation : reverse direction
    //     ///
    //     /// @tparam propagator_state_t is the type of Propagagor state
    //     /// @tparam stepper_t Type of the stepper
    //     ///
    //     /// @param state is the mutable propagator state object
    //     /// @param stepper The stepper in use
    //     /// @param result is the mutable result state objecte
    //     template <typename propagator_state_t, typename stepper_t>
    //     void reverse(propagator_state_t& state, stepper_t& stepper,
    //                  result_type& result) const {
    //       // Remember the navigation direciton has been reversed
    //       result.isReversed = true;
    //       // The reset is used for resetting the navigation
    //       result.doReset = true;
    //
    //       // Reverse navigation direction
    //       state.stepping.navDir =
    //           (state.stepping.navDir == forward) ? backward : forward;
    //
    //       // Reset propagator options
    //       state.options.maxStepSize =
    //           state.stepping.navDir * std::abs(state.options.maxStepSize);
    //       // Not sure if reset of pathLimit during propagation makes any
    //       sense state.options.pathLimit =
    //           state.stepping.navDir * std::abs(state.options.pathLimit);
    //
    //       // Reset stepping&navigation state using last measurement track
    //       state on
    //       // sensitive surface
    //       state.navigation = typename propagator_t::NavigatorState();
    //
    //       for (const auto idx : result.currentTips) {
    //         result.fittedStates.applyBackwards(idx, [&](auto st) {
    //           if (st.typeFlags().test(Acts::TrackStateFlag::MeasurementFlag))
    //           {
    //             // Set the navigation state
    //             state.navigation.startSurface = &st.referenceSurface();
    //             if (state.navigation.startSurface->associatedLayer() !=
    //             nullptr) {
    //               state.navigation.startLayer =
    //                   state.navigation.startSurface->associatedLayer();
    //             }
    //             state.navigation.startVolume =
    //                 state.navigation.startLayer->trackingVolume();
    //             state.navigation.targetSurface = m_config.targetSurface;
    //             state.navigation.currentSurface =
    //             state.navigation.startSurface; state.navigation.currentVolume
    //             = state.navigation.startVolume;
    //
    //             // Update the stepping state
    //             stepper.resetState(state.stepping, st.filtered(),
    //                                st.filteredCovariance(),
    //                                st.referenceSurface(),
    //                                state.stepping.navDir,
    //                                state.options.maxStepSize);
    //
    //             // For the last measurement state, smoothed is filtered
    //             st.smoothed() = st.filtered();
    //             st.smoothedCovariance() = st.filteredCovariance();
    //             result.passedAgainSurfaces.push_back(&st.referenceSurface());
    //
    //             // Update material effects for last measurement state in
    //             reversed
    //             // direction
    //             // materialInteractor(state.navigation.currentSurface,
    //             //             state, stepper);
    //
    //             return false;  // abort execution
    //           }
    //           return true;  // continue execution
    //         });
    //       }
    //     }
  };

  template <typename gsf_actor_t>
  class Aborter {
   public:
    /// Broadcast the result_type
    using action_type = gsf_actor_t;

    /// The call operator
    template <typename propagator_state_t, typename stepper_t,
              typename result_t>
    bool operator()(propagator_state_t& state, const stepper_t& /*stepper*/,
                    const result_t& result) const {
      const auto& logger = state.options.logger;

      if (result.isFinished) {
        ACTS_VERBOSE("Terminated by GSF Aborter");
        return true;
      }
      return false;
    }
  };

  /// The propagator instance used by the fit function
  propagator_t m_propagator;

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

    ACTS_VERBOSE("Number of surfaces in sSequence: " << sSequence.size());
    ACTS_VERBOSE("Final measuerement map size: " << inputMeasurements.size());

    // Create the ActionList and AbortList
    using GSFActor = Actor<source_link_t, GainMatrixUpdater, outlier_finder_t,
                           calibrator_t, GainMatrixSmoother>;
    using GSFAborter = Aborter<GSFActor>;

    using Actors =
        ActionList<DirectNavigator::Initializer, GSFActor, MultiSteppingLogger>;
    using Aborters = AbortList<GSFAborter>;

    // Create relevant options for the propagation options
    PropagatorOptions<Actors, Aborters> propOptions(
        options.geoContext, options.magFieldContext, logger);

    // Catch the actor and set the measurements
    auto& actor = propOptions.actionList.template get<GSFActor>();
    actor.m_config.inputMeasurements = inputMeasurements;
    actor.m_config.maxComponents = 4;  // for easier debugging just 4 components
    actor.m_calibrator = options.calibrator;
    actor.m_outlierFinder = options.outlierFinder;

    // Set the surface sequence
    auto& dInitializer =
        propOptions.actionList.template get<DirectNavigator::Initializer>();
    dInitializer.navSurfaces = sSequence;

    // Make Single Start Parameters to Multi Start parameters
    throw_assert(sParameters.covariance() != std::nullopt,
                 "we need a covariance here...");

    ACTS_VERBOSE("Start parameters: \n" << sParameters);

    MultiComponentBoundTrackParameters<SinglyCharged> sMultiPars(
        sParameters.referenceSurface().getSharedPtr(), sParameters.parameters(),
        sParameters.covariance());

    // Run the fitter forward
    ACTS_VERBOSE("Start forward propagation");
    propOptions.direction = NavigationDirection::forward;
    auto propResultForward = m_propagator.propagate(sMultiPars, propOptions);

    if (!propResultForward.ok()) {
      return propResultForward.error();
    }

    GsfResult gsfForwardResult =
        (*propResultForward).template get<GsfResult<source_link_t>>();

    // Create the start parameters for the backwards run
    std::vector<std::tuple<double, BoundVector, BoundSymMatrix>> cmps;
    std::shared_ptr<const Surface> surf;

    for (const auto idx : gsfForwardResult.currentTips) {
      auto trackProxy = gsfForwardResult.fittedStates.getTrackState(idx);
      cmps.push_back({gsfForwardResult.weightsOfStates.at(idx),
                      trackProxy.filtered(), trackProxy.filteredCovariance()});
      if (!surf) {
        surf = trackProxy.referenceSurface().getSharedPtr();
      } else {
        throw_assert(
            surf->geometryId() == trackProxy.referenceSurface().geometryId(),
            "surfaces must match");
      }
    }

    MultiComponentBoundTrackParameters<SinglyCharged> sMultiParsBackward(surf,
                                                                         cmps);

    // Initialize direct navigator
    auto reverseSequence = sSequence;
    std::reverse(reverseSequence.begin(), reverseSequence.end());
    reverseSequence.push_back(options.referenceSurface);
    auto& revDInitializer =
        propOptions.actionList.template get<DirectNavigator::Initializer>();
    revDInitializer.navSurfaces = reverseSequence;
    
    // Run backward
    ACTS_VERBOSE("Start backward propagation");
    propOptions.direction = NavigationDirection::backward;
    auto propBackwardResult = m_propagator.propagate(
        sMultiParsBackward, propOptions);

    if (!propBackwardResult.ok()) {
      return propBackwardResult.error();
    }

    GsfResult gsfBackwardResult =
        (*propBackwardResult).template get<GsfResult<source_link_t>>();

    // Do weighted-mean smoothing
    std::reverse(gsfBackwardResult.combinedPars.begin(),
                 gsfBackwardResult.combinedPars.end());

    for (auto i = 0ul; i < std::max(gsfBackwardResult.combinedPars.size(),
                                    gsfForwardResult.combinedPars.size());
         ++i) {
      if (i < gsfForwardResult.combinedPars.size())
        std::cout
            << "Fwd: "
            << gsfForwardResult.combinedPars[i].referenceSurface().geometryId();
      else
        std::cout << "Fwd: "
                  << "invalid";

      std::cout << "\t<->\t";

      if (i < gsfBackwardResult.combinedPars.size())
        std::cout << "Bwd: "
                  << gsfBackwardResult.combinedPars[i]
                         .referenceSurface()
                         .geometryId();
      else
        std::cout << "Bwd: "
                  << "invalid";

      std::cout << "\n";
    }

    const auto weightedMeanSmoothed = detail::combineForwardAndBackwardPass(
        gsfForwardResult.combinedPars, gsfBackwardResult.combinedPars);

    // Create Kalman result for return
    Acts::KalmanFitterResult<source_link_t> kalmanResult;
    kalmanResult.lastTrackIndex = SIZE_MAX;

    for (const auto& stage : weightedMeanSmoothed) {
      kalmanResult.lastTrackIndex = kalmanResult.fittedStates.addTrackState(
          TrackStatePropMask::All, kalmanResult.lastTrackIndex);

      auto proxy =
          kalmanResult.fittedStates.getTrackState(kalmanResult.lastTrackIndex);

      std::cout << "stage pars: " << stage.parameters().transpose() << "\n";

      proxy.smoothed() = stage.parameters();
      if (stage.covariance()) {
        proxy.smoothedCovariance() = *stage.covariance();
      }
      proxy.setReferenceSurface(stage.referenceSurface().getSharedPtr());
    }

    kalmanResult.lastMeasurementIndex = kalmanResult.lastTrackIndex;
    kalmanResult.fittedParameters = weightedMeanSmoothed.front();

    return kalmanResult;
  }
};

}  // namespace Acts
