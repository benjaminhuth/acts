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
struct GSFOptions {
  using Calibrator = calibrator_t;
  using OutlierFinder = outlier_finder_t;

  Calibrator calibrator;
  OutlierFinder outlierFinder;

  std::reference_wrapper<const GeometryContext> geoContext;
  std::reference_wrapper<const MagneticFieldContext> magFieldContext;

  const Surface referenceSurface = nullptr;

  LoggerWrapper logger;
};

template <typename propagator_t>
struct GaussianSumFitter {
  GaussianSumFitter(propagator_t propagator)
      : m_propagator(std::move(propagator)) {}

  template <typename source_link_t>
  struct Result {
    /// The multi-trajectory which stores the graph of components
    MultiTrajectory<source_link_t> fittedStates;

    /// The current indexes for the active components in the multi trajectory
    std::vector<std::size_t> currentTips;

    /// The number of measurement states created
    std::size_t measurementStates = 0;

    /// The collected BoundStates along the trajectory
    std::vector<BoundTrackParameters> combinedTrackParameters;

    /// Boolean flag which indicates the fitting is done
    bool isFinished = false;

    /// Boolean flag which indicates the components have been smoothed
    bool isSmoothed = false;
  };

  template <typename source_link_t, typename updater_t,
            typename outlier_finder_t, typename calibrator_t,
            typename smoother_t /*, typename parameters_t*/>
  struct Actor {
    /// Enforce default construction
    Actor() = default;

    /// Broadcast the result_type
    using result_type = Result<source_link_t>;

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

      /// Wether to transport covariance
      bool doCovTransport = true;

      /// Whether to consider multiple scattering.
      bool multipleScattering = true;

      /// Whether to consider energy loss.
      bool energyLoss = true;

      /// Target surface for reverse filtering
      const Surface *targetSurface;
    } m_config;

    /// Configurable components:
    updater_t m_updater;
    outlier_finder_t m_outlierFinder;
    calibrator_t m_calibrator;
    smoother_t m_smoother;

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

      // Check if we have a surface
      if (not state.navigation.currentSurface) {
        ACTS_VERBOSE("GSF actor: no surface, fast return");
        return;
      }

      const auto& surface = *state.navigation.currentSurface;

      ACTS_VERBOSE("GSF actor on surface with geoID "
                   << surface.geometryId() << ", "
                   << surface.geometryId().value());

      // Check if we have a measurement
      const auto sourcelink_it =
          m_config.inputMeasurements.find(surface.geometryId());

      if (sourcelink_it == m_config.inputMeasurements.end()) {
        ACTS_VERBOSE("GSF actor: no measurement for surface, fast return");
        return;
      }

      const auto& measurement = sourcelink_it->second;

      // If the MultiTrajectory is empty we are at the start (hopefully), so we
      // set all tips to SIZE_MAX as specified by the MultiTrajectory class
      if (result.currentTips.empty()) {
        result.currentTips.resize(stepper.numberComponents(state.stepping),
                                  SIZE_MAX);
      }

      // We assume that the number of components did not change during stepping,
      // but only here in the gsf actor
      if (stepper.numberComponents(state.stepping) !=
          result.currentTips.size()) {
        std::stringstream msg;
        msg << "GSF actor: Number of components mismatch: currentTips = "
            << result.currentTips.size()
            << ", numComponents = " << stepper.numberComponents(state.stepping);
        throw std::runtime_error(msg.str());
      }

      ACTS_VERBOSE("GSF actor: " << stepper.numberComponents(state.stepping)
                                 << " at the start");

      filter(state, stepper, result, measurement);
    }

    /// @brief A filtering step
    template <typename propagator_state_t, typename stepper_t>
    void filter(propagator_state_t& state, const stepper_t& stepper,
                result_type& result, const source_link_t& measurement) const {
      const auto& logger = state.options.logger;
      const auto& surface = *state.navigation.currentSurface;

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
      kalmanUpdate(state, result, measurement, componentCache);

      // Reweigth components according to measurement
      detail::reweightComponents(componentCache);

      // update stepper state with the collected component caches
      stepper.updateComponents(state.stepping, componentCache, surface);

      // Add the combined track state to results
      result.combinedTrackParameters.push_back(
          detail::combineMultiComponentState(componentCache, surface));

      ACTS_VERBOSE("GSF actor: combined multicomponent state");

      // At the moment finished when everything is done
      if (result.measurementStates == m_config.inputMeasurements.size()) {
        ACTS_VERBOSE(
            "GSF actor: finished, because of number of states reached number "
            "of measurements");
        result.isFinished = true;
      }

      ACTS_VERBOSE("GSF actor: " << stepper.numberComponents(state.stepping)
                                 << " at the end");
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

        std::cout << "mixture.size() = " << mixture.size() << "\n";

        // Create all possible new components
        for (const auto& gaussian : mixture) {
          std::cout << "create gaussian component TTTTTTTTTTTTTT\n";
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

          ACTS_VERBOSE("GSF actor: created new component with pars "
                       << new_cmp.predictedPars.transpose());

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
    Acts::Result<void> kalmanUpdate(
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

      return Acts::Result<void>::success();
    }

    // TODO not clear if smoothing would work this way, see "track fitting with
    // non-gaussian noise, R. frühwirth"
    //     /// @brief Kalman actor operation : finalize
    //     ///
    //     /// @tparam propagator_state_t is the type of Propagagor state
    //     /// @tparam stepper_t Type of the stepper
    //     ///
    //     /// @param state is the mutable propagator state object
    //     /// @param stepper The stepper in use
    //     /// @param result is the mutable result state object
    //     template <typename propagator_state_t, typename stepper_t>
    //     Acts::Result<void> finalize(propagator_state_t& state,
    //                                 const stepper_t& /*stepper*/,
    //                                 result_type& result) const {
    //       const auto& logger = state.options.logger;
    //
    //       ACTS_VERBOSE("GSF actor: Start smoothing");
    //
    //       // Remember you smoothed the track states TODO why this is set here
    //       // (copied from Kalman filter)
    //       result.isSmoothed = true;
    //
    //       // Copy to new MultiTrajectory object in a way, that there are no
    //       shared
    //       // points between the components. Furthermore, we copy only if
    //       there is a
    //       // measurement.
    //       std::vector<size_t> newTips(result.currentTips.size(), SIZE_MAX);
    //       decltype(result.fittedStates) newMultitrajectory;
    //
    //       for (auto i = 0ul; i < result.currentTips.size(); ++i) {
    //         result.fittedStates.visitBackwards(
    //             result.currentTips[i], [&](auto proxy) {
    //               if
    //               (proxy.typeFlags().test(TrackStateFlag::MeasurementFlag)) {
    //                 newTips[i] = newMultitrajectory.addTrackState(
    //                     TrackStatePropMask::All, newTips[i]);
    //
    //                 auto newProxy =
    //                 newMultitrajectory.getTrackState(newTips[i]);
    //                 newProxy.copyFrom(proxy);
    //               }
    //             });
    //       }
    //
    //       // Smooth the track states TODO due to the above the tracks will be
    //       // reversed, is this a problem for the smoothing?
    //       for (auto tip : newTips) {
    //         auto smoothRes =
    //             m_smoother(state.geoContext, newMultitrajectory, tip,
    //             logger);
    //         if (!smoothRes.ok()) {
    //           ACTS_ERROR("GSF actor: Smoothing step failed: " <<
    //           smoothRes.error()); return smoothRes.error();
    //         }
    //       }
    //
    //       ACTS_VERBOSE("GSF actor: Smoothing done!");
    //
    //       // In case the loop ran through without an error, we can report
    //       success return Acts::Result<void>::success();
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
    bool operator()(propagator_state_t& /*state*/, const stepper_t& /*stepper*/,
                    const result_t& result) const {
      if (result.isFinished && result.isSmoothed) {
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
  auto fit(
      const std::vector<source_link_t>& sourcelinks,
      const start_parameters_t& sParameters,
      const GSFOptions<calibrator_t, outlier_finder_t>& options) const {
    const auto& logger = options.logger;
    static_assert(SourceLinkConcept<source_link_t>,
                  "Source link does not fulfill SourceLinkConcept");

    // To be able to find measurements later, we put them into a map
    // We need to copy input SourceLinks anyways, so the map can own them.
    ACTS_VERBOSE("Preparing " << sourcelinks.size() << " input measurements");
    std::map<GeometryIdentifier, source_link_t> inputMeasurements;
    for (const auto& sl : sourcelinks) {
      inputMeasurements.emplace(sl.geometryId(), sl);
    }

    ACTS_VERBOSE("Final measuerement map size: " << inputMeasurements.size());

    // Create the ActionList and AbortList
    using GSFActor = Actor<source_link_t, GainMatrixUpdater, outlier_finder_t,
                           calibrator_t, GainMatrixSmoother>;
    using GSFAborter = Aborter<GSFActor>;
    using GSFResult = Result<source_link_t>;

    using Actors = ActionList<GSFActor, MultiSteppingLogger>;
    using Aborters = AbortList<GSFAborter, EndOfWorldReached>;

    // Create relevant options for the propagation options
    PropagatorOptions<Actors, Aborters> propOptions(
        options.geoContext, options.magFieldContext, logger);

    // Catch the actor and set the measurements
    auto& actor = propOptions.actionList.template get<GSFActor>();
    actor.m_config.inputMeasurements = inputMeasurements;
    actor.m_config.maxComponents = 16;
    actor.m_calibrator = options.calibrator;
    actor.m_outlierFinder = options.outlierFinder;

    // Make Single Start Parameters to Multi Start parameters
    throw_assert(sParameters.covariance() != std::nullopt,
                 "we need a covariance here...");
    MultiComponentBoundTrackParameters<SinglyCharged> sMultiPars(
        sParameters.referenceSurface().getSharedPtr(), sParameters.parameters(),
        sParameters.covariance());

    // Run the fitter forward direction
    ACTS_VERBOSE("Run forward pass");
    propOptions.direction = NavigationDirection::forward;
    auto forwardResult = m_propagator.propagate(sMultiPars, propOptions);

    // Use last point in trajectory as start point
    // TODO at the moment we cannot start with a multi component state, since we
    // do not have the weights here
    auto lastParams = (*forwardResult)
                          .template get<GSFResult>()
                          .combinedTrackParameters.back();
    MultiComponentBoundTrackParameters<SinglyCharged> backwardsStartParameters(
        lastParams.referenceSurface().getSharedPtr(), lastParams.parameters(),
        lastParams.covariance());

    // Run the fitter backwards again
    ACTS_VERBOSE("Run backwards pass");
    propOptions.direction = NavigationDirection::backward;
    auto backwardsResult =
        m_propagator.propagate(backwardsStartParameters, propOptions);

    // Combine the two
    const auto forwardTrack =
        (*forwardResult).template get<GSFResult>().combinedTrackParameters;
    auto backwardTrack =
        (*backwardsResult).template get<GSFResult>().combinedTrackParameters;
    std::reverse(backwardTrack.begin(), backwardTrack.end());

    const auto weightedMeanSmoothed =
        detail::combineForwardAndBackwardPass(forwardTrack, backwardTrack);

    return forwardResult;
  }
};

}  // namespace Acts
