// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/Material/ISurfaceMaterial.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include <Acts/EventData/MultiTrajectoryHelpers.hpp>

#include <map>
#include <numeric>

#include "BetheHeitlerApprox.hpp"

namespace Acts {

// template <typename propagator_t>
struct GaussianSumFitter {
  /// struct to store the data related to a component during processing
  struct ComponentCache {
    std::size_t parentMultiTrajectoryIndex;
    BoundVector pars;
    std::optional<BoundSymMatrix> cov;
    BoundMatrix jacobian;
    BoundToFreeMatrix jacToGlobal;
    FreeMatrix jacTransport;
    FreeVector derivative;
    ActsScalar pathLength;
    ActsScalar weight;
    FreeVector filteredPars;
    std::optional<BoundSymMatrix> filteredCov;
  };

  template <typename source_link_t>
  struct Result {
    /// The multi-trajectory which stores the graph of components
    MultiTrajectory<source_link_t> fittedStates;

    /// The current indexes for the active components in the multi trajectory
    std::vector<std::size_t> currentTips;

    /// The number of measurement states created
    int measurementStates = 0;

    /// A std::vector storing the component caches so that it must not be
    /// reallocated every pass
    std::vector<ComponentCache> componentCache;
  };

  template <
      typename source_link_t, typename updater_t, typename outlier_finder_t,
      typename calibrator_t /*, typename parameters_t, typename smoother_t*/>
  struct Actor {
    /// Enforce default construction
    Actor() = default;

    /// Broadcast the result_type
    using result_type = Result<source_link_t>;

    // Actor configuration
    struct Config {
      /// Maximum number of components which the GSF should handle
      std::size_t maxComponents;

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

      /// Whether run reversed filtering
      bool reversedFiltering = false;
    } m_config;

    /// Configurable components:
    updater_t m_updater;
    outlier_finder_t m_outlierFinder;
    calibrator_t m_calibrator;

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
        throw_assert(&surface == state.navigation.startSurface, "");
        result.currentTips.resize(stepper.numberComponents(state.stepping),
                                  SIZE_MAX);
      }

      if (stepper.numberComponents(state.stepping) !=
          result.currentTips.size()) {
        throw std::runtime_error("GSF: Number of components mismatch");
      }

      // Apply material effect by creating new components
      create_new_components(state.stepping, stepper, result.componentCache,
                            result.currentTips, surface);

      // Reduce the number of components to a managable degree
      if (result.componentCache.size() > m_config.maxComponents ||
          result.componentCache.size()  > stepper.maxComponents) {
        reduceNumberOfComponents(
            result.componentCache, std::min(static_cast<int>(m_config.maxComponents),
                                    stepper.maxComponents));
      }

      // Update the Multi trajectory with the new components
      updateMultiTrajectoryAndDoKalmanUpdate(state, stepper, result,
                                             measurement, result.componentCache);
    }

    /// @brief Expands all existing components to new components by using a
    /// gaussian-mixture approximation for the Bethe-Heitler distribution.
    ///
    /// @return a std::vector with all new components (parent tip, weight,
    /// parameters, covariance)
    template <typename stepper_state_t, typename stepper_t>
    void create_new_components(
        stepper_state_t& stepping, const stepper_t& stepper,
        std::vector<ComponentCache>& new_components,
        const std::vector<std::size_t>& parentTrajectoryIdxs,
        const Surface& surface) const {
      // Some shortcuts
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

          new_cmp.pars[eBoundQOverP] = old_bound.charge() / (p_prev + delta_p);

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

            new_cmp.cov = old_bound.covariance();
            (*new_cmp.cov)(eBoundQOverP, eBoundQOverP) += varInvP;
          }

          // Here we combine the new child weight with the parent weight. when
          // done correctely, the sum of the weights should stay at 1
          new_cmp.weight = gaussian.weight * old_cmp.weight();

          // Set the remaining things and push to vector
          new_cmp.jacobian = jac;
          new_cmp.pathLength = pathLength;
          new_cmp.jacToGlobal = old_cmp.jacToGlobal();
          new_cmp.derivative = old_cmp.derivative();
          new_cmp.jacTransport = old_cmp.jacTransport();
          new_cmp.parentMultiTrajectoryIndex = parentTrajectoryIdxs[i];

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

    template <typename propagator_state_t, typename stepper_t>
    Acts::Result<void> updateMultiTrajectoryAndDoKalmanUpdate(
        propagator_state_t& state, const stepper_t& stepper,
        result_type& result, const source_link_t measurement,
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
            TrackStatePropMask::All, cmp.parentMultiTrajectoryIndex));

        auto trackProxy =
            result.fittedStates.getTrackState(result.currentTips.back());

        // Set surface
        trackProxy.setReferenceSurface(surface.getSharedPtr());

        // assign the source link to the track state
        trackProxy.uncalibrated() = measurement;

        // Set track parameters
        trackProxy.predicted() = std::move(cmp.pars);
        if (cmp.cov) {
          trackProxy.predictedCovariance() = std::move(*cmp.cov);
        }

        trackProxy.jacobian() = std::move(cmp.jacobian);
        trackProxy.pathLength() = std::move(cmp.pathLength);

        // We have predicted parameters, so calibrate the uncalibrated input
        // measuerement TODO is std::visit necessary here (didn't compile, so
        // removed it)
        // TODO Something is not working here
        //         auto calibrated = m_calibrator(trackProxy.uncalibrated(),
        //         trackProxy.predicted());
        //
        //         if( !calibrated.ok() )
        //             return calibrated.error();
        //
        //         trackProxy.setCalibrated(calibrated.value());

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

          // TODO ACTS_VERBOSE message

          trackProxy.typeFlags().set(TrackStateFlag::MeasurementFlag);

          // update component cache using filtered parameters after
          cmp.filteredPars = MultiTrajectoryHelpers::freeFiltered(
              state.options.geoContext, trackProxy);
          if (m_config.doCovTransport) {
            cmp.filteredCov = trackProxy.filteredCovariance();
          }

          // We count the state with measurement
          ++result.measurementStates;
        } else {
          // TODO ACTS_VERBOSE message

          // Set the outlier type flag
          trackProxy.typeFlags().set(TrackStateFlag::OutlierFlag);
          trackProxy.data().ifiltered = trackProxy.data().ipredicted;
        }
      }

      // update stepper state with the collected component caches
      stepper.updateComponents(state.stepping, new_components, surface);

      return Acts::Result<void>::success();
    }
  };
};

}  // namespace Acts
