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

#include <map>
#include <numeric>

#include "BetheHeitlerApprox.hpp"

namespace Acts {

// template <typename propagator_t>
struct GaussianSumFitter {
  template <typename source_link_t>
  struct Result {
    /// The multi-trajectory which stores the graph of components
    MultiTrajectory<source_link_t> fittedStates;

    std::vector<std::size_t> lastActiveComponentTips;
  };

  template <typename source_link_t, typename updater_t/*, typename parameters_t, typename calibrator_t,
          typename outlier_finder_t, typename smoother_t*/>
  struct Actor {
    /// Broadcast the result_type
    using result_type = Result<source_link_t>;

    /// How to represent a component internally TODO resolve dependence on
    /// EigenStepper<>
    using ComponentRep = std::tuple<ActsScalar, EigenStepper<>::BoundState>;

    // Actor configuration
    struct Config {
      /// Maximum number of components which the GSF should handle
      std::size_t maxComponents;

      /// Input measurements
      std::map<GeometryIdentifier, source_link_t> inputMeasurements;

      /// Bethe Heitler Approximator
      detail::BHApprox bethe_heitler_approx =
          detail::BHApprox(detail::bh_cmps6_order5_data);

      /// Whether to consider multiple scattering.
      bool multipleScattering = true;

      /// Whether to consider energy loss.
      bool energyLoss = true;

      /// Whether run reversed filtering
      bool reversedFiltering = false;
    } m_config;
    
    updater_t m_updater;

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
          m_config.inputMeasurements->find(surface->geometryId());
          
      if( sourcelink_it == m_config.inputMeasurements.end() ) {
        ACTS_VERBOSE("GSF actor: no measurement for surface, fast return");
        return;
      }
      
      const auto &measurement = sourcelink_it->second;

      // TODO do we need the whole bound state here? Think about later...
      const auto components =
          stepper.extractComponents(state.stepping, surface, true);

      // If the MultiTrajectory is empty assume we are at the start, and we
      // initialize the MultiTrajectory and exit. TODO is this a good
      // assumtion? Are there cases where startSurface is not set?
      if (result.lastActiveComponentTips.empty()) {
        throw_assert(surface == state.navigation.startSurface, "");

        initialize_multi_trajectory(components, measurement, result.lastActiveComponentTips,
                                    result.fittedStates);
        
        // TODO do we need a kalman update on the first step?
        return;
      }

      if (components.size() != result.lastActiveComponentTips.size()) {
        throw std::runtime_error("GSF: Number of components mismatch");
      }

      // Apply material effect by creating new components
      auto newComponents = create_new_components(
          components, result.lastActiveComponentTips, surface,
          state.stepping.navDir, state.stepping.geoContext);

      // Reduce the number of components to a managable degree
      if (newComponents.size() > m_config.maxComponents ||
          newComponents.size() > stepper.maxComponents) {
        reduceNumberOfComponents(
            newComponents, std::min(static_cast<int>(m_config.maxComponents),
                                    stepper.maxComponents));
      }

      // Update the Multi trajectory with the new components
      updateMultiTrajectory(result.lastActiveComponentTips, result.fittedStates,
                            surface, newComponents);
      
    }
    
    template <typename propagator_state_t, typename stepper_t>
    void kalmanUpdate(propagator_state_t &state, const stepper_t &stepper)
    {
        
      // Do the Kalman update
              if (not m_outlierFinder(trackStateProxy)) {
          // Run Kalman update
          auto updateRes = m_updater(state.geoContext, trackStateProxy,
                                     state.stepping.navDir, logger);
          if (!updateRes.ok()) {
            ACTS_ERROR("Update step failed: " << updateRes.error());
            return updateRes.error();
          }
          // Set the measurement type flag
          typeFlags.set(TrackStateFlag::MeasurementFlag);
          // Update the stepping state with filtered parameters
          ACTS_VERBOSE("Filtering step successful, updated parameters are : \n"
                       << trackStateProxy.filtered().transpose());
          // update stepping state using filtered parameters after kalman
          stepper.update(state.stepping,
                         MultiTrajectoryHelpers::freeFiltered(
                             state.options.geoContext, trackStateProxy),
                         trackStateProxy.filteredCovariance());
          // We count the state with measurement
          ++result.measurementStates;
        } else {
          ACTS_VERBOSE(
              "Filtering step successful. But measurement is deterimined "
              "to "
              "be an outlier. Stepping state is not updated.")
          // Set the outlier type flag
          typeFlags.set(TrackStateFlag::OutlierFlag);
          trackStateProxy.data().ifiltered = trackStateProxy.data().ipredicted;
        }
    }

    /// @brief initializes the MultiTrajectory with the components from the
    /// stepper state
    void initialize_multi_trajectory(
        const std::vector<std::optional<ComponentRep>>& comps,
        const source_link_t &measurement,
        const Surface& currentSurface, std::vector<std::size_t>& currentTips,
        MultiTrajectory<source_link_t>& multitraj) {
      currentTips.clear();

      for (const auto& cmp : comps) {
        if (!cmp) {
          continue;
        }

        // Create Track State
        currentTips.push_back(multitraj.addTrackState());
        auto trackProxy = multitraj.getTrackState(currentTips.back());

        // Set surface
        trackProxy.setReferenceSurface(currentSurface.getSharedPtr());
        
        // Set measurement TODO do we need this here?
        trackProxy.uncalibrated() = measurement;

        // Set prediction from stepper
        const auto& [parent_weight, parent_bound_state] = *cmp;
        const auto& [parent_bound, jac, pathLength] = parent_bound_state;

        trackProxy.predicted() = std::move(parent_bound.parameters());
        if (parent_bound.covariance()) {
          trackProxy.predictedCovariance() =
              std::move(*parent_bound.covariance());
        }

        trackProxy.jacobian() = std::move(jac);
        trackProxy.pathLength() = std::move(pathLength);
      }
    }

    /// @brief Expands all existing components to new components by using a
    /// gaussian-mixture approximation for the Bethe-Heitler distribution.
    ///
    /// @return a std::vector with all new components (parent tip, weight,
    /// parameters, covariance)
    std::vector<std::tuple<std::size_t, ComponentRep>> create_new_components(
        const std::vector<std::optional<ComponentRep>>& comps,
        const std::vector<std::size_t>& parentTrajectoryTips,
        const Surface& surface, const NavigationDirection navDir,
        const GeometryContext& gctx) const {
      std::vector<std::tuple<std::size_t, ComponentRep>> new_components;

      const auto surfaceMaterial = surface.surfaceMaterial();

      for (auto i = 0ul; i < comps.size(); ++i) {
        // Approximate bethe-heitler distribution as gaussian mixture
        const auto& [parent_weight, parent_bound_state] = *comps[i];
        const auto [parent_bound, jac, pathLength] =
            std::move(parent_bound_state);
        const auto p_prev = parent_bound.absoluteMomentum();
        const auto slab =
            surfaceMaterial->materialSlab(parent_bound.position(gctx), navDir,
                                          MaterialUpdateStage::fullUpdate);

        const auto mixture =
            m_config.bethe_heitler_approx.mixture(slab.thicknessInX0());

        for (const auto& gaussian : mixture) {
          BoundVector new_pars = parent_bound.parameters();
          BoundSymMatrix new_cov;

          // compute delta p from mixture and update parameters
          const auto delta_p = [&]() {
            if (navDir == NavigationDirection::forward)
              return p_prev * (gaussian.mean - 1.);
            else
              return p_prev * (1. / gaussian.mean - 1.);
          }();

          throw_assert(p_prev + delta_p > 0.,
                       "new momentum after bethe-heitler must be > 0");

          new_pars[eBoundQOverP] = parent_bound.charge() / (p_prev + delta_p);

          // compute inverse variance of p from mixture and update covariance
          if (parent_bound.covariance()) {
            const auto varInvP = [&]() {
              if (navDir == NavigationDirection::forward) {
                const auto f = 1. / (p_prev * gaussian.mean);
                return f * f * gaussian.var;
              } else {
                return gaussian.var / (p_prev * p_prev);
              }
            }();

            new_cov(eBoundQOverP, eBoundQOverP) += varInvP;
          }

          // Here we combine the new child weight with the parent weight. when
          // done correctely, the sum of the weights should stay at 1
          const auto new_weight = gaussian.weight * parent_weight;
          const BoundTrackParameters new_bound(surface.getSharedPtr(), new_pars,
                                               new_cov);

          new_components.push_back(
              {parentTrajectoryTips[i],
               {new_weight, {new_bound, jac, pathLength}}});
        }
      }

      return new_components;
    }

    /// @brief Function that reduces the number of components. at the moment,
    /// this is just by erasing the components with the lowest weights.
    /// Finally, the components are reweighted so the sum of the weights is
    /// still 1
    void reduceNumberOfComponents(
        std::vector<std::tuple<std::size_t, ComponentRep>>& components,
        std::size_t numberOfRemainingCompoents) const {
      // The elements should be sorted by weight (high -> low)
      std::sort(begin(components), end(components),
                [](const auto& a, const auto& b) {
                  return std::get<ActsScalar>(std::get<1>(a)) >
                         std::get<ActsScalar>(std::get<1>(b));
                });

      // Remove elements by resize
      components.resize(numberOfRemainingCompoents);

      // Reweight after removal
      const auto sum_of_weights =
          std::accumulate(begin(components), end(components), 0.0,
                          [](auto sum, const auto& cmp) {
                            return sum + std::get<double>(cmp);
                          });

      for (auto& cmp : components) {
        std::get<ActsScalar>(std::get<1>(cmp)) /= sum_of_weights;
      }
    }

    void updateMultiTrajectory(
        std::vector<std::size_t>& currentTips,
        MultiTrajectory<source_link_t>& multitraj,
        const Surface& surface,
        const std::vector<std::tuple<std::size_t, ComponentRep>>&
            new_components) {
      currentTips.clear();

      for (const auto& [parentTip, cmpRep] : new_components) {
        const auto& [weight, bound_state] = cmpRep;

        // Create new track state
        currentTips.push_back(
            multitraj.addTrackState(TrackStatePropMask::All, parentTip));

        auto trackProxy = multitraj.getTrackState(currentTips.back());

        // Set surface
        trackProxy.setReferenceSurface(surface.getSharedPtr());

        // Set track parameters
        const auto& [parent_bound, jac, pathLength] = bound_state;

        trackProxy.predicted() = std::move(parent_bound.parameters());
        if (parent_bound.covariance()) {
          trackProxy.predictedCovariance() =
              std::move(*parent_bound.covariance());
        }

        trackProxy.jacobian() = std::move(jac);
        trackProxy.pathLength() = std::move(pathLength);
    
        // Get and set the type flags
        trackProxy.typeFlags().set(TrackStateFlag::ParameterFlag);
        if (surface.surfaceMaterial() != nullptr) {
          trackProxy.typeFlags().set(TrackStateFlag::MaterialFlag);
        }
      }
    }
  };
};

}  // namespace Acts
