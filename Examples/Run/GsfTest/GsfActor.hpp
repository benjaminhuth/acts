// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Material/IVolumeMaterial.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Surfaces/Surface.hpp"

#include <map>

#include "BetheHeitlerApprox.hpp"
#include "MultiEigenStepperCommon.hpp"

namespace Acts {

// template <typename source_link_t, typename parameters_t, typename
// calibrator_t,
//           typename outlier_finder_t, typename updater_t, typename smoother_t>
class GsfActor {
 public:
  struct Config {
    std::size_t maxComponents;

    /// Whether to consider multiple scattering.
    bool multipleScattering = true;

    /// Whether to consider energy loss.
    bool energyLoss = true;

    /// Whether run reversed filtering
    bool reversedFiltering = false;
  };

  struct Result {
    // nothinge here so far
  };

 private:
  //   /// The Kalman updater
  //   updater_t m_updater;
  //
  //   /// The Kalman smoother
  //   smoother_t m_smoother;
  //
  //   /// The measurement calibrator
  //   calibrator_t m_calibrator;
  //
  //   /// The outlier finder
  //   outlier_finder_t m_outlierFinder;

  /// The Surface beeing
  //   SurfaceReached targetReached;

  /// Configuration
  Config m_config;

  /// Bethe Heitler Approximator
  detail::BHApprox m_bethe_heitler_approx =
      detail::BHApprox(detail::bh_cmps6_order5_data);

 public:
  /// Broadcast the result_type
  using result_type = Result;

  /// The target surface
  const Surface* targetSurface = nullptr;

  /// Allows retrieving measurements for a surface
  //   const std::map<GeometryIdentifier, source_link_t>* inputMeasurements =
  //       nullptr;

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
                  result_type& /*result*/) const {
    const auto& logger = state.options.logger;

    if (not state.navigation.currentSurface ||
        not state.navigation.currentVolume) {
      ACTS_VERBOSE(
          "GSF actor: no currentSurface or currentVolume, fast return");
      return;
    }

    const auto volumeMaterial =
        state.navigation.currentVolume->volumeMaterial();
    const auto components = stepper.extractComponents(
        state.stepping, *state.navigation.currentSurface, true);

    auto newComponents =
        create_new_components(components, *volumeMaterial,
                              state.stepping.navDir, state.stepping.geoContext);

    if (newComponents.size() > m_config.maxComponents ||
        newComponents.size() > stepper.maxComponents) {
      combineComponents(newComponents,
                        std::min(static_cast<int>(m_config.maxComponents),
                                 stepper.maxComponents));
    }
  }

  auto create_new_components(
      const std::vector<detail::CommonComponentRep>& comps,
      const IVolumeMaterial& volumeMaterial, const NavigationDirection navDir,
      const GeometryContext& gctx) const
      -> std::vector<std::tuple<double, BoundVector, BoundSymMatrix>> {
    std::vector<std::tuple<double, BoundVector, BoundSymMatrix>> new_components;

    for (const auto& c : comps) {
      // Approximate bethe-heitler distribution as gaussian mixture
      const auto& bound = std::get<BoundTrackParameters>(c.boundState);
      const auto p_prev = bound.absoluteMomentum();
      const auto material = volumeMaterial.material(bound.position(gctx));
      const Acts::MaterialSlab slab(material, c.pathLengthSinceLast);

      const auto mixture = m_bethe_heitler_approx.mixture(slab.thicknessInX0());

      for (const auto& gaussian : mixture) {
        BoundVector new_pars = bound.parameters();
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

        new_pars[eBoundQOverP] = bound.charge() / (p_prev + delta_p);

        // compute inverse variance of p from mixture and update covariance
        if (bound.covariance()) {
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
        new_components.push_back(
            {gaussian.weight * c.weight, new_pars, new_cov});
      }
    }

    return new_components;
  }

  /// @brief Function that reduces the number of components. at the moment, this
  /// is just by erasing the components with the lowest weights. Finally, the components are reweighted so the sum of the weights is still 1
  void combineComponents(
      std::vector<std::tuple<double, BoundVector, BoundSymMatrix>>& components,
      std::size_t numberOfRemainingCompoents) const {
    // The elements should be ordered high -> low
    std::sort(begin(components), end(components),
              [](const auto& a, const auto& b) {
                return std::get<double>(a) > std::get<double>(b);
              });
    components.resize(numberOfRemainingCompoents);

    const auto sum_of_weights = std::accumulate(
        begin(components), end(components), 0.0,
        [](auto sum, const auto& cmp) { return sum + std::get<double>(cmp); });

    for (auto& cmp : components)
      std::get<double>(cmp) /= sum_of_weights;
  }
}
};

}  // namespace Acts
