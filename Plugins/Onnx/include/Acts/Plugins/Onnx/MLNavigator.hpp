// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Plugins/Onnx/PairwiseScoreModel.hpp"
#include "Acts/Plugins/Onnx/TargetPredModel.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <fstream>
#include <map>
#include <set>

using namespace std::string_literals;

namespace Acts {

/// A navigator which uses a neural network and a graph of surfaces to estimate
/// the next surface. The neural network computes a score [0,1] from a pair of
/// surfaces (represented by their 3D center coordinates) and the current track
/// parameters (direction, qop). If all candidates from the graph are scored,
/// they are sorted by score and one after another is tried as a target surface.
/// The Beampipe-Surface is splitted along the z-axis
template <typename NavigationModel>
class MLNavigator {
  std::shared_ptr<NavigationModel> m_model;
  std::shared_ptr<const Acts::TrackingGeometry> m_tgeo;
  std::set<const Acts::Surface *> m_start_surfaces;

 public:
  struct State {
    std::vector<Acts::SurfaceIntersection> navSurfaces;
    std::vector<Acts::SurfaceIntersection>::const_iterator navSurfaceIter;
    const Acts::Surface *currentSurface;
    const Acts::Surface *startSurface;

    bool navigationBreak = false;
    bool targetReached = false;

    const Acts::TrackingVolume *currentVolume = nullptr;
    const Acts::Surface *targetSurface = nullptr;
  };

  MLNavigator() { throw_assert(false, "Don't use this constructor!"); }

  /// Constructor of the ML Navigator
  MLNavigator(std::shared_ptr<NavigationModel> model,
              std::shared_ptr<const Acts::TrackingGeometry> tgeo,
              const std::set<const Acts::Surface *> &start_surfaces)
      : m_model(model), m_tgeo(tgeo), m_start_surfaces(start_surfaces) {}

  /// The whole purpose of this function is to set currentSurface, if possible
  template <typename propagator_state_t, typename stepper_t>
  void status(propagator_state_t &state, const stepper_t &stepper) const {
    try {
      const auto &logger = state.options.logger;
      ACTS_VERBOSE(">>>>>>>> STATUS <<<<<<<<<");

      if (state.navigation.navigationBreak)
        return;

      // Handle initialization
      if (state.navigation.navSurfaces.empty()) {
        ACTS_VERBOSE(
            "We have no navSurfaceIter, so are during intialization hopefully");

        // TODO This is hacky, but for now assume we start at beamline
        state.navigation.currentSurface = m_tgeo->getBeamline();
        state.navigation.currentVolume = m_tgeo->highestTrackingVolume();
        return;
      }

      // Navigator status always resets the current surface
      state.navigation.currentSurface = nullptr;

      // Establish the surface status
      auto surfaceStatus = stepper.updateSurfaceStatus(
          state.stepping, *state.navigation.navSurfaceIter->representation,
          false);

      if (surfaceStatus == Acts::Intersection3D::Status::onSurface) {
        // Set the current surface
        state.navigation.currentSurface =
            state.navigation.navSurfaceIter->representation;
        ACTS_VERBOSE("Current surface set to  "
                     << state.navigation.currentSurface->geometryId());

        // Release Stepsize
        ACTS_VERBOSE("Release Stepsize");
        stepper.releaseStepSize(state.stepping);

        // Reset state
        state.navigation.navSurfaces.clear();
        state.navigation.navSurfaceIter = state.navigation.navSurfaces.end();

        // Check if we can navigate further
        if (m_start_surfaces.find(state.navigation.currentSurface) ==
            m_start_surfaces.end()) {
          ACTS_VERBOSE(
              "The current surface was not found in graph, so we stop the "
              "navigation here!");
          state.navigation.navigationBreak = true;
          state.navigation.currentVolume = nullptr;
        } else {
          ACTS_VERBOSE(
              "The current Surface was found in the graph, so we can go on.");
        }
      } else if (surfaceStatus == Acts::Intersection3D::Status::reachable) {
        ACTS_VERBOSE("Next surface reachable at distance  "
                     << stepper.outputStepSize(state.stepping));
      } else {
        ACTS_VERBOSE(
            "Surface unreachable or missed, hopefully the target(...) call "
            "fixes this");
      }
    } catch (std::exception &e) {
      throw std::runtime_error("Error in MLNavigator::status - "s + e.what());
    }
  }

  template <typename propagator_state_t, typename stepper_t>
  void target(propagator_state_t &state, const stepper_t &stepper) const {
    const auto &logger = state.options.logger;

    // TrackML detector aborter
    auto x = state.stepping.pars[eFreePos0];
    auto y = state.stepping.pars[eFreePos1];
    auto z = state.stepping.pars[eFreePos2];
    auto r = std::sqrt(x*x + y*y);
    
    if( std::abs(z) > 3200 || r > 1200 )
    {
        state.navigation.navigationBreak = true;
        state.navigation.currentVolume = nullptr;
    }
    
    try {
      ACTS_VERBOSE(">>>>>>>> TARGET <<<<<<<<<");
      auto &navstate = state.navigation;

      // Predict new targets if there are no candidates in state
      if (navstate.navSurfaceIter == navstate.navSurfaces.end()) {
        // This means also we are currently on a surface
        assert(navstate.currentSurface != nullptr);
        ACTS_VERBOSE(
            "It seems like we are on a surface and must predict new targets");

        navstate.navSurfaces = predict_next(state, stepper);
        navstate.navSurfaceIter = navstate.navSurfaces.begin();
      }

      // It seems like we are done
      if (navstate.navigationBreak) {
        ACTS_VERBOSE("No target Surface, job done.");
        return;
      }

      // Check if we are in a correct state
      assert(!state.navigation.navSurfaces.empty());
      assert(state.navigation.navSurfaceIter !=
             state.navigation.navSurfaces.end());

      // Navigator target always resets the current surface
      // It is set later by the status call if possible
      navstate.currentSurface = nullptr;

      ACTS_VERBOSE("Ask for SurfaceStatus of currently most probable target");

      // Establish & update the surface status
      auto surfaceStatus = stepper.updateSurfaceStatus(
          state.stepping, *navstate.navSurfaceIter->representation, false);

      // Everything OK
      if (surfaceStatus == Acts::Intersection3D::Status::reachable) {
        ACTS_VERBOSE("Navigation stepSize set to "
                     << stepper.outputStepSize(state.stepping));
      }
      // Try another surface
      else {
        ACTS_VERBOSE(
            "Surface not reachable, search another one which is "
            "reachable");

        // Search in the current surface pool
        for (; navstate.navSurfaceIter != navstate.navSurfaces.end();
             ++navstate.navSurfaceIter) {
          if (stepper.updateSurfaceStatus(
                  state.stepping, *navstate.navSurfaceIter->representation,
                  false) == Intersection3D::Status::reachable)
            break;
        }

        // If we did not find the surface, stop the navigation
        if (navstate.navSurfaceIter == navstate.navSurfaces.end()) {
          ACTS_ERROR("Could not find the surface, stop navigation!");
          navstate.currentVolume = nullptr;
          navstate.navigationBreak = true;
        } else {
          ACTS_VERBOSE(
              "Found a reachable surface "
              << navstate.navSurfaceIter->representation->geometryId());
        }
      }
    } catch (std::exception &e) {
      throw std::runtime_error("Error in MLNavigator::target() - "s + e.what());
    }
  }

  /// @brief Performs prediction and processes the resulting std::vector of
  /// surfaces
  /// @return Sorted std::vector of SurfaceIntersections
  template <typename propagator_state_t, typename stepper_t>
  auto predict_next(const propagator_state_t &state,
                    const stepper_t &stepper) const {
    const auto &logger = state.options.logger;
    
    // Do prediction
    const std::vector<const Surface *> predictions =
        m_model->predict_next(state, stepper);

    // Important parameters
    const FreeVector &params = state.stepping.pars;
    const GeometryContext &gctx = state.geoContext;
//     const double olimit = stepper.overstepLimit(state.stepping);

    // Make intersection objects
    std::vector<SurfaceIntersection> sfis(predictions.size());
    std::transform(
        predictions.begin(), predictions.end(), sfis.begin(), [&](auto s) {
          return s->intersect(
              gctx, params.segment<3>(eFreePos0),
              state.stepping.navDir * params.segment<3>(eFreeDir0), true);
        });
    
    ACTS_VERBOSE("predict_next: have " << sfis.size() << " candidate intersections");

    sfis.erase(std::remove_if(sfis.begin(), sfis.end(),
                                [](auto sfi) {
//                                 auto sfil = sfi.intersection.pathLength;
                                return !static_cast<bool>(sfi);
                                }),
                sfis.end());
    
    ACTS_VERBOSE("predict_next: after remove " << sfis.size() << " candidates remain");

    if (state.stepping.navDir == forward) {
      std::sort(sfis.begin(), sfis.end());
    } else {
      std::sort(sfis.begin(), sfis.end(), std::greater<>());
    }

    return sfis;
  }
};

}  // namespace Acts
