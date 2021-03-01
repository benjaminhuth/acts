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
class MLNavigator {
  std::shared_ptr<Acts::PairwiseScoreModel> m_mainModel;
  std::shared_ptr<Acts::TargetPredModel> m_fallbackModel;
  std::shared_ptr<const Acts::TrackingGeometry> m_tgeo;

 public:
  struct State {
    std::vector<const Acts::Surface *> navSurfaces;
    std::vector<const Acts::Surface *> navSurfacesFallback;
    std::vector<const Acts::Surface *>::const_iterator navSurfaceIter;
    const Acts::Surface *currentSurface;
    const Acts::Surface *startSurface;

    bool navigationBreak = false;
    bool targetReached = false;
    bool inFallbackMode = false;

    const Acts::TrackingVolume *currentVolume = nullptr;
    const Acts::Surface *targetSurface = nullptr;
  };

  MLNavigator() { throw_assert(false, "Don't use this constructor!"); }

  /// Constructor of the ML Navigator
  MLNavigator(std::shared_ptr<Acts::PairwiseScoreModel> psm,
              std::shared_ptr<Acts::TargetPredModel> tpm,
              std::shared_ptr<const Acts::TrackingGeometry> tgeo)
      : m_mainModel(std::move(psm)),
        m_fallbackModel(std::move(tpm)),
        m_tgeo(tgeo) {}

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
          state.stepping, **state.navigation.navSurfaceIter, false);

      if (surfaceStatus == Acts::Intersection3D::Status::onSurface) {
        // Set the current surface
        state.navigation.currentSurface = *state.navigation.navSurfaceIter;
        ACTS_VERBOSE("Current surface set to  "
                     << state.navigation.currentSurface->geometryId());

        // Release Stepsize
        ACTS_VERBOSE("Release Stepsize");
        stepper.releaseStepSize(state.stepping);

        // Reset state
        state.navigation.navSurfaces.clear();
        state.navigation.navSurfaceIter = state.navigation.navSurfaces.end();

        // Check if we can navigate further
        if (m_mainModel->possible_start_surfaces().find(
                state.navigation.currentSurface) ==
            m_mainModel->possible_start_surfaces().end()) {
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

    try {
      ACTS_VERBOSE(">>>>>>>> TARGET <<<<<<<<<");
      auto &navstate = state.navigation;

      // Predict new targets if there are no candidates in state
      if (navstate.navSurfaceIter == navstate.navSurfaces.end()) {
        // This means also we are currently on a surface
        assert(navstate.currentSurface != nullptr);
        ACTS_VERBOSE(
            "It seems like we are on a surface and must predict new targets");

        navstate.navSurfaces = m_mainModel->predict_next(
            navstate.currentSurface, state.stepping.pars, logger);

        // Can be done later but for now here is simplest. also don't have to
        // worry about currentSurface which is nullptr later
        navstate.navSurfaces = m_fallbackModel->predict_next<20>(
            navstate.currentSurface, state.stepping.pars, logger);

        navstate.inFallbackMode = false;
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
          state.stepping, **navstate.navSurfaceIter, false);

      ACTS_VERBOSE("After updateSurfaceStatus");

      // Everything OK
      if (surfaceStatus == Acts::Intersection3D::Status::reachable) {
        ACTS_VERBOSE("Navigation stepSize set to "
                     << stepper.outputStepSize(state.stepping));
      }
      // Try another surface
      else if (surfaceStatus == Acts::Intersection3D::Status::unreachable) {
        ACTS_VERBOSE(
            "Surface not reachable anymore, search another one which is "
            "reachable (in fallback mode: "
            << std::boolalpha << navstate.inFallbackMode << ")");

        navstate.navSurfaces.erase(navstate.navSurfaceIter);
        navstate.navSurfaceIter = navstate.navSurfaces.end();

        // Search in the current surface pool
        auto found = std::find_if(navstate.navSurfaces.begin(),
                                  navstate.navSurfaces.end(), [&](auto s) {
                                    return stepper.updateSurfaceStatus(
                                               state.stepping, *s, false) ==
                                           Intersection3D::Status::reachable;
                                  });

        // If not found and not yet in fallback mode, load fallback pool and
        // search again
        if (found == navstate.navSurfaces.end() && !navstate.inFallbackMode) {
          ACTS_VERBOSE(
              "the main model could not find a surface, use fallback model");

          navstate.navSurfaces = navstate.navSurfacesFallback;
          navstate.inFallbackMode = true;

          found = std::find_if(navstate.navSurfaces.begin(),
                               navstate.navSurfaces.end(), [&](auto s) {
                                 return stepper.updateSurfaceStatus(
                                            state.stepping, *s, false) ==
                                        Intersection3D::Status::reachable;
                               });
        }

        // If we did not find the surface, stop the navigation
        if (found == navstate.navSurfaces.end()) {
          ACTS_ERROR("Could not find the surface, stop navigation!");
          navstate.currentVolume = nullptr;
          navstate.navigationBreak = true;
        } else {
          navstate.navSurfaceIter = found;
        }
      }
      // Something strange happended
      else {
        throw std::runtime_error(
            "surface status is 'missed' or 'on_surface', thats not what we "
            "want here");
      }
    } catch (std::exception &e) {
      throw std::runtime_error("Error in MLNavigator::target() - "s + e.what());
    }
  }
};

}  // namespace Acts
