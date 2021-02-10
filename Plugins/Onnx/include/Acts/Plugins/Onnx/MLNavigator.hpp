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
#include "Acts/Plugins/Onnx/OnnxModel.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <fstream>
#include <map>
#include <set>

using namespace std::string_literals;

namespace Acts {

auto get_beampline_id(double pos_z, std::vector<double> bpsplit) {
  // set new highest and lowest boundary
  bpsplit.front() = std::numeric_limits<double>::lowest();
  bpsplit.back() = std::numeric_limits<double>::max();
    
  auto it = bpsplit.cbegin();
  for (; it != std::prev(bpsplit.cend()); ++it) {
      if (pos_z >= *it && pos_z < *std::next(it))
        break;
  }
    
  const auto id = static_cast<std::size_t>(std::distance(bpsplit.cbegin(), it));
    
  throw_assert(id < (bpsplit.size() - 1ul), "");
  return id;
}

struct NoSurfaceLeftException : std::runtime_error {
  NoSurfaceLeftException(std::string msg) : std::runtime_error(msg) {}
};


/// A navigator which uses a neural network and a graph of surfaces to estimate
/// the next surface. The neural network computes a score [0,1] from a pair of
/// surfaces (represented by their 3D center coordinates) and the current track
/// parameters (direction, qop). If all candidates from the graph are scored,
/// they are sorted by score and one after another is tried as a target surface.
/// The Beampipe-Surface is splitted along the z-axis
class MLNavigator {
 public:
struct Config {
  // Embedding vectors
  using RealSpaceVec = Eigen::Matrix<float, 3, 1>;
  using EmbeddingVec = Eigen::Matrix<float, 10, 1>;
  
  // Some convenience typedefs
  using SurfaceTargetGraph =
      std::map<const Acts::Surface *, std::set<const Acts::Surface *>>;
  using BeampipeGraph = std::map<std::size_t, std::set<const Acts::Surface *>>;
  using SurfaceToEmbeddingMapping = std::map<const Acts::Surface *, EmbeddingVec>;

  // The ONNX Models
  std::shared_ptr<OnnxModel<3, 1>> pairwise_score_model;
  std::shared_ptr<OnnxModel<2, 1>> target_pred_model;
  std::shared_ptr<OnnxModel<1, 1>> embedding_model;
  
  // Geometry information
  std::shared_ptr<const Acts::TrackingGeometry> tgeo;
  const SurfaceTargetGraph surfaceToTargets;
  const BeampipeGraph bpsplitIdToTargets;
  const SurfaceToEmbeddingMapping surfaceToEmbedding10;

  // Beampipe split
  const std::vector<double> bpsplitZBounds;
  const std::size_t bpsplitPhi;
};
     

 private:
  Config m_config;

 public:
  struct State {
    std::vector<const Acts::Surface *> navSurfaces;
    std::vector<const Acts::Surface *>::const_iterator navSurfaceIter;
    const Acts::Surface *currentSurface;
    const Acts::Surface *startSurface;

    bool navigationBreak = false;
    bool targetReached = false;

    const Acts::TrackingVolume *currentVolume = nullptr;
    const Acts::Surface *targetSurface = nullptr;
  };

  /// Constructor of the ML
  MLNavigator(const MLNavigatorConfig &n = MLNavigatorConfig()) : m_config(n) {
    // TODO Maybe bring a logger wrapper reference here?
    auto max_size = [](const auto &map) {
      return std::max_element(map.begin(), map.end(),
                              [](auto a, auto b) {
                                return a.second.size() < b.second.size();
                              })
          ->second.size();
    };

    std::cout << "Initialilzed ML Navigator\n";
    std::cout << "- Graph size: " << m_config.surfaceToTargets.size() << "\n";
    std::cout << "  - Max targets: " << max_size(m_config.surfaceToTargets)
              << "\n";
    std::cout << "- BPSplit Bounds size: " << m_config.bpsplitZBounds.size()
              << "\n";
    std::cout << "- BPSplit Graph size: " << m_config.bpsplitIdToTargets.size()
              << "\n";
    std::cout << "  - Max targets: " << max_size(m_config.bpsplitIdToTargets)
              << std::endl;
  }

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
        state.navigation.currentSurface = m_config.tgeo->getBeamline();
        state.navigation.currentVolume = m_config.tgeo->highestTrackingVolume();
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
        if (m_config.surfaceToTargets.find(state.navigation.currentSurface) ==
            m_config.surfaceToTargets.end()) {
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
      const auto &navstate = state.navigation;

      // Predict new targets if there are no candidates in state
      if (navstate.navSurfaceIter == navstate.navSurfaces.end()) {
        // This means also we are currently on a surface
        assert(navstate.currentSurface != nullptr);
        ACTS_VERBOSE(
            "It seems like we are on a surface and must predict new targets");

        predict_new_target(state.navigation, logger, state.stepping.pars,
                           state.stepping.geoContext.get());
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
      state.navigation.currentSurface = nullptr;

      ACTS_VERBOSE("Ask for SurfaceStatus of currently most probable target");

      // Establish & update the surface status
      auto surfaceStatus = stepper.updateSurfaceStatus(
          state.stepping, **state.navigation.navSurfaceIter, false);

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
            "reachable");

        state.navigation.navSurfaces.erase(state.navigation.navSurfaceIter);
        state.navigation.navSurfaceIter = navstate.navSurfaces.end();

        if (navstate.navSurfaces.empty())
          throw NoSurfaceLeftException("There is now surface left to try");

        for (auto iter = navstate.navSurfaces.begin();
             iter != navstate.navSurfaces.end(); ++iter) {
          auto new_status =
              stepper.updateSurfaceStatus(state.stepping, **iter, false);

          if (new_status == Acts::Intersection3D::Status::reachable) {
            state.navigation.navSurfaceIter = iter;
            break;
          }
        }

        if (navstate.navSurfaceIter == navstate.navSurfaces.end())
          throw NoSurfaceLeftException("We did not find a suitable surface");
      }
      // Something strange happended
      else {
        throw std::runtime_error(
            "surface status is 'missed' or 'on_surface', thats not what we "
            "want here");
      }
    } catch (NoSurfaceLeftException &e) {
      ACTS_ERROR("No surface left ("s + e.what() +
                 ") -> Stop navigation for now!");

      state.navigation.currentVolume = nullptr;
      state.navigation.navigationBreak = true;
      return;
    } catch (std::exception &e) {
      throw std::runtime_error("Error in MLNavigator::target() - "s + e.what());
    }
  }

 private:
  /// Predicts new targets with the 'pairwise-score' model (i.e.
  void predict_new_target(State &nav_state, const Acts::LoggerWrapper &logger,
                          const Acts::FreeVector &free_params,
                          const Acts::GeometryContext &gctx) const {
    // Helper function for beampipe split
    auto get_bpsplit_surface_position = [&](double z_pos) {
      auto it = m_config.bpsplitZBounds.cbegin();
      for (; it != std::prev(m_config.bpsplitZBounds.cend()); ++it)
        if (z_pos >= *it && z_pos < *std::next(it))
          break;

      return Config::RealSpaceVec{
          0.f, 0.f, static_cast<float>(*it + 0.5 * (*std::next(it) - *it))};
    };
    
    ACTS_VERBOSE("Entered 'predict_new_target' function");

    // Prepare fixed nn parameters (start surface, track params)
    const auto curSurf = nav_state.currentSurface;

    if (curSurf->geometryId().value() != 0) {
      throw_assert((m_config.surfaceToTargets.find(curSurf) !=
                    m_config.surfaceToTargets.end()),
                   "");
    } else {
      throw_assert((m_config.bpsplitIdToTargets.find(get_beampline_id(
                        free_params[eFreePos2], m_config.bpsplitZBounds)) !=
                    m_config.bpsplitIdToTargets.end()),
                   "");
    }

    const auto possible_targets =
        curSurf->geometryId().value() == 0
            ? m_config.bpsplitIdToTargets.at(get_beampline_id(
                  free_params[eFreePos2], m_config.bpsplitZBounds))
            : m_config.surfaceToTargets.at(curSurf);

    const Config::RealSpaceVec start_emb =
        curSurf->geometryId().value() == 0
            ? get_bpsplit_surface_position(free_params[Acts::eFreePos2])
            : curSurf->center(gctx).cast<float>();

    const Eigen::Vector4f in_params =
        free_params.segment<4>(Acts::eFreeDir0).cast<float>();

    std::vector<std::pair<float, const Acts::Surface *>> predictions;
    predictions.reserve(possible_targets.size());

    // Loop over all possible targets and predict score
    ACTS_VERBOSE("Start target prediction loop with " << possible_targets.size()
                                                      << " targets");

    for (auto target : possible_targets) {
      const Eigen::Vector3f target_embedding =
          target->center(gctx).cast<float>();

      auto output = std::tuple<Eigen::Matrix<float, 1, 1>>();
      auto input = std::tuple{start_emb, target_embedding, in_params};

      m_config.pairwise_score_model->predict(output, input);

      predictions.push_back({std::get<0>(output)[0], target});
    }

    ACTS_VERBOSE("Finished target prediction loop");

    // Sort by score and extract pointers, then set state.navSurfaces
    std::sort(predictions.begin(), predictions.end(),
              [&](auto a, auto b) { return a.first > b.first; });

    ACTS_VERBOSE("Highest score is " << predictions[0].first << " ("
                                     << predictions[0].second->geometryId()
                                     << ")");

    std::vector<const Acts::Surface *> target_surfaces(predictions.size());
    std::transform(predictions.begin(), predictions.end(),
                   target_surfaces.begin(), [](auto a) { return a.second; });

    nav_state.navSurfaces = target_surfaces;
    nav_state.navSurfaceIter = nav_state.navSurfaces.begin();

    ACTS_VERBOSE("Set 'navSurfaces' and 'navSurfaceIter'");
  }
  
  void predict_new_target_fallback(State &nav_state, const Acts::LoggerWrapper &logger,
                                   const Acts::FreeVector &free_params,
                                   const Acts::GeometryContext &gctx) const {
    ACTS_VERBOSE("Entered fallback prediction function");
    
    std::tuple<Eigen::Matrix<float, 1, 1>> start_id;
    std::get<0>(start_id)[0] = m_config.surfaceToNumber.at(nav_state.currentSurface);
    
    std::tuple<Config::EmbeddingVec> start_emb;
    
    m_config.embedding_model->predict(start_id, start_emb);
    
    
    
    
    
  }
};

}  // namespace Acts
