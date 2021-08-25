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

#define RETURN_OR_ABORT(error) \
  if (m_config.abortOnError) { \
    std::abort();              \
  } else {                     \
    return error;              \
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

  /// The current indexes for the active components in the multi trajectory
  std::vector<std::size_t> currentTips;

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

      /// Whether to abort immediately when an error occurs
      bool abortOnError = false;

      /// A not so nice workaround to get the first backward state in the
      /// MultiTrajectory for the DirectNavigator
      std::optional<std::tuple<
          std::reference_wrapper<const MultiTrajectory<source_link_t>>,
          std::reference_wrapper<const std::vector<std::size_t>>,
          std::reference_wrapper<const std::map<std::size_t, double>>>>
          initInfoForMT;
    } m_config;

    /// Configurable components:
    updater_t m_updater;
    outlier_finder_t m_outlierFinder;
    calibrator_t m_calibrator;
    smoother_t m_smoother;
    SurfaceReached m_targetReachedAborter;

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

      ACTS_VERBOSE("Gsf step at mean position "
                   << stepper.position(state.stepping).transpose()
                   << " with direction "
                   << stepper.direction(state.stepping).transpose());

      for (auto i = 0ul; i < stepper.numberComponents(state.stepping); ++i) {
        typename stepper_t::ComponentProxy cmp(state.stepping, i);

        ACTS_VERBOSE(
            "  #" << i << " pos: "
                  << cmp.pars().template segment<3>(eFreePos0).transpose()
                  << ", dir: "
                  << cmp.pars().template segment<3>(eFreeDir0).transpose());
      }

      // Workaround to initialize MT in backward mode
      if (m_config.initInfoForMT && !result.haveInitializedMT) {
        const auto& [mt, idxs, weights] = *m_config.initInfoForMT;
        result.currentTips.clear();

        ACTS_VERBOSE(
            "Initialize the MultiTrajectory with information provided to the "
            "Actor");

        for (const auto idx : idxs.get()) {
          result.currentTips.push_back(
              result.fittedStates.addTrackState(TrackStatePropMask::All));
          auto proxy =
              result.fittedStates.getTrackState(result.currentTips.back());
          proxy.copyFrom(mt.get().getTrackState(idx));
          result.weightsOfStates[result.currentTips.back()] =
              weights.get().at(idx);

          // Because we are backwards, we use forward filtered as predicted
          proxy.predicted() = proxy.filtered();
          proxy.predictedCovariance() = proxy.filteredCovariance();
        }

        result.haveInitializedMT = true;
      }

      // TODO handle surfaces without material
      if (state.navigation.currentSurface &&
          state.navigation.currentSurface->surfaceMaterial()) {
        const auto& surface = *state.navigation.currentSurface;

        ACTS_VERBOSE("Step is at surface " << surface.geometryId());

        const auto source_link =
            m_config.inputMeasurements.find(surface.geometryId())->second;

        if (result.currentTips.empty()) {
          result.currentTips.resize(stepper.numberComponents(state.stepping),
                                    SIZE_MAX);
        }

        std::string_view direction =
            (state.stepping.navDir == forward) ? "forward" : "backward";
        ACTS_VERBOSE("Perform " << direction << " filter step");
        result.result = filter(state, stepper, result, source_link);
      } else {
        ACTS_VERBOSE("No material on surface, continue without any action");
      }
    }

    /// @brief A filtering step
    template <typename propagator_state_t, typename stepper_t>
    Result<void> filter(propagator_state_t& state, const stepper_t& stepper,
                        result_type& result,
                        const source_link_t& measurement) const {
      const auto& logger = state.options.logger;
      const auto& surface = *state.navigation.currentSurface;

      if (result.currentTips.size() !=
          stepper.numberComponents(state.stepping)) {
        ACTS_ERROR("component number mismatch:"
                   << result.currentTips.size() << " vs "
                   << stepper.numberComponents(state.stepping));

        RETURN_OR_ABORT(GsfError::ComponentNumberMismatch);
      }

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

      if (componentCache.empty()) {
        ACTS_ERROR("No component created");
        RETURN_OR_ABORT(GsfError::NoComponentCreated);
      }

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

      ACTS_VERBOSE("GSF actor: " << componentCache.size()
                                 << " after component reduction");

      // Update the Multi trajectory with the new components
      auto res =
          kalmanUpdateForward(state, result, measurement, componentCache);

      if (!res.ok()) {
        RETURN_OR_ABORT(res.error());
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

      // Adjust qop to account for lost energy due to lost components
      // TODO do we need to adjust variance?
      double sumW_loss = 0.0, sumWeightedQOverP_loss = 0.0;
      double initialQOverP = 0.0;

      for (auto i = 0ul; i < stepper.numberComponents(stepping); ++i) {
        typename stepper_t::ComponentProxy cmp(stepping, i);

        if (cmp.status() != Intersection3D::Status::onSurface) {
          sumW_loss += cmp.weight();
          sumWeightedQOverP_loss += cmp.weight() * cmp.pars()[eFreeQOverP];
        }

        initialQOverP += cmp.weight() * cmp.pars()[eFreeQOverP];
      }

      double checkWeightSum = 0.0;
      double checkQOverPSum = 0.0;

      for (auto i = 0ul; i < stepper.numberComponents(stepping); ++i) {
        typename stepper_t::ComponentProxy cmp(stepping, i);

        if (cmp.status() == Intersection3D::Status::onSurface) {
          auto& weight = cmp.weight();
          auto& qop = cmp.pars()[eFreeQOverP];

          weight /= (1.0 - sumW_loss);
          qop = qop * (1.0 - sumW_loss) + sumWeightedQOverP_loss;

          checkWeightSum += weight;
          checkQOverPSum += weight * qop;
        }
      }

      throw_assert(std::abs(checkQOverPSum - initialQOverP) < 1.e-8,
                   "momentum mismatch, initial: "
                       << initialQOverP << ", final: " << checkQOverPSum);
      throw_assert(std::abs(checkWeightSum - 1.0) < 1.e-8,
                   "must sum up to 1 but is " << checkWeightSum);

      // Approximate bethe-heitler distribution as gaussian mixture
      for (auto i = 0ul; i < stepper.numberComponents(stepping); ++i) {
        typename stepper_t::ComponentProxy old_cmp(stepping, i);

        auto boundState = old_cmp.boundState(surface, m_config.doCovTransport);

        if (old_cmp.status() != Intersection3D::Status::onSurface) {
          continue;
        }

        if (!boundState.ok()) {
          ACTS_ERROR("Failed to compute boundState: " << boundState.error());
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
            RETURN_OR_ABORT(updateRes.error());
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
      actor.m_config.initInfoForMT =
          typename decltype(actor.m_config.initInfoForMT)::value_type{
              std::ref(fwdGsfResult.fittedStates),
              std::ref(fwdGsfResult.currentTips),
              std::ref(fwdGsfResult.weightsOfStates)};

      auto propResult = m_propagator.propagate(params, propOptions);

      actor.m_config.initInfoForMT.reset();

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

    // TODO here we in principle need the predicted state and not the filtered
    // state of the last forward state. But for now just do this since it is
    // easier
    //     detail::MultiComponentState bwdFirstState;
    //     bwdFirstState.first =
    //     sMultiParsBwd.referenceSurface().getSharedPtr(); for (const auto& cmp
    //     : sMultiParsBwd.components()) {
    //       bwdFirstState.second.push_back(
    //           {std::get<double>(cmp),
    //            std::get<BoundTrackParameters>(cmp).parameters(),
    //            std::get<BoundTrackParameters>(cmp).covariance()});
    //     }
    //     bwdFiltered.push_back(bwdFirstState);

    //////////////////////////////
    // Last part towards perigee
    //////////////////////////////

    //     using Projector =
    //         detail::MultiTrajectoryProjector<detail::StatesType::ePredicted,
    //                                          source_link_t>;
    //     const auto [lastPars, lastCov] = detail::combineComponentRange(
    //         bwdGsfResult.currentTips.begin(), bwdGsfResult.currentTips.end(),
    //         Projector{bwdGsfResult.fittedStates,
    //         bwdGsfResult.weightsOfStates});
    //
    //
    //     const auto& surface = bwdGsfResult.fittedStates
    //                               .getTrackState(bwdGsfResult.currentTips.front())
    //                               .referenceSurface();
    //     MultiComponentBoundTrackParameters<SinglyCharged> lastMultiPars(
    //         surface.getSharedPtr(), lastPars, lastCov);

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
