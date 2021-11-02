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

#include <fstream>
#include <ios>
#include <map>
#include <numeric>

#include "BetheHeitlerApprox.hpp"
#include "GsfError.hpp"
#include "GsfUtils.hpp"
#include "KLMixtureReduction.hpp"
#include "MultiStepperAborters.hpp"
#include "MultiSteppingLogger.hpp"

using namespace std::string_literals;

constexpr static auto myNAN =
    std::numeric_limits<Acts::ActsScalar>::quiet_NaN();

template <std::size_t N>
class SimpleCsvWriter {
  std::ofstream m_file;

 public:
  SimpleCsvWriter(const std::string& filename,
                  const std::array<std::string, N>& headers)
      : m_file(filename, std::ios::app) {
    for (auto header : headers) {
      m_file << header << ",";
    }
    m_file << "\n";
  }

  template <typename... Args>
  void write(const Args&... args) {
    static_assert(sizeof...(Args) == N);
    ((m_file << args << ","), ...);
    m_file << "\n";
  }
};

#define RETURN_ERROR_OR_ABORT_ACTOR(error) \
  if (m_config.abortOnError) {             \
    std::abort();                          \
  } else {                                 \
    return error;                          \
  }

#define SET_ERROR_AND_RETURN_OR_ABORT_ACTOR(error) \
  if (m_config.abortOnError) {                     \
    std::abort();                                  \
  } else {                                         \
    result.result = error;                         \
    return;                                        \
  }

#define RETURN_ERROR_OR_ABORT_FIT(error) \
  if (options.abortOnError) {            \
    std::abort();                        \
  } else {                               \
    return error;                        \
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

  bool abortOnError = true;

  std::size_t maxComponents = 4;
  std::size_t maxSteps = 1000;
  bool loopProtection = false;
};

template <typename source_link_t>
struct GsfResult {
  /// The multi-trajectory which stores the graph of components
  MultiTrajectory<source_link_t> fittedStates;
  std::map<size_t, ActsScalar> weightsOfStates;

  /// The current indexes for the newest components in the multi trajectory
  std::vector<std::size_t> currentTips;

  /// The indices of the parent states in the multi trajectory. This is needed
  /// since there can be cases with splitting but no Kalman-Update, so we need
  /// to preserve this information
  std::vector<std::size_t> parentTips;

  /// The number of measurement states created
  std::size_t measurementStates = 0;
  std::size_t processedStates = 0;
  std::set<Acts::GeometryIdentifier> visitedSurfaces;

  // Propagate potential errors to the outside
  Result<void> result{Result<void>::success()};

  // Used for workaround to initialize MT correctly
  bool haveInitializedMT = false;
};

template <typename propagator_t>
struct GaussianSumFitter {
  GaussianSumFitter(propagator_t propagator)
      : m_propagator(std::move(propagator)),
        m_failStatistics("gsf-fail-statistics.csv",
                         {"errorMsg", "fwdSteps", "fwdPathlength", "bwdSteps",
                          "bwdPathLenght", "absoluteMomentum",
                          "transverseMomentum", "theta"}),
        m_successStatistics(
            "gsf-success-statistics.csv",
            {"fwdSteps", "fwdPathlength", "bwdSteps", "bwdPathLenght",
             "absoluteMomentum", "transverseMomentum", "theta"}) {}

  /// The navigator type
  using GsfNavigator = typename propagator_t::Navigator;

  /// The propagator instance used by the fit function
  propagator_t m_propagator;

  /// Some output files for debugging
  mutable SimpleCsvWriter<8> m_failStatistics;
  mutable SimpleCsvWriter<7> m_successStatistics;

  template <typename source_link_t, typename updater_t,
            typename outlier_finder_t, typename calibrator_t,
            typename smoother_t>
  struct Actor {
    /// Enforce default construction
    Actor() = default;

    /// Broadcast the result_type
    using result_type = GsfResult<source_link_t>;

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

      /// When to discard components
      double weightCutoff = 1.0e-8;

      /// A not so nice workaround to get the first backward state in the
      /// MultiTrajectory for the DirectNavigator
      std::function<void(result_type&, const LoggerWrapper&)>
          multiTrajectoryInitializer;
    } m_config;

    /// Configurable components:
    updater_t m_updater;
    outlier_finder_t m_outlierFinder;
    calibrator_t m_calibrator;
    smoother_t m_smoother;

    /// Broadcast Cache Type
    using TrackProxy = typename MultiTrajectory<source_link_t>::TrackStateProxy;
    using ComponentCache =
        std::tuple<std::variant<detail::GsfComponentParameterCache, TrackProxy>,
                   detail::GsfComponentMetaCache>;

    struct ParametersCacheProjector {
      auto& operator()(ComponentCache& cache) const {
        return std::get<detail::GsfComponentParameterCache>(std::get<0>(cache));
      }
      const auto& operator()(const ComponentCache& cache) const {
        return std::get<detail::GsfComponentParameterCache>(std::get<0>(cache));
      }
    };

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

      // Some initial printing
      ACTS_VERBOSE("Gsf step "
                   << state.stepping.steps << " at mean position "
                   << stepper.position(state.stepping).transpose()
                   << " with direction "
                   << stepper.direction(state.stepping).transpose());
      ACTS_VERBOSE(
          "Propagation is in "
          << (state.stepping.navDir == forward ? "forward" : "backward")
          << " mode");

      for (auto i = 0ul; i < stepper.numberComponents(state.stepping); ++i) {
        typename stepper_t::ComponentProxy cmp(state.stepping, i);

        if (cmp.status() == Acts::Intersection3D::Status::missed) {
          ACTS_VERBOSE("  #"
                       << i << " missed/unreachable, weight: " << cmp.weight());
        } else {
          auto getVector = [&](auto idx) {
            return cmp.pars().template segment<3>(idx).transpose();
          };
          ACTS_VERBOSE("  #" << i << " pos: " << getVector(eFreePos0)
                             << ", dir: " << getVector(eFreeDir0)
                             << ", weight: " << cmp.weight());
        }
      }

      // Workaround to initialize MT in backward mode
      if (!result.haveInitializedMT && m_config.multiTrajectoryInitializer) {
        m_config.multiTrajectoryInitializer(result, logger);
        result.haveInitializedMT = true;
      }

      // Initialize current tips on first pass
      if (result.parentTips.empty()) {
        result.parentTips.resize(stepper.numberComponents(state.stepping),
                                 SIZE_MAX);
      }

      // Check if we have the right number of components
      if (result.parentTips.size() !=
          stepper.numberComponents(state.stepping)) {
        ACTS_ERROR("component number mismatch:"
                   << result.parentTips.size() << " vs "
                   << stepper.numberComponents(state.stepping));

        SET_ERROR_AND_RETURN_OR_ABORT_ACTOR(GsfError::ComponentNumberMismatch);
      }

      // We only need to do something if we are on a surface
      if (state.navigation.currentSurface) {
        const auto& surface = *state.navigation.currentSurface;
        ACTS_VERBOSE("Step is at surface " << surface.geometryId());

        // Early return if we already were on this surface TODO why is this
        // necessary
        const auto [it, success] =
            result.visitedSurfaces.insert(surface.geometryId());

        if (!success) {
          ACTS_VERBOSE("Already visited surface, return");
          return;
        }

        // Check what we have on this surface
        const auto found_source_link =
            m_config.inputMeasurements.find(surface.geometryId());
        const bool haveMaterial =
            state.navigation.currentSurface->surfaceMaterial();
        const bool haveMeasurement =
            found_source_link != m_config.inputMeasurements.end();

        // Early return if nothing happens
        if (not haveMaterial && not haveMeasurement) {
          ACTS_VERBOSE("No material or measurement, return");
          return;
        }

        // Create Cache
        thread_local std::vector<ComponentCache> componentCache;
        componentCache.clear();

        // Projectors
        auto mapToProxyAndWeight = [&](auto& cmp) {
          auto& proxy = std::get<TrackProxy>(std::get<0>(cmp));
          return std::tie(proxy, result.weightsOfStates.at(proxy.index()));
        };

        auto mapProxyToWeightParsCov = [&](auto& variant) {
          auto& proxy = std::get<TrackProxy>(variant);
          return std::make_tuple(result.weightsOfStates.at(proxy.index()),
                                 proxy.filtered(), proxy.filteredCovariance());
        };

        auto mapCacheToWeightParsCov = [&](auto& variant) {
          auto& c = std::get<detail::GsfComponentParameterCache>(variant);
          return std::tie(c.weight, c.boundPars, c.boundCov);
        };

        ///////////////////////////////////////////
        // Component Splitting AND Kalman Update
        ///////////////////////////////////////////
        if (haveMaterial && haveMeasurement) {
          ACTS_VERBOSE("Material and measurement");
          detail::extractComponents(
              state, stepper, result.parentTips,
              detail::ComponentSplitter{m_config.bethe_heitler_approx,
                                        m_config.weightCutoff},
              m_config.doCovTransport, componentCache);

          detail::reduceWithKLDistance(
              componentCache,
              std::min(static_cast<std::size_t>(stepper.maxComponents),
                       m_config.maxComponents),
              ParametersCacheProjector{});

          result.result = kalmanUpdate(state, found_source_link->second, result,
                                       componentCache);
          detail::reweightComponents(componentCache, mapToProxyAndWeight,
                                     m_config.weightCutoff);
          result.currentTips = updateCurrentTips(
              componentCache, result.currentTips, mapProxyToWeightParsCov);
          result.parentTips = result.currentTips;

          detail::updateStepper(state, stepper, componentCache,
                                mapProxyToWeightParsCov);

          throw_assert(detail::componentWeightsAreNormalized(
                           componentCache, mapProxyToWeightParsCov),
                       "weights not normalized (material & measurement)");
        }
        /////////////////////////////////////////////
        // Component Splitting BUT NO Kalman Update
        /////////////////////////////////////////////
        else if (haveMaterial && not haveMeasurement) {
          ACTS_VERBOSE("Only material");
          detail::extractComponents(
              state, stepper, result.parentTips,
              detail::ComponentSplitter{m_config.bethe_heitler_approx,
                                        m_config.weightCutoff},
              m_config.doCovTransport, componentCache);

          detail::reduceWithKLDistance(
              componentCache,
              std::min(static_cast<std::size_t>(stepper.maxComponents),
                       m_config.maxComponents),
              ParametersCacheProjector{});

          result.parentTips.clear();
          for (const auto& [variant, meta] : componentCache) {
            result.parentTips.push_back(meta.parentIndex);
          }

          detail::normalizeWeights(componentCache, mapCacheToWeightParsCov);

          detail::updateStepper(
              state, stepper, componentCache, [&](const auto& variant) {
                return std::get<detail::GsfComponentParameterCache>(variant);
              });

          throw_assert(detail::componentWeightsAreNormalized(
                           componentCache, mapCacheToWeightParsCov),
                       "weights not normalized (only material)");
        }
        /////////////////////////////////////////////
        // Kalman Update BUT NO Component Splitting
        /////////////////////////////////////////////
        else if (not haveMaterial && haveMeasurement) {
          ACTS_VERBOSE("Only measurement");
          detail::extractComponents(state, stepper, result.parentTips,
                                    detail::ComponentForwarder{},
                                    m_config.doCovTransport, componentCache);

          result.result = kalmanUpdate(state, found_source_link->second, result,
                                       componentCache);

          detail::reweightComponents(componentCache, mapToProxyAndWeight,
                                     m_config.weightCutoff);

          result.currentTips = updateCurrentTips(
              componentCache, result.currentTips, mapProxyToWeightParsCov);
          result.parentTips = result.currentTips;

          detail::updateStepper(state, stepper, componentCache,
                                mapProxyToWeightParsCov);

          throw_assert(detail::componentWeightsAreNormalized(
                           componentCache, mapProxyToWeightParsCov),
                       "weights not normalized (only measurement)");
        }

        // Finally print the Components at the end
        ACTS_VERBOSE("Components after processing:");
        for (auto i = 0ul; i < stepper.numberComponents(state.stepping); ++i) {
          typename stepper_t::ComponentProxy cmp(state.stepping, i);

          if (cmp.status() == Acts::Intersection3D::Status::missed) {
            ACTS_VERBOSE(
                "  #" << i << " missed/unreachable, weight: " << cmp.weight());
          } else {
            ACTS_VERBOSE(
                "  #" << i << " pos: "
                      << cmp.pars().template segment<3>(eFreePos0).transpose()
                      << ", dir: "
                      << cmp.pars().template segment<3>(eFreeDir0).transpose()
                      << ", weight: " << cmp.weight());
          }
        }
      }
    }

    /// This is not very nice, include this in other functions or so
    template <typename projector_t>
    auto updateCurrentTips(const std::vector<ComponentCache>& components,
                           const std::vector<std::size_t>& currentTips,
                           const projector_t& proj) const {
      throw_assert(components.size() == currentTips.size(), "size mismatch");
      std::vector<std::size_t> newCurrentTips;
      for (auto i = 0ul; i < components.size(); ++i) {
        const auto& [variant, meta] = components[i];
        const auto& [weight, pars, cov] = proj(variant);

        if (weight == 0.0) {
          continue;
        } else {
          newCurrentTips.push_back(currentTips[i]);
        }
      }
      return newCurrentTips;
    }

    template <typename propagator_state_t>
    Result<void> kalmanUpdate(const propagator_state_t& state,
                              const source_link_t& source_link,
                              result_type& result,
                              std::vector<ComponentCache>& components) const {
      const auto& logger = state.options.logger;
      const auto& surface = *state.navigation.currentSurface;
      result.currentTips.clear();

      // Boolean flag, so that not every component increases the
      // result.measurementStates counter
      bool counted_as_measurement_state = false;

      for (auto& [variant, meta] : components) {
        // Create new track state
        result.currentTips.push_back(result.fittedStates.addTrackState(
            TrackStatePropMask::All, meta.parentIndex));

        const auto [weight, pars, cov] =
            std::get<detail::GsfComponentParameterCache>(variant);

        variant = result.fittedStates.getTrackState(result.currentTips.back());
        auto& trackProxy = std::get<TrackProxy>(variant);

        // Set track parameters
        trackProxy.predicted() = std::move(pars);
        if (cov) {
          trackProxy.predictedCovariance() = std::move(*cov);
        }
        result.weightsOfStates[result.currentTips.back()] = weight;

        // Set surface
        trackProxy.setReferenceSurface(surface.getSharedPtr());

        // assign the source link to the track state
        trackProxy.uncalibrated() = source_link;

        trackProxy.jacobian() = std::move(meta.jacobian);
        trackProxy.pathLength() = std::move(meta.pathLength);

        // We have predicted parameters, so calibrate the uncalibrated
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
            RETURN_ERROR_OR_ABORT_ACTOR(updateRes.error());
          }

          trackProxy.typeFlags().set(TrackStateFlag::MeasurementFlag);

          // We count the state with measurement TODO does this metric make
          // sense for a GSF?
          if (!counted_as_measurement_state) {
            ++result.measurementStates;
            counted_as_measurement_state = true;
          }
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

  /// @brief The fit function for the Direct navigator
  template <typename source_link_t, typename start_parameters_t,
            typename calibrator_t, typename outlier_finder_t>
  Acts::Result<Acts::KalmanFitterResult<source_link_t>> fit(
      const std::vector<source_link_t>& sourcelinks,
      const start_parameters_t& sParameters,
      const GsfOptions<calibrator_t, outlier_finder_t>& options,
      const std::vector<const Surface*>& sSequence) const {
    // Check if we have the correct navigator
    static_assert(
        std::is_same_v<DirectNavigator, typename propagator_t::Navigator>);

    using GSFActor = Actor<source_link_t, GainMatrixUpdater, outlier_finder_t,
                           calibrator_t, GainMatrixSmoother>;

    // Initialize the forward propagation with the DirectNavigator
    auto fwdPropInitializer = [&sSequence](const auto& opts,
                                           const auto& logger) {
      using Actors = ActionList<GSFActor, DirectNavigator::Initializer>;
      using Aborters = AbortList<>;

      PropagatorOptions<Actors, Aborters> propOptions(
          opts.geoContext, opts.magFieldContext, logger);

      propOptions.loopProtection = opts.loopProtection;
      propOptions.actionList.template get<DirectNavigator::Initializer>()
          .navSurfaces = sSequence;
      return propOptions;
    };

    // Initialize the backward propagation with the DirectNavigator
    auto bwdPropInitializer = [&sSequence](const auto& opts,
                                           const auto& logger) {
      using Actors = ActionList<GSFActor, DirectNavigator::Initializer>;
      using Aborters = AbortList<>;

      PropagatorOptions<Actors, Aborters> propOptions(
          opts.geoContext, opts.magFieldContext, logger);

      std::vector<const Surface*> backwardSequence(
          std::next(sSequence.rbegin()), sSequence.rend());
      backwardSequence.push_back(opts.referenceSurface);

      propOptions.loopProtection = opts.loopProtection;
      propOptions.actionList.template get<DirectNavigator::Initializer>()
          .navSurfaces = std::move(backwardSequence);

      return propOptions;
    };

    return fit_impl(sourcelinks, sParameters, options, fwdPropInitializer,
                    bwdPropInitializer);
  }

  /// @brief The fit function for the standard navigator
  template <typename source_link_t, typename start_parameters_t,
            typename calibrator_t, typename outlier_finder_t>
  Acts::Result<Acts::KalmanFitterResult<source_link_t>> fit(
      const std::vector<source_link_t>& sourcelinks,
      const start_parameters_t& sParameters,
      const GsfOptions<calibrator_t, outlier_finder_t>& options) const {
    // Check if we have the correct navigator
    static_assert(std::is_same_v<Navigator, typename propagator_t::Navigator>);

    // Create the ActionList and AbortList
    using GSFActor = Actor<source_link_t, GainMatrixUpdater, outlier_finder_t,
                           calibrator_t, GainMatrixSmoother>;

    // Initialize the forward propagation with the DirectNavigator
    auto fwdPropInitializer = [](const auto& opts, const auto& logger) {
      using Actors = ActionList<GSFActor>;
      using Aborters = AbortList<EndOfWorldReached>;

      PropagatorOptions<Actors, Aborters> propOptions(
          opts.geoContext, opts.magFieldContext, logger);
      propOptions.maxSteps = opts.maxSteps;
      propOptions.loopProtection = opts.loopProtection;

      return propOptions;
    };

    // Initialize the backward propagation with the DirectNavigator
    auto bwdPropInitializer = [](const auto& opts, const auto& logger) {
      using Actors = ActionList<GSFActor>;
      using Aborters = AbortList<EndOfWorldReached>;

      PropagatorOptions<Actors, Aborters> propOptions(
          opts.geoContext, opts.magFieldContext, logger);
      propOptions.maxSteps = opts.maxSteps;
      propOptions.loopProtection = opts.loopProtection;

      return propOptions;
    };

    return fit_impl(sourcelinks, sParameters, options, fwdPropInitializer,
                    bwdPropInitializer);
  }

  template <typename source_link_t, typename start_parameters_t,
            typename calibrator_t, typename outlier_finder_t,
            typename fwd_prop_initializer_t, typename bwd_prop_initializer_t>
  Acts::Result<Acts::KalmanFitterResult<source_link_t>> fit_impl(
      const std::vector<source_link_t>& sourcelinks,
      const start_parameters_t& sParameters,
      const GsfOptions<calibrator_t, outlier_finder_t>& options,
      const fwd_prop_initializer_t& fwdPropInitializer,
      const bwd_prop_initializer_t& bwdPropInitializer) const {
    static_assert(SourceLinkConcept<source_link_t>,
                  "Source link does not fulfill SourceLinkConcept");
    // The logger
    const auto& logger = options.logger;

    // Print some infos about the start parameters
    ACTS_VERBOSE("Run Gsf with start parameters: \n" << sParameters);

    auto intersectionStatusStartSurface =
        sParameters.referenceSurface()
            .intersect(GeometryContext{},
                       sParameters.position(GeometryContext{}),
                       sParameters.unitDirection(), true)
            .intersection.status;

    if (intersectionStatusStartSurface != Intersection3D::Status::onSurface) {
      ACTS_ERROR(
          "Surface intersection of start parameters with bound-check failed");
      m_failStatistics.write("StartParametersNotOnStartSurface"s, myNAN, myNAN,
                             myNAN, myNAN, sParameters.absoluteMomentum(),
                             sParameters.transverseMomentum(),
                             sParameters.template get<eBoundTheta>());
      return GsfError::StartParametersNotOnStartSurface;
    }

    // To be able to find measurements later, we put them into a map
    // We need to copy input SourceLinks anyways, so the map can own them.
    ACTS_VERBOSE("Preparing " << sourcelinks.size() << " input measurements");
    std::map<GeometryIdentifier, source_link_t> inputMeasurements;
    for (const auto& sl : sourcelinks) {
      inputMeasurements.emplace(sl.geometryId(), sl);
    }

    ACTS_VERBOSE(
        "Gsf: Final measuerement map size: " << inputMeasurements.size());
    throw_assert(sParameters.covariance() != std::nullopt,
                 "we need a covariance here...");

    using GSFActor = Actor<source_link_t, GainMatrixUpdater, outlier_finder_t,
                           calibrator_t, GainMatrixSmoother>;

    /////////////////
    // Forward pass
    /////////////////
    ACTS_VERBOSE("+-----------------------------+");
    ACTS_VERBOSE("| Gsf: Do forward propagation |");
    ACTS_VERBOSE("+-----------------------------+");

    auto fwdResult = [&]() {
      MultiComponentBoundTrackParameters<SinglyCharged> params(
          sParameters.referenceSurface().getSharedPtr(),
          sParameters.parameters(), sParameters.covariance());

      auto fwdPropOptions = fwdPropInitializer(options, logger);

      // Catch the actor and set the measurements
      auto& actor = fwdPropOptions.actionList.template get<GSFActor>();
      actor.m_config.inputMeasurements = inputMeasurements;
      actor.m_config.maxComponents = options.maxComponents;
      actor.m_calibrator = options.calibrator;
      actor.m_outlierFinder = options.outlierFinder;
      actor.m_config.abortOnError = options.abortOnError;

      fwdPropOptions.direction = Acts::forward;

      return m_propagator.propagate(params, fwdPropOptions);
    }();

    if (!fwdResult.ok()) {
      m_failStatistics.write(fwdResult.error().message(), myNAN, myNAN, myNAN,
                             myNAN, sParameters.absoluteMomentum(),
                             sParameters.transverseMomentum(),
                             sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(fwdResult.error());
    }

    auto& fwdGsfResult = (*fwdResult).template get<GsfResult<source_link_t>>();

    if (!fwdGsfResult.result.ok()) {
      m_failStatistics.write(fwdGsfResult.result.error().message(),
                             (*fwdResult).steps, (*fwdResult).pathLength, myNAN,
                             myNAN, sParameters.absoluteMomentum(),
                             sParameters.transverseMomentum(),
                             sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(fwdGsfResult.result.error());
    }

    if (fwdGsfResult.processedStates == 0) {
      m_failStatistics.write("noProcessedStates"s, (*fwdResult).steps,
                             (*fwdResult).pathLength, myNAN, myNAN,
                             sParameters.absoluteMomentum(),
                             sParameters.transverseMomentum(),
                             sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(GsfError::NoStatesCreated);
    }

    ACTS_VERBOSE("Finished forward propagation");
    ACTS_VERBOSE("- visited surfaces: " << fwdGsfResult.visitedSurfaces.size());
    ACTS_VERBOSE("- processed states: " << fwdGsfResult.processedStates);
    ACTS_VERBOSE("- measuerement states: " << fwdGsfResult.measurementStates);

    //////////////////
    // Backward pass
    //////////////////
    ACTS_VERBOSE("+------------------------------+");
    ACTS_VERBOSE("| Gsf: Do backward propagation |");
    ACTS_VERBOSE("+------------------------------+");

    auto bwdResult = [&]() {
      const auto params = detail::extractMultiComponentState(
          fwdGsfResult.fittedStates, fwdGsfResult.currentTips,
          fwdGsfResult.weightsOfStates, detail::StatesType::eFiltered);

      auto bwdPropOptions = bwdPropInitializer(options, logger);

      auto& actor = bwdPropOptions.actionList.template get<GSFActor>();
      actor.m_config.inputMeasurements = inputMeasurements;
      actor.m_config.maxComponents = options.maxComponents;
      actor.m_calibrator = options.calibrator;
      actor.m_outlierFinder = options.outlierFinder;
      actor.m_config.abortOnError = options.abortOnError;

      // Workaround to get the first state into the MultiTrajectory seems also
      // to be necessary for standard navigator to prevent double kalman
      // update on the last surface
      actor.m_config.multiTrajectoryInitializer = [&fwdGsfResult](
                                                      auto& result,
                                                      const auto& logger) {
        result.currentTips.clear();

        ACTS_VERBOSE(
            "Initialize the MultiTrajectory with information provided to the "
            "Actor");

        for (const auto idx : fwdGsfResult.currentTips) {
          result.currentTips.push_back(
              result.fittedStates.addTrackState(TrackStatePropMask::All));
          auto proxy =
              result.fittedStates.getTrackState(result.currentTips.back());
          proxy.copyFrom(fwdGsfResult.fittedStates.getTrackState(idx));
          result.weightsOfStates[result.currentTips.back()] =
              fwdGsfResult.weightsOfStates.at(idx);

          // Because we are backwards, we use forward filtered as predicted
          proxy.predicted() = proxy.filtered();
          proxy.predictedCovariance() = proxy.filteredCovariance();

          // Mark surface as visited
          result.visitedSurfaces.insert(proxy.referenceSurface().geometryId());
        }
      };

      bwdPropOptions.direction = Acts::backward;

      return m_propagator
          .template propagate<decltype(params), decltype(bwdPropOptions),
                              MultiStepperSurfaceReached>(
              params, *options.referenceSurface, bwdPropOptions);
    }();

    if (!bwdResult.ok()) {
      m_failStatistics.write(bwdResult.error().message(), (*fwdResult).steps,
                             (*fwdResult).pathLength, myNAN, myNAN,
                             sParameters.absoluteMomentum(),
                             sParameters.transverseMomentum(),
                             sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(bwdResult.error());
    }

    auto& bwdGsfResult = (*bwdResult).template get<GsfResult<source_link_t>>();

    if (!bwdGsfResult.result.ok()) {
      m_failStatistics.write(
          bwdGsfResult.result.error().message(), (*fwdResult).steps,
          (*fwdResult).pathLength, (*bwdResult).steps, (*bwdResult).pathLength,
          sParameters.absoluteMomentum(), sParameters.transverseMomentum(),
          sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(bwdGsfResult.result.error());
    }

    if (bwdGsfResult.processedStates == 0) {
      m_failStatistics.write(
          "noProcessedStates"s, (*fwdResult).steps, (*fwdResult).pathLength,
          (*bwdResult).steps, (*bwdResult).pathLength,
          sParameters.absoluteMomentum(), sParameters.transverseMomentum(),
          sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(GsfError::NoStatesCreated);
    }

    const auto& finalParameters = (*bwdResult).endParameters;

    if (finalParameters->referenceSurface().geometryId() !=
        options.referenceSurface->geometryId()) {
      m_failStatistics.write(
          "PropagationEndedOnWrongSurface"s, (*fwdResult).steps,
          (*fwdResult).pathLength, (*bwdResult).steps, (*bwdResult).pathLength,
          sParameters.absoluteMomentum(), sParameters.transverseMomentum(),
          sParameters.template get<eBoundTheta>());
      RETURN_ERROR_OR_ABORT_FIT(GsfError::PropagationEndedOnWrongSurface);
    }

    ////////////////////////////////////
    // Smooth and create Kalman Result
    ////////////////////////////////////
    ACTS_VERBOSE("Gsf: Do smoothing");

    const auto [combinedTraj, lastTip] = detail::smoothAndCombineTrajectories(
        fwdGsfResult.fittedStates, fwdGsfResult.currentTips,
        fwdGsfResult.weightsOfStates, bwdGsfResult.fittedStates,
        bwdGsfResult.currentTips, bwdGsfResult.weightsOfStates, logger);

    // Some test
    if (lastTip == SIZE_MAX) {
      m_failStatistics.write(
          "NoStatesCreated", (*fwdResult).steps, (*fwdResult).pathLength,
          (*bwdResult).steps, (*bwdResult).pathLength,
          sParameters.absoluteMomentum(), sParameters.transverseMomentum(),
          sParameters.template get<eBoundTheta>());
      return GsfError::NoStatesCreated;
    }

    Acts::KalmanFitterResult<source_link_t> kalmanResult;
    kalmanResult.lastTrackIndex = lastTip;
    kalmanResult.fittedStates = std::move(combinedTraj);
    kalmanResult.smoothed = true;
    kalmanResult.reversed = true;
    kalmanResult.finished = true;
    kalmanResult.lastMeasurementIndex = lastTip;
    kalmanResult.fittedParameters = *finalParameters;

    m_successStatistics.write((*fwdResult).steps, (*fwdResult).pathLength,
                              (*bwdResult).steps, (*bwdResult).pathLength,
                              sParameters.absoluteMomentum(),
                              sParameters.transverseMomentum(),
                              sParameters.template get<eBoundTheta>());

    return kalmanResult;
  }
};

}  // namespace Acts
