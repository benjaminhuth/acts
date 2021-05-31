// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// Workaround for building on clang+libstdc++
#include "Acts/Utilities/detail/ReferenceWrapperAnyCompat.hpp"

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/MultiComponentBoundTrackParameters.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/Propagator/DefaultExtension.hpp"
#include "Acts/Propagator/DenseEnvironmentExtension.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Propagator/EigenStepperError.hpp"
#include "Acts/Propagator/StepperExtensionList.hpp"
#include "Acts/Propagator/detail/Auctioneer.hpp"
#include "Acts/Propagator/detail/SteppingHelper.hpp"
#include "Acts/Utilities/Intersection.hpp"
#include "Acts/Utilities/Result.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

namespace Acts {

using namespace Acts::UnitLiterals;

/// @brief Stepper based on the EigenStepper, but handles Multi-Component Tracks
/// (e.g., for the GSF)
template <typename extensionlist_t = StepperExtensionList<DefaultExtension>,
          typename auctioneer_t = detail::VoidAuctioneer>
class MultiEigenStepper : public EigenStepper<extensionlist_t, auctioneer_t> {
 public:
  /// @brief Scoped enum which describes, if a track component is still not on a
  /// surface, on a surface or has missed the surface

  /// @brief Typedef to the Single-Component Eigen Stepper
  using SingleStepper = EigenStepper<extensionlist_t, auctioneer_t>;

  /// @brief Typedef to the State of the single component Stepper
  using SingleState = typename SingleStepper::State;

  using BoundState = typename SingleStepper::BoundState;
  using CurvilinearState = typename SingleStepper::CurvilinearState;
  using Covariance = typename SingleStepper::Covariance;
  using Jacobian = typename SingleStepper::Jacobian;

  struct State {
    State() = delete;

    /// Constructor from the initial bound track parameters
    ///
    /// @tparam charge_t Type of the bound parameter charge
    ///
    /// @param [in] gctx is the context object for the geometry
    /// @param [in] mctx is the context object for the magnetic field
    /// @param [in] par The track parameters at start
    /// @param [in] ndir The navigation direciton w.r.t momentum
    /// @param [in] ssize is the maximum step size
    /// @param [in] stolerance is the stepping tolerance
    ///
    /// @note the covariance matrix is copied when needed
    template <typename charge_t>
    explicit State(
        const GeometryContext& gctx, const MagneticFieldContext& mctx,
        const std::shared_ptr<const MagneticFieldProvider>& bField,
        const MultiComponentBoundTrackParameters<charge_t>& multipars,
        NavigationDirection ndir = forward,
        double ssize = std::numeric_limits<double>::max(),
        double stolerance = s_onSurfaceTolerance)
        : navDir(ndir) {
      throw_assert(!multipars.components().empty(),
                   "Empty MultiComponent state");

      for (const auto& [weight, single_component] : multipars.components()) {
        components.push_back(
            {SingleState(gctx, bField->makeCache(mctx), single_component, ndir,
                         ssize, stolerance),
             weight, Intersection3D::Status::reachable});
      }
    }

    struct Component {
      SingleState state;
      ActsScalar weight;
      Intersection3D::Status status;
    };

    std::vector<Component> components;

    /// Required through stepper concept
    /// TODO how can they be interpreted for a Multistepper
    bool covTransport = false;
    Covariance cov = Covariance::Zero();
    NavigationDirection navDir;
    double pathAccumulated = 0.;
  };

  /// Get the number of components
  std::size_t numberComponents(const State& state) const {
    return state.components.size();
  }

  template <typename charge_t>
  State makeState(std::reference_wrapper<const GeometryContext> gctx,
                  std::reference_wrapper<const MagneticFieldContext> mctx,
                  const MultiComponentBoundTrackParameters<charge_t>& par,
                  NavigationDirection ndir = forward,
                  double ssize = std::numeric_limits<double>::max(),
                  double stolerance = s_onSurfaceTolerance) const {
    return State(gctx, mctx, SingleStepper::m_bField, par, ndir, ssize,
                 stolerance);
  }

  /// Constructor
  MultiEigenStepper(std::shared_ptr<const MagneticFieldProvider> bField)
      : EigenStepper<extensionlist_t, auctioneer_t>(bField) {}

  /// Get the field for the stepping, it checks first if the access is still
  /// within the Cell, and updates the cell if necessary.
  ///
  /// @param [in,out] state is the propagation state associated with the track
  ///                 the magnetic field cell is used (and potentially updated)
  /// @param [in] pos is the field position
  Vector3 getField(State& state, const Vector3& pos) const {
    // get the field from the cell
    return SingleStepper::getField(state.components.front().state, pos);
  }

  /// Global particle position accessor
  ///
  /// @param state [in] The stepping state (thread-local cache)
  Vector3 position(const State& state) const {
    return std::accumulate(
        begin(state.components), end(state.components), Vector3::Zero().eval(),
        [this](const auto& sum, const auto& cmp) -> Vector3 {
          return sum + cmp.weight * SingleStepper::position(cmp.state);
        });
  }

  auto position(std::size_t i, const State& state) const {
    return SingleStepper::position(state.components.at(i).state);
  }

  /// Momentum direction accessor
  ///
  /// @param state [in] The stepping state (thread-local cache)
  Vector3 direction(const State& state) const {
    return std::accumulate(
               begin(state.components), end(state.components),
               Vector3::Zero().eval(),
               [this](const auto& sum, const auto& cmp) -> Vector3 {
                 return sum + cmp.weight * SingleStepper::direction(cmp.state);
               })
        .normalized();
  }

  auto direction(std::size_t i, const State& state) const {
    return SingleStepper::direction(state.components.at(i).state);
  }

  /// Absolute momentum accessor
  ///
  /// @param state [in] The stepping state (thread-local cache)
  double momentum(const State& state) const {
    return std::accumulate(
        state.components.begin(), state.components.end(), 0.0,
        [this](const double& sum, const auto& cmp) {
          return sum + cmp.weight * SingleStepper::momentum(cmp.state);
        });
  }

  auto momentum(std::size_t i, const State& state) const {
    return SingleStepper::momentum(state.components.at(i).state);
  }

  /// Charge access
  ///
  /// @param state [in] The stepping state (thread-local cache)
  double charge(const State& state) const {
    return SingleStepper::charge(state.components.front().state);
  }

  /// Time access
  ///
  /// @param state [in] The stepping state (thread-local cache)
  double time(const State& state) const {
    // time is not identical for each component because of relativity, right?
    return std::accumulate(
        state.components.begin(), state.components.end(), 0.0,
        [this](const double& sum, const auto& cmp) -> double {
          return sum + cmp.weight * SingleStepper::time(cmp.state);
        });
  }

  auto time(std::size_t i, const State& state) const {
    return SingleStepper::time(state.components.at(i).state);
  }

  /// Update surface status
  ///
  /// It checks the status to the reference surface & updates
  /// the step size accordingly
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  /// @param surface [in] The surface provided
  /// @param bcheck [in] The boundary check for this status update
  Intersection3D::Status updateSurfaceStatus(
      State& state, const Surface& surface, const BoundaryCheck& bcheck) const {
    std::array<int, 4> counts = {0, 0, 0, 0};

    for (auto& component : state.components) {
      component.status =
          SingleStepper::updateSurfaceStatus(component.state, surface, bcheck);
      ++counts[static_cast<std::size_t>(component.status)];
    }

    // This is a 'any_of' criterium. As long as any of the components has a
    // certain state, this determines the total state (in the order of a
    // somewhat importance)
    using Status = Intersection3D::Status;

    if (counts[static_cast<std::size_t>(Status::reachable)] > 0)
      return Status::reachable;
    else if (counts[static_cast<std::size_t>(Status::onSurface)] > 0)
      return Status::onSurface;
    else if (counts[static_cast<std::size_t>(Status::unreachable)] > 0)
      return Status::unreachable;
    else
      return Status::missed;
  }

  /// Update step size
  ///
  /// This method intersects the provided surface and update the navigation
  /// step estimation accordingly (hence it changes the state). It also
  /// returns the status of the intersection to trigger onSurface in case
  /// the surface is reached.
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  /// @param oIntersection [in] The ObjectIntersection to layer, boundary, etc
  /// @param release [in] boolean to trigger step size release
  template <typename object_intersection_t>
  void updateStepSize(State& state, const object_intersection_t& oIntersection,
                      bool release = true) const {
    //     std::cout << "BEFORE: " << outputStepSize(state) << std::endl;

    for (auto& component : state.components) {
      const auto intersection = oIntersection.representation->intersect(
          component.state.geoContext, SingleStepper::position(component.state),
          state.navDir * SingleStepper::direction(component.state), false);

      SingleStepper::updateStepSize(component.state, intersection, release);
    }

    //     std::cout << "BEFORE: " << outputStepSize(state) << std::endl;
  }

  /// Set Step size - explicitely with a double
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  /// @param stepSize [in] The step size value
  /// @param stype [in] The step size type to be set
  void setStepSize(State& state, double stepSize,
                   ConstrainedStep::Type stype = ConstrainedStep::actor) const {
    for (auto& component : state.components) {
      SingleStepper::setStepSize(component.state, stepSize, stype);
    }
  }

  /// Get the step size
  /// TODO add documentation
  double getStepSize(const State& state, ConstrainedStep::Type stype) const {
    return SingleStepper::getStepSize(
        std::min_element(begin(state.components), end(state.components),
                         [this, stype](const auto& a, const auto& b) {
                           return SingleStepper::getStepSize(a.state, stype) <
                                  SingleStepper::getStepSize(b.state, stype);
                         })
            ->state,
        stype);
  }

  /// Release the Step size
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  void releaseStepSize(State& state) const {
    for (auto& component : state.components) {
      SingleStepper::releaseStepSize(component.state);
    }
  }

  /// Output the Step Size - single component
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  std::string outputStepSize(const State& state) const {
    std::stringstream ss;
    for (const auto& component : state.components)
      ss << component.state.stepSize.toString() << " || ";

    return ss.str();
  }

  /// Overstep limit
  ///
  /// @param state [in] The stepping state (thread-local cache)
  double overstepLimit(const State& state) const {
    // A dynamic overstep limit could sit here
    return SingleStepper::overstepLimit(state.components.front().state);
  }

  /// @brief Resets the state
  ///
  /// @param [in, out] state State of the stepper
  /// @param [in] boundParams Parameters in bound parametrisation
  /// @param [in] cov Covariance matrix
  /// @param [in] surface The reference surface of the bound parameters
  /// @param [in] navDir Navigation direction
  /// @param [in] stepSize Step size
  void resetState(
      State& state, const BoundVector& boundParams, const BoundSymMatrix& cov,
      const Surface& surface, const NavigationDirection navDir = forward,
      const double stepSize = std::numeric_limits<double>::max()) const {
    for (auto& component : state.components) {
      SingleStepper::resetState(component.state, boundParams, cov, surface,
                                navDir, stepSize);
    }
  }

  /// Create and return the bound state at the current position
  ///
  /// @brief This transports (if necessary) the covariance
  /// to the surface and creates a bound state. It does not check
  /// if the transported state is at the surface, this needs to
  /// be guaranteed by the propagator
  ///
  /// @param [in] state State that will be presented as @c BoundState
  /// @param [in] surface The surface to which we bind the state
  /// @param [in] transportCov Flag steering covariance transport
  ///
  /// @return A bound state:
  ///   - the parameters at the surface
  ///   - the stepwise jacobian towards it (from last bound)
  ///   - and the path length (from start - for ordering)
  Result<BoundState> boundState(State& /*state*/, const Surface& /*surface*/,
                                bool /*transportCov = true*/) const {
    throw std::runtime_error(
        "'boundState' not yet implemented for MultiEigenStepper");
  }

  /// Create and return a curvilinear state at the current position
  ///
  /// @brief This transports (if necessary) the covariance
  /// to the current position and creates a curvilinear state.
  ///
  /// @param [in] state State that will be presented as @c CurvilinearState
  /// @param [in] transportCov Flag steering covariance transport
  ///
  /// @return A curvilinear state:
  ///   - the curvilinear parameters at given position
  ///   - the stepweise jacobian towards it (from last bound)
  ///   - and the path length (from start - for ordering)
  /// TODO only returns curvilinear state at front at the moment
  CurvilinearState curvilinearState(State& state,
                                    bool transportCov = true) const {
    return SingleStepper::curvilinearState(state.components.front().state,
                                           transportCov);
  }

  /// Method to update a stepper state to the some parameters
  ///
  /// @param [in,out] state State object that will be updated
  /// @param [in] pars Parameters that will be written into @p state
  /// TODO is this function useful for a MultiStepper?
  void update(State& state, const FreeVector& parameters,
              const Covariance& covariance) const {
    for (auto& component : state.components) {
      SingleStepper::update(component.state, parameters, covariance);
    }
  }

  /// Method to update momentum, direction and p
  ///
  /// @param [in,out] state State object that will be updated
  /// @param [in] uposition the updated position
  /// @param [in] udirection the updated direction
  /// @param [in] up the updated momentum value
  /// TODO is this function useful for a MultiStepper?
  void update(State& state, const Vector3& uposition, const Vector3& udirection,
              double up, double time) const {
    for (auto& component : state.components) {
      SingleStepper::update(component.state, uposition, udirection, up, time);
    }
  }

  /// Method for on-demand transport of the covariance
  /// to a new curvilinear frame at current  position,
  /// or direction of the state
  ///
  /// @param [in,out] state State of the stepper
  ///
  /// @return the full transport jacobian
  void transportCovarianceToCurvilinear(State& state) const {
    for (auto& component : state.components) {
      SingleStepper::transportCovarianceToCurvilinear(component.state);
    }
  }

  /// Method for on-demand transport of the covariance
  /// to a new curvilinear frame at current position,
  /// or direction of the state
  ///
  /// @tparam surface_t the Surface type
  ///
  /// @param [in,out] state State of the stepper
  /// @param [in] surface is the surface to which the covariance is forwarded
  /// to
  /// @note no check is done if the position is actually on the surface
  void transportCovarianceToBound(State& state, const Surface& surface) const {
    for (auto& component : state.components) {
      SingleStepper::transportCovarianceToBound(component.state, surface);
    }
  }

  /// Perform a Runge-Kutta track parameter propagation step
  ///
  /// @param [in,out] state is the propagation state associated with the track
  /// parameters that are being propagated.
  ///
  ///                      the state contains the desired step size.
  ///                      It can be negative during backwards track
  ///                      propagation,
  ///                      and since we're using an adaptive algorithm, it can
  ///                      be modified by the stepper class during
  ///                      propagation.
  template <typename propagator_state_t>
  Result<double> step(propagator_state_t& state) const {
    std::vector<Result<double>> results;

    struct SinglePropState {
      decltype(state.options)& options;
      SingleState& stepping;
    };

    for (auto& component : state.stepping.components) {
      if (component.status != Intersection3D::Status::reachable)
        continue;

      SinglePropState single_state{state.options, component.state};
      results.push_back(SingleStepper::step(single_state));
    }

    //     std::cout << "\nSTEP RESULT:\n";
    //     for(const auto &[st, weight, status] : state.stepping.components)
    //     {
    //         std::cout << st.pars.transpose() << std::endl;
    //     }
    //     std::cout << "-------------------------\n";

    if (results.empty())
      return 0.0;

    std::vector<Result<double>*> ok_results;
    for (auto& res : results)
      ok_results.push_back(&res);

//     std::cout << "h array = ";
//     for(auto r : ok_results)
//         std::cout << r->value() << " ";
//     std::cout << '\n';
    
    if (ok_results.empty())
      return results.front().error();
    else {
      const auto avg_step =
          std::accumulate(
              begin(ok_results), end(ok_results), 0.,
              [](auto sum, auto res) { return sum + res->value(); }) /
          static_cast<double>(ok_results.size());
      state.stepping.pathAccumulated += avg_step;
      return avg_step;
    }
  }
};

}  // namespace Acts
