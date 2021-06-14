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
#include <vector>

#include "MultiEigenStepperCommon.hpp"
#include "NewGenericDefaultExtension.hpp"
#include "SimdHelpers.hpp"

namespace Acts {

namespace detail {

// TODO maybe implement an `SimdObjectIterable` which provides an SIMD object
// iterator... Or this can be done with Eigen::Map types, see below
template <int N, int R, int C>
auto extractFromSimdObject(const Eigen::Matrix<SimdType<N>, R, C>& m) {
  static_assert(R != Eigen::Dynamic && C != Eigen::Dynamic);
  std::array<Eigen::Matrix<typename SimdType<N>::Scalar, R, C>, N> ret;

  for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
      for (int n = 0; n < N; ++n)
        ret[n](r, c) = m(r, c)[n];

  return ret;
}

template <std::size_t N, int R, int C>
auto combineInSimdObject(
    const std::array<Eigen::Matrix<typename SimdType<N>::Scalar, R, C>, N>& a) {
  static_assert(R != Eigen::Dynamic && C != Eigen::Dynamic);
  Eigen::Matrix<SimdType<N>, R, C> ret;

  for (int c = 0; c < C; ++c)
    for (int r = 0; r < R; ++r)
      for (auto n = 0ul; n < N; ++n)
        ret(r, c)[n] = a[n](r, c);

  return ret;
}

}  // namespace detail

/// @brief Reducer struct for the SIMD MultiEigenStepper which reduces the
/// multicomponent state to simply by summing the weighted values
struct WeightedComponentReducerSIMD {
  template <int N>
  using SimdFreeMatrix = Eigen::Matrix<SimdType<N>, eFreeSize, eFreeSize>;

  template <int N>
  using SimdVector3 = Eigen::Matrix<SimdType<N>, 3, 1>;

  template <int N>
  using SimdFreeVector = Eigen::Matrix<SimdType<N>, eFreeSize, 1>;

  template <typename T, int N>
  static auto reduceWeightedSum(const Eigen::MatrixBase<T>& m,
                                const SimdType<N>& w) {
    constexpr int R = T::RowsAtCompileTime;
    constexpr int C = T::ColsAtCompileTime;

    static_assert(std::is_same_v<typename T::Scalar, SimdType<N>>);

    Eigen::Matrix<typename SimdType<N>::Scalar, R, C> ret;

    for (int c = 0; c < C; ++c)
      for (int r = 0; r < R; ++r)
        ret(r, c) = SimdHelpers::sum(m(r, c) * w);

    return ret;
  }

  template <typename stepper_state_t>
  static Vector3 position(const stepper_state_t& s) {
    return reduceWeightedSum(s.pars.template segment<3>(eFreePos0), s.weights);
  }

  template <typename stepper_state_t>
  static Vector3 direction(const stepper_state_t& s) {
    return reduceWeightedSum(s.pars.template segment<3>(eFreeDir0), s.weights)
        .normalized();
  }

  template <typename stepper_state_t>
  static ActsScalar momentum(const stepper_state_t& s) {
    return SimdHelpers::sum((1 / (s.pars[eFreeQOverP] / s.q)) * s.weights);
  }

  template <typename stepper_state_t>
  static ActsScalar time(const stepper_state_t& s) {
    return SimdHelpers::sum(s.pars[eFreeTime] * s.weights);
  }

  template <typename stepper_state_t>
  static FreeVector pars(const stepper_state_t& s) {
    return reduceWeightedSum(s.pars, s.weights);
  }

  template <typename stepper_state_t>
  static FreeMatrix cov(const stepper_state_t& s) {
    return reduceWeightedSum(s.pars, s.weights);
  }
};

/// @brief Stepper based on the EigenStepper, but handles Multi-Component Tracks
/// (e.g., for the GSF)
/// TODO inherit from EigenStepper<extensionlist_t, ...> when they are fully
/// compatible
template <int NComponents, typename extensionlist_t,
          typename component_reducer_t = WeightedComponentReducerSIMD,
          typename auctioneer_t = detail::VoidAuctioneer>
class MultiEigenStepperSIMD
    : public EigenStepper<
          StepperExtensionList<DefaultExtension, DenseEnvironmentExtension>,
          auctioneer_t> {
 public:
  /// @brief Typedef to the Single-Component Eigen Stepper TODO this should work
  /// with the NewExtensions in the end
  using SingleStepper = EigenStepper<
      StepperExtensionList<DefaultExtension, DenseEnvironmentExtension>,
      auctioneer_t>;

  /// @brief Typedef to the extensionlist of the underlying EigenStepper
  using SingleExtension = decltype(SingleStepper::State::extension);

  /// @brief Typedef to the State of the single component Stepper
  using SingleState = typename SingleStepper::State;

  /// @brief Use the definitions from the Single-stepper
  using typename SingleStepper::BoundState;
  using typename SingleStepper::Covariance;
  using typename SingleStepper::CurvilinearState;
  using typename SingleStepper::Jacobian;

  /// @brief The reducer type
  using Reducer = component_reducer_t;

  /// @brief How many components can this stepper manage?
  static constexpr int maxComponents = NComponents;

  /// @brief SIMD typedefs
  using SimdScalar = SimdType<NComponents>;
  using SimdVector3 = Eigen::Matrix<SimdScalar, 3, 1>;
  using SimdFreeVector = Eigen::Matrix<SimdScalar, eFreeSize, 1>;
  using SimdFreeMatrix = Eigen::Matrix<SimdScalar, eFreeSize, eFreeSize>;

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
        const GeometryContext& gctx, MagneticFieldProvider::Cache fieldCacheIn,
        const MultiComponentBoundTrackParameters<charge_t>& multipars,
        NavigationDirection ndir = forward,
        double ssize = std::numeric_limits<double>::max(),
        double /*stolerance*/ = s_onSurfaceTolerance)
        : q(multipars.charge()),
          navDir(ndir),
          fieldCache(std::move(fieldCacheIn)),
          geoContext(gctx) {
      throw_assert(!multipars.components().empty(),
                   "Empty MultiComponent state");
      throw_assert(multipars.components().size() <= NComponents,
                   "Missmatching component number: template="
                       << NComponents
                       << ", multipars=" << multipars.components().size());

      for (auto i = 0ul; i < multipars.components().size(); ++i) {
        // set bit in active mask
        active[i] = true;

        // extract the single representation of the component
        const auto bound = std::get<SingleBoundTrackParameters<charge_t>>(
            multipars.components().at(i));

        // pars
        const auto pos = bound.position(gctx);
        const auto dir = bound.unitDirection();

        pars[eFreePos0][i] = pos[0];
        pars[eFreePos1][i] = pos[1];
        pars[eFreePos2][i] = pos[2];
        pars[eFreeTime][i] = bound.parameters()[eBoundTime];
        pars[eFreeDir0][i] = dir[0];
        pars[eFreeDir1][i] = dir[1];
        pars[eFreeDir2][i] = dir[2];
        pars[eFreeQOverP][i] = bound.parameters()[eBoundQOverP];

        // weight
        weights[i] = std::get<double>(multipars.components().at(i));

        // handle covariance
        if (bound.covariance()) {
          const auto& surface = bound.referenceSurface();

          covTransport = true;
          covs[i] = BoundSymMatrix(*bound.covariance());
          jacToGlobals[i] =
              surface.boundToFreeJacobian(gctx, bound.parameters());
        }
      }

      // TODO Smater initialization when moving to std::array...
      for (auto i = 0ul; i < NComponents; ++i) {
        stepSizes.push_back(ConstrainedStep(ssize));
        status[i] = Intersection3D::Status::reachable;
      }
    }

    /// Mask which component is active
    std::bitset<NComponents> active;

    /// SIMD objects parameters
    SimdScalar weights;
    SimdScalar pathLengthSinceLastSurface = SimdScalar::Zero();
    SimdFreeVector pars;
    SimdFreeVector derivative;
    SimdFreeMatrix jacTransport = SimdFreeMatrix::Identity();

    /// Scalar objects in arrays TODO should they also be SIMD?
    std::array<Intersection3D::Status, NComponents> status;
    std::array<Covariance, NComponents> covs;
    std::array<Jacobian, NComponents> jacobians;
    std::array<BoundToFreeMatrix, NComponents> jacToGlobals;

    // no std::array, because ConstrainedStep is not default constructable.
    // TODO solve this later
    std::vector<ConstrainedStep> stepSizes;

    /// Scalar parameters
    double q = 1.;
    bool covTransport = false;
    NavigationDirection navDir;
    double pathAccumulated = 0.;

    // Bfield cache
    MagneticFieldProvider::Cache fieldCache;

    /// List of algorithmic extensions
    extensionlist_t extension;

    /// Auctioneer for choosing the extension
    auctioneer_t auctioneer;

    /// geoContext
    std::reference_wrapper<const GeometryContext> geoContext;

    // TODO Why is this here and not function-scope-local?
    struct {
      SimdVector3 B_first, B_middle, B_last;
      SimdVector3 k1, k2, k3, k4;
      std::array<SimdScalar, 4> kQoP;
    } stepData;
  };

  /// Proxy class that redirects calls to multi-calls
  struct MultiProxyStepper {
    auto direction(const State& state) const {
      return MultiEigenStepperSIMD::multiDirection(state);
    }
    auto position(const State& state) const {
      return MultiEigenStepperSIMD::multiPosition(state);
    }
    auto charge(const State& state) const {
      return MultiEigenStepperSIMD::charge_static(state);
    }
    auto momentum(const State& state) const {
      return MultiEigenStepperSIMD::multiMomentum(state);
    }
    auto reducedPosition(const State& state) const {
      return MultiEigenStepperSIMD::Reducer::position(state);
    }
  };

  /// Proxy stepper which acts of a specific component
  struct SingleProxyStepper {
    using State = MultiEigenStepperSIMD::State;

    const std::size_t i = 0;
    const double minus_olimit = 0.0;

    auto direction(const State& state) const {
      return MultiEigenStepperSIMD::direction(i, state);
    }
    auto position(const State& state) const {
      return MultiEigenStepperSIMD::position(i, state);
    }
    auto charge(const State& state) const {
      return MultiEigenStepperSIMD::charge_static(state);
    }
    auto momentum(const State& state) const {
      return MultiEigenStepperSIMD::momentum(i, state);
    }
    auto overstepLimit(const State&) const { return minus_olimit; }
    void setStepSize(
        State& state, double stepSize,
        ConstrainedStep::Type stype = ConstrainedStep::actor) const {
      state.stepSizes[i].update(stepSize, stype, true);
    }
    void releaseStepSize(State& state) const {
      state.stepSizes[i].release(ConstrainedStep::actor);
    };
    auto getStepSize(const State& state, ConstrainedStep::Type stype) const {
      return state.stepSizes[i].value(stype);
    };
  };

  /// Get the number of components
  std::size_t numberComponents(const State& state) const {
    return state.active.count();
  }

  auto extractComponents(State& state, const Surface& surface,
                         bool transportCov) const {
    std::vector<detail::CommonComponentRep> ret;
    ret.reserve(NComponents);

    // TODO Could use Eigen::Map types here instead of extracting and combining?
    // But then detail::boundState(...) must be templated...
    auto jacTransports = detail::extractFromSimdObject(state.jacTransport);
    const auto pars = detail::extractFromSimdObject(state.pars);
    auto derivatives = detail::extractFromSimdObject(state.derivative);

    for (int i = 0; i < NComponents; ++i) {
      if (!state.active[i])
        continue;

      ret.push_back(
          {detail::boundState(state.geoContext, state.covs[i],
                              state.jacobians[i], jacTransports[i],
                              derivatives[i], state.jacToGlobals[i], pars[i],
                              state.covTransport && transportCov,
                              state.pathAccumulated, surface)
               .value(),
           state.pathLengthSinceLastSurface[i], state.weights[i]});
    }

    state.jacTransport = detail::combineInSimdObject(jacTransports);
    state.derivative = detail::combineInSimdObject(derivatives);
    state.pathLengthSinceLastSurface = SimdScalar::Zero();

    return ret;
  }

  template <typename charge_t>
  State makeState(std::reference_wrapper<const GeometryContext> gctx,
                  std::reference_wrapper<const MagneticFieldContext> mctx,
                  const MultiComponentBoundTrackParameters<charge_t>& par,
                  NavigationDirection ndir = forward,
                  double ssize = std::numeric_limits<double>::max(),
                  double stolerance = s_onSurfaceTolerance) const {
    return State(gctx, SingleStepper::m_bField->makeCache(mctx), par, ndir,
                 ssize, stolerance);
  }

  /// Constructor
  MultiEigenStepperSIMD(std::shared_ptr<const MagneticFieldProvider> bField)
      : SingleStepper(bField) {}

  /// Get the field for the stepping, it checks first if the access is still
  /// within the Cell, and updates the cell if necessary.
  ///
  /// @param [in,out] state is the propagation state associated with the track
  ///                 the magnetic field cell is used (and potentially updated)
  /// @param [in] pos is the field position
  SimdVector3 getMultiField(State& state, const SimdVector3& multi_pos) const {
    SimdVector3 ret;

    for (auto i = 0ul; i < NComponents; ++i) {
      const Vector3 pos{multi_pos[0][i], multi_pos[1][i], multi_pos[2][i]};
      const Vector3 bf =
          SingleStepper::m_bField->getField(pos, state.fieldCache);

      ret[0][i] = bf[0];
      ret[1][i] = bf[1];
      ret[2][i] = bf[2];
    }

    return ret;
  }

  Vector3 getField(State& state, const Vector3& pos) const {
    // get the field from the cell
    return SingleStepper::m_bField->getField(pos, state.fieldCache);
  }

  /// Global particle position accessor
  ///
  /// @param state [in] The stepping state (thread-local cache)
  Vector3 position(const State& state) const {
    return Reducer::position(state);
  }

  static SimdVector3 multiPosition(const State& state) {
    return state.pars.template segment<3>(eFreePos0);
  }

  static Vector3 position(std::size_t i, const State& state) {
    return Vector3{state.pars[eFreePos0][i], state.pars[eFreePos1][i],
                   state.pars[eFreePos2][i]};
  }

  /// Momentum direction accessor
  ///
  /// @param state [in] The stepping state (thread-local cache)
  Vector3 direction(const State& state) const {
    return Reducer::direction(state);
  }

  static SimdVector3 multiDirection(const State& state) {
    return state.pars.template segment<3>(eFreeDir0);
  }

  static Vector3 direction(std::size_t i, const State& state) {
    return Vector3{state.pars[eFreeDir0][i], state.pars[eFreeDir1][i],
                   state.pars[eFreeDir2][i]};
  }

  /// Absolute momentum accessor
  ///
  /// @param state [in] The stepping state (thread-local cache)
  double momentum(const State& state) const { return Reducer::momentum(state); }

  static SimdScalar multiMomentum(const State& state) {
    return state.q / state.pars[eFreeQOverP];
  }

  static double momentum(std::size_t i, const State& state) {
    return state.q / state.pars[eFreeQOverP][i];
  }

  /// Charge access
  ///
  /// @param state [in] The stepping state (thread-local cache)
  double charge(const State& state) const { return state.q; }
  static double charge_static(const State& state) { return state.q; }

  static SimdScalar multiCharge(const State& state) { return state.q; }

  /// Time access
  ///
  /// @param state [in] The stepping state (thread-local cache)
  double time(const State& state) const { return Reducer::time(state); }

  static SimdScalar multiTime(const State& state) {
    return state.pars[eFreeTime];
  }

  static double time(std::size_t i, const State& state) {
    return state.pars[eFreeTime][i];
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
    // std::cout << "BEFORE updateSurfaceStatus(...): " << outputStepSize(state)
    // << std::endl;

    std::array<int, 4> counts = {0, 0, 0, 0};

    for (auto i = 0ul; i < NComponents; ++i) {
      state.status[i] = detail::updateSingleSurfaceStatus<SingleProxyStepper>(
          SingleProxyStepper{i, overstepLimit(state)}, state, surface, bcheck);
      ++counts[static_cast<std::size_t>(state.status[i])];
    }

    // std::cout << "COMPONENTS STATUS: ";
    // for (auto i = 0ul; i < NComponents; ++i) {
    //     std::cout << static_cast<std::size_t>(state.status[i]) << ", ";
    // }
    // std::cout << std::endl;

    // std::cout << "AFTER updateSurfaceStatus(...): " << outputStepSize(state)
    // << std::endl;

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
    // std::cout << "BEFORE updateStepSize(...): " << outputStepSize(state) <<
    // std::endl;

    for (auto i = 0ul; i < NComponents; ++i) {
      const auto intersection = oIntersection.representation->intersect(
          state.geoContext, position(i, state),
          state.navDir * direction(i, state), false);

      detail::updateSingleStepSize(state.stepSizes[i], intersection, release);
    }

    // std::cout << "AFTER updateStepSize(...): " << outputStepSize(state) <<
    // std::endl;
  }

  /// Set Step size - explicitely with a double
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  /// @param stepSize [in] The step size value
  /// @param stype [in] The step size type to be set
  void setStepSize(State& state, double stepSize,
                   ConstrainedStep::Type stype = ConstrainedStep::actor) const {
    //     state.previousStepSize = state.stepSize;
    for (auto& ss : state.stepSizes)
      ss.update(stepSize, stype, true);
  }

  double getStepSize(const State& state, ConstrainedStep::Type stype) const {
    return std::min_element(begin(state.stepSizes), end(state.stepSizes),
                            [stype](const auto& a, const auto& b) {
                              return a.value(stype) < b.value(stype);
                            })
        ->value(stype);
  }

  /// Release the Step size
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  void releaseStepSize(State& state) const {
    for (auto& ss : state.stepSizes)
      ss.release(ConstrainedStep::actor);
  }

  /// Output the Step Size - single component
  ///
  /// @param state [in,out] The stepping state (thread-local cache)
  std::string outputStepSize(const State& state) const {
    std::stringstream ss;
    for (const auto& s : state.stepSizes)
      ss << s.toString() << " || ";

    return ss.str();
  }

  /// Overstep limit
  ///
  /// @param state [in] The stepping state (thread-local cache)
  double overstepLimit(const State&) const {
    return -SingleStepper::m_overstepLimit;
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
      State& /*state*/, const BoundVector& /*boundParams*/,
      const BoundSymMatrix& /*cov*/, const Surface& /*surface*/,
      const NavigationDirection /*navDir*/ = forward,
      const double /*stepSize*/ = std::numeric_limits<double>::max()) const {
    throw std::runtime_error("'resetState' not yet implemented correctely");
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
  /// TODO reformulate with reducer functions
  CurvilinearState curvilinearState(State& state,
                                    bool transportCov = true) const {
    // std optional because CurvilinearState is not default constructable
    std::array<std::optional<CurvilinearState>, NComponents> states;

    // Compute all states
    for (auto i = 0ul; i < NComponents; ++i) {
      FreeVector pars, derivative;
      FreeMatrix jacTransport;

      for (auto j = 0ul; j < eFreeSize; ++j) {
        pars[j] = state.pars[j][i];
      }

      for (auto j = 0ul; j < eFreeSize; ++j)
        for (auto k = 0ul; k < eFreeSize; ++k) {
          jacTransport(j, k) = state.jacTransport(j, k)[i];
        }

      states[i] = detail::curvilinearState(
          state.covs[i], state.jacobians[i], jacTransport, derivative,
          state.jacToGlobals[i], pars, state.covTransport && transportCov,
          state.pathAccumulated);
    }

    // Sum everything up
    Vector4 pos = Vector4::Zero();
    Vector3 dir = Vector3::Zero();
    double p = 0., pathlen = 0.;
    Jacobian jac = Jacobian::Zero();

    for (const auto& curvstate : states) {
      const auto& curvpars = std::get<CurvilinearTrackParameters>(*curvstate);
      pos += curvpars.fourPosition(state.geoContext);
      dir += curvpars.unitDirection();
      p += curvpars.absoluteMomentum();
      pathlen += std::get<double>(*curvstate);
      jac += std::get<Jacobian>(*curvstate);
    }

    // Average over all
    double q = std::get<double>(*states.front());
    pos /= NComponents;
    dir.normalize();
    p /= NComponents;
    pathlen /= NComponents;
    jac /= NComponents;

    return CurvilinearState{CurvilinearTrackParameters(pos, dir, p, q), jac,
                            pathlen};
  }

  /// Method to update a stepper state to the some parameters
  ///
  /// @param [in,out] state State object that will be updated
  /// @param [in] pars Parameters that will be written into @p state
  /// TODO is this function useful for a MultiStepper?
  void update(State& /*state*/, const FreeVector& /*parameters*/,
              const Covariance& /*covariance*/) const {
    throw std::runtime_error("'update' not yet implemented correctely");
  }

  /// Method to update momentum, direction and p
  ///
  /// @param [in,out] state State object that will be updated
  /// @param [in] uposition the updated position
  /// @param [in] udirection the updated direction
  /// @param [in] up the updated momentum value
  /// TODO is this function useful for a MultiStepper?
  void update(State& /*state*/, const Vector3& /*uposition*/,
              const Vector3& /*udirection*/, double /*up*/,
              double /*time*/) const {
    throw std::runtime_error("'update' not yet implemented correctely");
  }

  /// Method for on-demand transport of the covariance
  /// to a new curvilinear frame at current  position,
  /// or direction of the state
  ///
  /// @param [in,out] state State of the stepper
  ///
  /// @return the full transport jacobian
  void transportCovarianceToCurvilinear(State& /*state*/) const {
    throw std::runtime_error(
        "'transportCovarianceToCurvilinear' not yet implemented correctely");
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
  void transportCovarianceToBound(State& /*state*/,
                                  const Surface& /*surface*/) const {
    throw std::runtime_error(
        "'transportCovarianceToBound' not yet implemented correctely");
  }

  /*********************
   *
   *  Step Implementation
   *
   *******************/

  // Seperated stepsize estimate from Single eigen stepper
  /// TODO state should be constant, but then magne
  template <typename propagator_state_t>
  Result<double> estimate_step_size(const propagator_state_t& state,
                                    const Vector3& k1,
                                    MagneticFieldProvider::Cache& fieldCache,
                                    const SingleProxyStepper& stepper,
                                    const ConstrainedStep step_size) const {
    double error_estimate = 0.;
    auto current_estimate = step_size;

    // Create SingleExtension locally here
    SingleExtension extension;

    // Don't forget to bid, so that everything works
    extension.validExtensionForStep(state, stepper);

    // If not initialized to zero, we get undefined behaviour
    struct {
      Vector3 B_first = Vector3::Zero();
      Vector3 B_middle = Vector3::Zero();
      Vector3 B_last = Vector3::Zero();
      Vector3 k1 = Vector3::Zero();
      Vector3 k2 = Vector3::Zero();
      Vector3 k3 = Vector3::Zero();
      Vector3 k4 = Vector3::Zero();
      std::array<double, 4> kQoP = {0., 0., 0., 0.};
    } sd;

    sd.k1 = k1;

    const auto pos = stepper.position(state.stepping);
    const auto dir = stepper.direction(state.stepping);

    const auto tryRungeKuttaStep = [&](const ConstrainedStep& h) -> bool {
      // State the square and half of the step size
      const double h2 = h * h;
      const double half_h = h * 0.5;

      // Second Runge-Kutta point
      const Vector3 pos1 = pos + half_h * dir + h2 * 0.125 * sd.k1;
      sd.B_middle = SingleStepper::m_bField->getField(pos1, fieldCache);
      if (!extension.k2(state, stepper, sd.k2, sd.B_middle, sd.kQoP, half_h,
                        sd.k1)) {
        return false;
      }

      // Third Runge-Kutta point
      if (!extension.k3(state, stepper, sd.k3, sd.B_middle, sd.kQoP, half_h,
                        sd.k2)) {
        return false;
      }

      // Last Runge-Kutta point
      const Vector3 pos2 = pos + h * dir + h2 * 0.5 * sd.k3;
      sd.B_last = SingleStepper::m_bField->getField(pos2, fieldCache);
      if (!extension.k4(state, stepper, sd.k4, sd.B_last, sd.kQoP, h, sd.k3)) {
        return false;
      }

      // Compute and check the local integration error estimate
      error_estimate = std::max(
          h2 * ((sd.k1 - sd.k2 - sd.k3 + sd.k4).template lpNorm<1>() +
                std::abs(sd.kQoP[0] - sd.kQoP[1] - sd.kQoP[2] + sd.kQoP[3])),
          1e-20);

      return (error_estimate <= state.options.tolerance);
    };

    double stepSizeScaling = 1.;
    size_t nStepTrials = 0;
    // Select and adjust the appropriate Runge-Kutta step size as given
    // ATL-SOFT-PUB-2009-001
    while (!tryRungeKuttaStep(current_estimate)) {
      stepSizeScaling =
          std::min(std::max(0.25, std::pow((state.options.tolerance /
                                            std::abs(2. * error_estimate)),
                                           0.25)),
                   4.);

      current_estimate = current_estimate * stepSizeScaling;

      // If step size becomes too small the particle remains at the initial
      // place
      if (std::abs(current_estimate) < std::abs(state.options.stepSizeCutOff)) {
        // Not moving due to too low momentum needs an aborter
        return EigenStepperError::StepSizeStalled;
      }

      // If the parameter is off track too much or given stepSize is not
      // appropriate
      if (nStepTrials > state.options.maxRungeKuttaStepTrials) {
        // Too many trials, have to abort
        return EigenStepperError::StepSizeAdjustmentFailed;
      }
      nStepTrials++;
    }

    return current_estimate;
  }

  /// Perform a Runge-Kutta track parameter propagation step
  ///
  /// @param [in,out] state is the propagation state associated with the track
  /// parameters that are being propagated.
  template <typename propagator_state_t>
  Result<double> step(propagator_state_t& state) const {
    auto& sd = state.stepping.stepData;
    auto& stepping = state.stepping;

    const auto pos = multiPosition(stepping);
    const auto dir = multiDirection(stepping);

    // First Runge-Kutta point
    sd.B_first = getMultiField(stepping, pos);
    if (!stepping.extension.validExtensionForStep(state, MultiProxyStepper{}) ||
        !stepping.extension.k1(state, MultiProxyStepper{}, sd.k1, sd.B_first,
                               sd.kQoP)) {
      return EigenStepperError::StepInvalid;
    }

    // check for nan
    for (int i = 0; i < 3; ++i)
      throw_assert(!sd.k1[i].isNaN().any(),
                   "k1 contains nan for component " << i);

    // Now do stepsize estimate, use the minimum momentum component for this
    auto estimated_h = [&]() {
      Eigen::Index r, c;
      multiMomentum(stepping).minCoeff(&r, &c);

      const Vector3 k1{sd.k1[0][r], sd.k1[1][r], sd.k1[2][r]};
      const ConstrainedStep h = stepping.stepSizes[r];

      return estimate_step_size(state, k1, stepping.fieldCache,
                                SingleProxyStepper{static_cast<std::size_t>(r)},
                                h);
    }();

    if (!estimated_h.ok())
      return estimated_h.error();

    // Constant stepsize at the moment
    const SimdScalar h = [&]() {
      const ActsScalar h = *estimated_h;
      SimdScalar s = SimdScalar::Zero();

      for (auto i = 0ul; i < NComponents; ++i) {
        // h = 0 if surface not reachable, effectively suppress any progress
        if (stepping.active[i] &&
            stepping.status[i] == Intersection3D::Status::reachable) {
          // make sure we get the correct minimal stepsize
          s[i] = std::min(h, static_cast<ActsScalar>(stepping.stepSizes[i]));
        }
      }

      return s;
    }();

    // If everything is zero, nothing to do (TODO should this happen?)
    if (h.sum() == 0.0) {
      return 0.0;
    }

    const SimdScalar h2 = h * h;
    const SimdScalar half_h = h * SimdScalar(0.5);

    // Second Runge-Kutta point
    const SimdVector3 pos1 = pos + half_h * dir + h2 * 0.125 * sd.k1;
    sd.B_middle = getMultiField(stepping, pos1);

    if (!stepping.extension.k2(state, MultiProxyStepper{}, sd.k2, sd.B_middle,
                               sd.kQoP, half_h, sd.k1)) {
      return EigenStepperError::StepInvalid;
    }

    // check for nan
    for (int i = 0; i < 3; ++i)
      throw_assert(!sd.k2[i].isNaN().any(),
                   "k2 contains nan for component " << i);

    // Third Runge-Kutta point
    if (!stepping.extension.k3(state, MultiProxyStepper{}, sd.k3, sd.B_middle,
                               sd.kQoP, half_h, sd.k2)) {
      return EigenStepperError::StepInvalid;
    }

    // check for nan
    for (int i = 0; i < 3; ++i)
      throw_assert(!sd.k3[i].isNaN().any(),
                   "k3 contains nan for component " << i);

    // Last Runge-Kutta point
    const SimdVector3 pos2 = pos + h * dir + h2 * 0.5 * sd.k3;
    sd.B_last = getMultiField(stepping, pos2);

    if (!stepping.extension.k4(state, MultiProxyStepper{}, sd.k4, sd.B_last,
                               sd.kQoP, h, sd.k3)) {
      return EigenStepperError::StepInvalid;
    }

    // check for nan
    for (int i = 0; i < 3; ++i)
      throw_assert(!sd.k4[i].isNaN().any(),
                   "k4 contains nan for component " << i);

    // When doing error propagation, update the associated Jacobian matrix
    if (stepping.covTransport) {
      // The step transport matrix in global coordinates
      SimdFreeMatrix D;
      if (!stepping.extension.finalize(state, MultiProxyStepper{}, h, D)) {
        return EigenStepperError::StepInvalid;
      }

      // for moment, only update the transport part
      stepping.jacTransport = D * stepping.jacTransport;
    } else {
      if (!stepping.extension.finalize(state, MultiProxyStepper{}, h)) {
        return EigenStepperError::StepInvalid;
      }
    }

    // Update the track parameters according to the equations of motion
    stepping.pars.template segment<3>(eFreePos0) +=
        h * dir + h2 / SimdScalar(6.) * (sd.k1 + sd.k2 + sd.k3);
    stepping.pars.template segment<3>(eFreeDir0) +=
        h / SimdScalar(6.) * (sd.k1 + SimdScalar(2.) * (sd.k2 + sd.k3) + sd.k4);

    // Normalize the direction (TODO this can for sure be done smarter...)
    for (auto i = 0ul; i < NComponents; ++i) {
      Vector3 d{stepping.pars[eFreeDir0][i], stepping.pars[eFreeDir1][i],
                stepping.pars[eFreeDir2][i]};

      d.normalize();

      stepping.pars[eFreeDir0][i] = d[0];
      stepping.pars[eFreeDir1][i] = d[1];
      stepping.pars[eFreeDir2][i] = d[2];
    }

    // check for nan
    for (auto i = 0ul; i < eFreeSize; ++i)
      throw_assert(
          !stepping.pars[i].isNaN().any(),
          "AT THE END track parameters contain nan for component " << i);

    // Update pathLengthSinceLastSurface
    stepping.pathLengthSinceLastSurface += h;

    // Compute average step and return
    const auto avg_step = h.sum() / NComponents;

    stepping.pathAccumulated += avg_step;
    return avg_step;
  }
};

}  // namespace Acts
