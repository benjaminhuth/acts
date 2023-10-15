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
#include "Acts/Propagator/MultiStepperError.hpp"
#include "Acts/Propagator/StepperExtensionList.hpp"
#include "Acts/Propagator/detail/Auctioneer.hpp"
#include "Acts/Propagator/detail/SimdHelpers.hpp"
#include "Acts/Propagator/detail/SteppingHelper.hpp"
#include "Acts/Utilities/Intersection.hpp"
#include "Acts/Utilities/Result.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace Acts {

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
          typename auctioneer_t = detail::VoidAuctioneer,
          typename single_stepper_extensionlist_t =
              StepperExtensionList<DefaultExtension>>
class MultiEigenStepperSIMD
    : public EigenStepper<single_stepper_extensionlist_t, auctioneer_t> {
 public:
  /// @brief Typedef to the Single-Component Eigen Stepper TODO this should work
  /// with the NewExtensions in the end
  using SingleStepper =
      EigenStepper<single_stepper_extensionlist_t, auctioneer_t>;

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
    /// Number of current active components
    std::size_t numComponents;

    /// SIMD objects parameters
    SimdScalar weights;
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

    /// Particle hypothesis
    ParticleHypothesis particleHypothesis = ParticleHypothesis::pion();

    bool covTransport = false;
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
    explicit State(const GeometryContext& gctx,
                   const MagneticFieldContext& mctx,
                   const std::shared_ptr<const MagneticFieldProvider>& bfield,
                   const MultiComponentBoundTrackParameters& multipars,
                   double ssize = std::numeric_limits<double>::max())
        : particleHypothesis(multipars.particleHypothesis()),
          fieldCache(bfield->makeCache(mctx)),
          geoContext(gctx) {
      assert(!multipars.components().empty() && "empty cmps");
      assert(multipars.components().size() <= NComponents && "mismatching cmp number");

      numComponents = multipars.components().size();

      for (auto i = 0ul; i < multipars.components().size(); ++i) {
        // extract the single representation of the component
        const auto [weight, bound, cov] = multipars.components().at(i);
        const auto free = Acts::detail::transformBoundToFreeParameters(
            multipars.referenceSurface(), gctx, bound);

        for (auto e = eFreePos0; e < eFreeSize;
             e = static_cast<FreeIndices>(e + 1)) {
          pars[e][i] = free[e];
        }

        // weight
        weights[i] = weight;

        // handle covariance
        if (cov) {
          covTransport = true;
          covs[i] = BoundSquareMatrix(*cov);
          jacToGlobals[i] =
              multipars.referenceSurface().boundToFreeJacobian(gctx, bound);
        }
      }

      // TODO Smater initialization when moving to std::array...
      for (auto i = 0ul; i < NComponents; ++i) {
        stepSizes.push_back(ConstrainedStep(ssize));
        status[i] = Intersection3D::Status::reachable;
      }
    }
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
    void setStepSize(State& state, double stepSize,
                     ConstrainedStep::Type stype = ConstrainedStep::actor,
                     bool /*release*/ = true) const {
      state.stepSizes[i].update(stepSize, stype, true);
    }
    void releaseStepSize(State& state) const {
      state.stepSizes[i].release(ConstrainedStep::actor);
    };
    auto getStepSize(const State& state, ConstrainedStep::Type stype) const {
      return state.stepSizes[i].value(stype);
    };
  };

  /// A template class which contains all const member functions, that should be
  /// available both in the mutable ComponentProxy and the ConstComponentProxy.
  /// @tparam component_t Must be a const or mutable State::Component.
  template <typename state_t>
  struct ComponentProxyBase {
    state_t& m_state;
    const std::size_t m_i;

    template <typename T>
    static auto map(T& m, int i) {
      constexpr int Rows = std::decay_t<decltype(m)>::RowsAtCompileTime;
      constexpr int Cols = std::decay_t<decltype(m)>::ColsAtCompileTime;
      static_assert(Rows != Eigen::Dynamic && Cols != Eigen::Dynamic);
      constexpr int iStride = NComponents;
      constexpr int oStride = iStride * Rows;

      if constexpr (std::is_const_v<T>) {
        return Eigen::Map<const Eigen::Matrix<ActsScalar, Rows, Cols>,
                          Eigen::Unaligned, Eigen::Stride<oStride, iStride>>(
            m(0, 0).data() + i);
      } else {
        return Eigen::Map<Eigen::Matrix<ActsScalar, Rows, Cols>,
                          Eigen::Unaligned, Eigen::Stride<oStride, iStride>>(
            m(0, 0).data() + i);
      }
    }

   public:
    ComponentProxyBase(state_t& s, std::size_t i) : m_state(s), m_i(i) {
      assert(i < NComponents && "Cannot create proxy: out of range");
    }

    auto weight() const { return m_state.weights[m_i]; }
    auto charge() const { return m_state.q; }
    auto pathAccumulated() const { return m_state.pathAccumulated; }
    auto status() const { return m_state.status[m_i]; }
    auto pars() const { return map(m_state.pars, m_i); }
    auto derivative() const { return map(m_state.derivative, m_i); }
    auto jacTransport() const { return map(m_state.jacTransport, m_i); }
    const auto& cov() const { return m_state.covs[m_i]; }
    const auto& jacobian() const { return m_state.jacobians[m_i]; }
    const auto& jacToGlobal() const { return m_state.jacToGlobals[m_i]; }
  };

  /// A proxy struct which allows access to a single component of the
  /// multi-component state. It has the semantics of a const reference, i.e.
  /// it requires a const reference of the single-component state it
  /// represents
  using ConstComponentProxy = ComponentProxyBase<const State>;

  /// A proxy struct which allows access to a single component of the
  /// multi-component state. It has the semantics of a mutable reference, i.e.
  /// it requires a mutable reference of the single-component state it
  /// represents
  struct ComponentProxy : ComponentProxyBase<State> {
    using Base = ComponentProxyBase<State>;

    using Base::m_i;
    using Base::m_state;

    // Import the const accessors from ComponentProxyBase
    using Base::charge;
    using Base::cov;
    using Base::derivative;
    using Base::jacobian;
    using Base::jacToGlobal;
    using Base::jacTransport;
    using Base::pars;
    using Base::pathAccumulated;
    using Base::status;
    using Base::weight;

    ComponentProxy(State& s, std::size_t i) : Base(s, i) {}

    auto& weight() { return m_state.weights[m_i]; }
    auto& charge() { return m_state.q; }
    auto& pathAccumulated() { return m_state.pathAccumulated; }
    auto& status() { return m_state.status[m_i]; }
    auto pars() { return map(m_state.pars, m_i); }
    auto derivative() { return map(m_state.derivative, m_i); }
    auto jacTransport() { return map(m_state.jacTransport, m_i); }
    auto& cov() { return m_state.covs[m_i]; }
    auto& jacobian() { return m_state.jacobians[m_i]; }
    auto& jacToGlobal() { return m_state.jacToGlobals[m_i]; }

    Result<BoundState> boundState(const Surface& surface, bool transportCov) {
      if (status() != Intersection3D::Status::onSurface)
        return MultiStepperError::ComponentNotOnSurface;

      // TODO template detail::bounState(...) on Eigen::MatrixBase<T> to allow
      // the Map types to go in directely
      auto jacTransportMap = jacTransport();
      auto derivativeMap = derivative();
      auto parsMap = pars();

      Acts::FreeMatrix jacTransport(jacTransportMap);
      Acts::FreeVector derivative(derivativeMap);
      Acts::FreeVector pars(parsMap);

      auto bs = detail::boundState(
          m_state.geoContext, cov(), jacobian(), jacTransport, derivative,
          jacToGlobal(), pars, transportCov, m_state.pathAccumulated, surface);

      jacTransportMap = jacTransport;
      derivativeMap = derivative;
      parsMap = pars;

      if (!bs.ok())
        return bs.error();

      return std::move(bs.value());
    }
  };

  /// Creates an iterable which can be plugged into a range-based for-loop to
  /// iterate over components
  /// @note Use a for-loop with by-value semantics, since the Iterable returns a
  /// proxy internally holding a reference
  auto componentIterable(State& state) const {
    struct Iterator {
      using difference_type = std::ptrdiff_t;
      using value_type = ComponentProxy;
      using reference = ComponentProxy;
      using pointer = void;
      using iterator_category = std::forward_iterator_tag;

      State& s;
      std::size_t i;

      // clang-format off
      auto& operator++() { ++i; return *this; }
      auto operator!=(const Iterator& other) const { return i != other.i; }
      auto operator==(const Iterator& other) const { return i == other.i; }
      auto operator*() const { return ComponentProxy(s, i); }
      // clang-format on
    };

    struct Iterable {
      using iterator = Iterator;

      State& s;

      // clang-format off
      auto begin() { return Iterator{s, 0ul}; }
      auto end() { return Iterator{s, NComponents}; }
      // clang-format on
    };

    return Iterable{state};
  }

  /// Creates an constant iterable which can be plugged into a range-based
  /// for-loop to iterate over components
  /// @note Use a for-loop with by-value semantics, since the Iterable returns a
  /// proxy internally holding a reference
  auto constComponentIterable(const State& state) const {
    struct ConstIterator {
      using difference_type = std::ptrdiff_t;
      using value_type = ConstComponentProxy;
      using reference = ConstComponentProxy;
      using pointer = void;
      using iterator_category = std::forward_iterator_tag;

      const State& s;
      std::size_t i;

      // clang-format off
      auto& operator++() { ++i; return *this; }
      auto operator!=(const ConstIterator& other) const { return i != other.i; }
      auto operator==(const ConstIterator& other) const { return i == other.i; }
      auto operator*() const { return ConstComponentProxy{s, i}; }
      // clang-format on
    };

    struct Iterable {
      using iterator = ConstIterator;
      const State& s;

      // clang-format off
      auto begin() const { return ConstIterator{s, 0}; }
      auto end() const { return ConstIterator{s, NComponents}; }
      // clang-format on
    };

    return Iterable{state};
  }

  /// Get the number of components
  std::size_t numberComponents(const State& state) const {
    return state.numComponents;
  }

  State makeState(std::reference_wrapper<const GeometryContext> gctx,
                  std::reference_wrapper<const MagneticFieldContext> mctx,
                  const MultiComponentBoundTrackParameters& par,
                  Direction ndir = Direction::Forward,
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
  ///                 the magnetic field cell is used (and potentially
  ///                 updated)
  /// @param [in] pos is the field position
  SimdVector3 getMultiField(State& state, const SimdVector3& multi_pos) const {
    SimdVector3 ret;

    for (auto i = 0ul; i < NComponents; ++i) {
      const Vector3 pos{multi_pos[0][i], multi_pos[1][i], multi_pos[2][i]};
      const Vector3 bf =
          *SingleStepper::m_bField->getField(pos, state.fieldCache);

      ret[0][i] = bf[0];
      ret[1][i] = bf[1];
      ret[2][i] = bf[2];
    }

    return ret;
  }

  Result<Vector3> getField(State& state, const Vector3& pos) const {
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
      State& state, const Surface& /*surface*/, const BoundaryCheck& /*bcheck*/,
      const Logger& /*logger*/ = getDummyLogger()) const {
    // std::cout << "BEFORE updateSurfaceStatus(...): " <<
    // outputStepSize(state)
    // << std::endl;

    std::array<int, 4> counts = {0, 0, 0, 0};

    for (auto i = 0ul; i < NComponents; ++i) {
      assert(false);
      // state.status[i] =
      // detail::updateSingleSurfaceStatus<SingleProxyStepper>(
      //     SingleProxyStepper{i, overstepLimit(state)}, state, surface,
      //     bcheck, logger);
      ++counts[static_cast<std::size_t>(state.status[i])];
    }

    // std::cout << "COMPONENTS STATUS: ";
    // for (auto i = 0ul; i < NComponents; ++i) {
    //     std::cout << static_cast<std::size_t>(state.status[i]) << ", ";
    // }
    // std::cout << std::endl;

    // std::cout << "AFTER updateSurfaceStatus(...): " <<
    // outputStepSize(state)
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
                   ConstrainedStep::Type stype = ConstrainedStep::actor,
                   bool /*release*/ = true) const {
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
      const BoundSquareMatrix& /*cov*/, const Surface& /*surface*/,
      const Direction /*navDir*/ = Direction::Forward,
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
  Result<BoundState> boundState(
      State& /*state*/, const Surface& /*surface*/,
      bool /*transportCov*/ = true,
      const FreeToBoundCorrection& /*freeToBoundCorrection*/ =
          FreeToBoundCorrection(false)) const {
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

  /// Method to update the components individually
  template <typename component_rep_t>
  void updateComponents(State& state, const std::vector<component_rep_t>& cmps,
                        const Surface&) const {
    assert(cmps.size() <= NComponents && "tried to create more components than possible");

    state.numComponents = cmps.size();

    for (auto i = 0ul; i < cmps.size(); ++i) {
      ComponentProxy proxy(state, i);

      proxy.pars() = cmps[i].trackStateProxy->filtered();
      if (state.covTransport) {
        proxy.cov() = cmps[i].trackStateProxy->filteredCovariance();
      }
      proxy.jacobian() = cmps[i].jacobian;
      proxy.jacToGlobal() = cmps[i].jacToGlobal;
      proxy.derivative() = cmps[i].derivative;
      proxy.jacTransport() = cmps[i].jacTransport;
      proxy.weight() = cmps[i].weight;
    }
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
  void transportCovarianceToBound(
      State& /*state*/, const Surface& /*surface*/,
      const FreeToBoundCorrection& /*freeToBoundCorrection*/
      = FreeToBoundCorrection(false)) const {
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
  template <typename propagator_state_t, typename navigator_t>
  Result<double> estimate_step_size(const propagator_state_t& state,
                                    const navigator_t &navigator,
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
      const double h2 = h.value() * h.value();
      const double half_h = h.value() * 0.5;

      // Second Runge-Kutta point
      const Vector3 pos1 = pos + half_h * dir + h2 * 0.125 * sd.k1;
      sd.B_middle = *SingleStepper::m_bField->getField(pos1, fieldCache);
      if (!extension.k2(state, stepper, navigator, sd.k2, sd.B_middle, sd.kQoP, half_h,
                        sd.k1)) {
        return false;
      }

      // Third Runge-Kutta point
      if (!extension.k3(state, stepper, navigator, sd.k3, sd.B_middle, sd.kQoP, half_h,
                        sd.k2)) {
        return false;
      }

      // Last Runge-Kutta point
      const Vector3 pos2 = pos + h * dir + h2 * 0.5 * sd.k3;
      sd.B_last = *SingleStepper::m_bField->getField(pos2, fieldCache);
      if (!extension.k4(state, stepper, navigator, sd.k4, sd.B_last, sd.kQoP, h, sd.k3)) {
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

      current_estimate.setValue(current_estimate.value() * stepSizeScaling);

      // If step size becomes too small the particle remains at the initial
      // place
      if (std::abs(current_estimate.value()) <
          std::abs(state.options.stepSizeCutOff)) {
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
  template <typename propagator_state_t, typename navigator_t>
  Result<double> step(propagator_state_t& state,
                      const navigator_t& navigator) const {
    auto& sd = state.stepping.stepData;
    auto& stepping = state.stepping;

    const auto pos = multiPosition(stepping);
    const auto dir = multiDirection(stepping);

    // First Runge-Kutta point
    sd.B_first = getMultiField(stepping, pos);
    if (!stepping.extension.validExtensionForStep(state, MultiProxyStepper{}, navigator) ||
        !stepping.extension.k1(state, MultiProxyStepper{}, navigator, sd.k1, sd.B_first,
                               sd.kQoP)) {
      return EigenStepperError::StepInvalid;
    }

    // check for nan
    for (int i = 0; i < 3; ++i) {
      assert(!sd.k1[i].isNaN().any() && "k1 contains nan");
    }

    // Now do stepsize estimate, use the minimum momentum component for this
    auto estimated_h = [&]() {
      Eigen::Index r, c;
      multiMomentum(stepping).minCoeff(&r, &c);

      const Vector3 k1{sd.k1[0][r], sd.k1[1][r], sd.k1[2][r]};
      const ConstrainedStep h = stepping.stepSizes[r];

      return estimate_step_size(state, navigator, k1, stepping.fieldCache,
                                SingleProxyStepper{static_cast<std::size_t>(r)},
                                h);
    }();

    if (!estimated_h.ok())
      return estimated_h.error();

    // Constant stepsize at the moment
    const SimdScalar h = [&]() {
      SimdScalar s = SimdScalar::Zero();

      for (auto i = 0ul; i < stepping.numComponents; ++i) {
        // h = 0 if surface not reachable, effectively suppress any progress
        if (stepping.status[i] == Intersection3D::Status::reachable) {
          // make sure we get the correct minimal stepsize
          s[i] = std::min(*estimated_h,
                          static_cast<ActsScalar>(stepping.stepSizes[i]));
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

    if (!stepping.extension.k2(state, MultiProxyStepper{}, navigator, sd.k2, sd.B_middle,
                               sd.kQoP, half_h, sd.k1)) {
      return EigenStepperError::StepInvalid;
    }

    // check for nan
    for (int i = 0; i < 3; ++i) {
      assert(!sd.k2[i].isNaN().any() && "k2 contains nan");
    }

    // Third Runge-Kutta point
    if (!stepping.extension.k3(state, MultiProxyStepper{}, navigator, sd.k3, sd.B_middle,
                               sd.kQoP, half_h, sd.k2)) {
      return EigenStepperError::StepInvalid;
    }

    // check for nan
    for (int i = 0; i < 3; ++i) {
      assert(!sd.k3[i].isNaN().any() && "k3 contains nan");
    }

    // Last Runge-Kutta point
    const SimdVector3 pos2 = pos + h * dir + h2 * 0.5 * sd.k3;
    sd.B_last = getMultiField(stepping, pos2);

    if (!stepping.extension.k4(state, MultiProxyStepper{}, navigator, sd.k4, sd.B_last,
                               sd.kQoP, h, sd.k3)) {
      return EigenStepperError::StepInvalid;
    }

    // check for nan
    for (int i = 0; i < 3; ++i) {
      assert(!sd.k4[i].isNaN().any() && "k4 contains nan");
    }

    // When doing error propagation, update the associated Jacobian matrix
    if (stepping.covTransport) {
      // The step transport matrix in global coordinates
      SimdFreeMatrix D;
      if (!stepping.extension.finalize(state, MultiProxyStepper{}, navigator, h, D)) {
        return EigenStepperError::StepInvalid;
      }

      // for moment, only update the transport part
      stepping.jacTransport = D * stepping.jacTransport;
    } else {
      if (!stepping.extension.finalize(state, MultiProxyStepper{}, navigator, h)) {
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
      assert(!stepping.pars[i].isNaN().any() && "free parameters contain nan");

    // Compute average step and return
    const auto avg_step = h.sum() / NComponents;

    stepping.pathAccumulated += avg_step;
    return avg_step;
  }
};

}  // namespace Acts
