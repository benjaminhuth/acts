// This file is part of the Acts project.
//
// Copyright (C) 2023 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <boost/test/unit_test.hpp>

#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/MagneticField/NullBField.hpp"
#include "Acts/Propagator/MultiEigenStepperLoop.hpp"
#include "Acts/Propagator/MultiStepperAborters.hpp"
#include "Acts/Propagator/Navigator.hpp"

namespace Acts {
struct MultiStepperSurfaceReached;
}  // namespace Acts

using namespace Acts;
using namespace Acts::VectorHelpers;

template <typename T>
concept Castable = requires(T a) {
  { static_cast<double>(a) } -> std::convertible_to<double>;
};

template <typename T>
concept Eigenable = requires(T &a) {
  { a } -> std::convertible_to<Eigen::MatrixBase<T> &>;
};

#define CHECK_CLOSE_GENERIC(arg1, arg2, eps)             \
  do {                                                   \
    using T1 = decltype(arg1);                           \
    using T2 = decltype(arg2);                           \
    if constexpr (Eigenable<T1> && Eigenable<T2>) {      \
      BOOST_CHECK(arg1.isApprox(arg2, eps));             \
    } else if constexpr (Castable<T1> && Castable<T2>) { \
      auto a = static_cast<double>(arg1);                \
      auto b = static_cast<double>(arg2);                \
      BOOST_CHECK_CLOSE(a, b, eps);                      \
    }                                                    \
  } while (false)

#define CHECK_EIGEN_CLOSE(a, b, e) \
  do {                             \
    BOOST_CHECK(a.isApprox(b, e)); \
    if (not a.isApprox(b, e)) {    \
      std::cout << a << "\n\n";    \
      std::cout << b << "\n\n";    \
    }                              \
  } while (false)

template <typename T>
using components_t = typename T::components;

using SingleStepper = EigenStepper<StepperExtensionList<DefaultExtension>>;

struct MockNavigator {
  MockNavigator() = default;
};
struct Navigation {
  Navigation() = default;
};

struct Options {
  double tolerance = 1e-4;
  double stepSizeCutOff = 0.0;
  std::size_t maxRungeKuttaStepTrials = 10;
  Direction direction = Direction::Backward;
  const Acts::Logger &logger = Acts::getDummyLogger();
};

template <typename stepper_state_t>
struct DummyPropState {
  stepper_state_t &stepping;
  Options options;
  Navigation navigation;
  GeometryContext geoContext;

  DummyPropState(Direction direction, stepper_state_t &ss)
      : stepping(ss),
        options(Options{}),
        navigation(Navigation{}),
        geoContext(GeometryContext{}) {
    options.direction = direction;
  }
};

// Makes random bound parameters and covariance and a plane surface at {0,0,0}
// with normal {1,0,0}. Optionally some external fixed bound parameters can be
// supplied
auto makeDefaultBoundPars(bool cov = true, std::size_t n = 4,
                          std::optional<BoundVector> ext_pars = std::nullopt) {
  std::vector<std::tuple<double, BoundVector, std::optional<BoundSquareMatrix>>>
      cmps;
  using Opt = std::optional<BoundSquareMatrix>;

  auto make_random_sym_matrix = []() {
    auto c = BoundSquareMatrix::Random().eval();
    c *= c.transpose();
    return c;
  };

  for (auto i = 0ul; i < n; ++i) {
    cmps.push_back({1. / n, ext_pars ? *ext_pars : BoundVector::Random(),
                    cov ? Opt{make_random_sym_matrix()} : Opt{}});
  }

  auto surface = Acts::Surface::makeShared<Acts::PlaneSurface>(
      Vector3::Zero(), Vector3{1., 0., 0.});

  return MultiComponentBoundTrackParameters(surface, cmps,
                                            ParticleHypothesis::pion());
}

/////////////////////////////////////////////////////
// Tester struct
/////////////////////////////////////////////////////
struct MultiStepperTester {
  const MockNavigator mockNavigator;

  const MagneticFieldContext magCtx;
  const GeometryContext geoCtx;

  const double defaultStepSize = 12.3;
  const Direction defaultNDir = Direction::Backward;
  const ParticleHypothesis particleHypothesis = ParticleHypothesis::pion();

  const std::shared_ptr<MagneticFieldProvider> defaultBField =
      std::make_shared<ConstantBField>(Vector3(1.0, 2.5, 2.0));
  const std::shared_ptr<MagneticFieldProvider> defaultNullBField =
      std::make_shared<ConstantBField>(Vector3(0.0, 0.0, 0.0));

  double epsilon = 1.e-8;

  MultiStepperTester(double e = 1.e-8) : epsilon(e) {}

  //////////////////////////////////////////////////////
  /// Test the construction of the MultiStepper::State
  //////////////////////////////////////////////////////
  template <typename multi_stepper_t, bool Cov>
  void test_multi_stepper_state() const {
    using MultiState = typename multi_stepper_t::State;
    using MultiStepper = multi_stepper_t;

    constexpr std::size_t N = 4;
    const auto multi_pars = makeDefaultBoundPars(Cov, N, BoundVector::Ones());

    MultiState state(geoCtx, magCtx, defaultBField, multi_pars,
                     defaultStepSize);

    MultiStepper ms(defaultBField);

    BOOST_CHECK_EQUAL(N, ms.numberComponents(state));

    // Test the result & compare with the input/test for reasonable members
    auto const_iterable = ms.constComponentIterable(state);
    for (const auto cmp : const_iterable) {
      CHECK_EIGEN_CLOSE(cmp.jacTransport(), FreeMatrix::Identity(), epsilon);
      CHECK_EIGEN_CLOSE(cmp.derivative(), FreeVector::Zero(), epsilon);
      if constexpr (not Cov) {
        CHECK_EIGEN_CLOSE(cmp.jacToGlobal(), BoundToFreeMatrix::Zero(),
                          epsilon);
        CHECK_EIGEN_CLOSE(cmp.cov(), BoundSquareMatrix::Zero(), epsilon);
      }
    }

    BOOST_CHECK_CLOSE(state.pathAccumulated, 0., epsilon);
    for (const auto cmp : const_iterable) {
      BOOST_CHECK_CLOSE(cmp.pathAccumulated(), 0., epsilon);
    }

    // covTransport in the MultiEigenStepperLoop is redundant and
    // thus not part of the interface. However, we want to check them for
    // consistency.
    if constexpr (Acts::Concepts::exists<components_t, MultiState>) {
      BOOST_CHECK(not state.covTransport);
      for (const auto &cmp : state.components) {
        BOOST_CHECK(cmp.state.covTransport == Cov);
      }
    }
  }

  template <typename multi_stepper_t>
  void test_multi_stepper_state_invalid() const {
    using MultiState = typename multi_stepper_t::State;

    // Empty component vector
    const auto multi_pars = makeDefaultBoundPars(false, 0);

    BOOST_CHECK_THROW(
        MultiState(geoCtx, magCtx, defaultBField, multi_pars, defaultStepSize),
        std::invalid_argument);
  }

  /////////////////////////////
  // Test stepsize accessors
  /////////////////////////////

  // TODO do this later, when we introduce the MultiEigenStepperSIMD, which
  // there needs new interfaces...

  ////////////////////////////////////////////////////
  // Test the modifying accessors to the components
  ////////////////////////////////////////////////////

  template <typename multi_stepper_t>
  void test_components_modifying_accessors() const {
    using MultiState = typename multi_stepper_t::State;
    using MultiStepper = multi_stepper_t;

    const auto multi_pars = makeDefaultBoundPars();

    MultiState mutable_multi_state(geoCtx, magCtx, defaultBField, multi_pars,
                                   defaultStepSize);
    const MultiState const_multi_state(geoCtx, magCtx, defaultBField,
                                       multi_pars, defaultStepSize);

    MultiStepper multi_stepper(defaultBField);

    // Here test the mutable overloads of the mutable iterable
    auto components = multi_stepper.componentIterable(mutable_multi_state);
    for (auto cmp : components) {
      cmp.status() = static_cast<Intersection3D::Status>(
          static_cast<int>(cmp.status()) + 1);
      cmp.pathAccumulated() *= 2.0;
      cmp.weight() *= 2.0;
      cmp.pars() *= 2.0;
      cmp.cov() *= 2.0;
      cmp.jacTransport() *= 2.0;
      cmp.derivative() *= 2.0;
      cmp.jacobian() *= 2.0;
      cmp.jacToGlobal() *= 2.0;
    }

    auto mutable_state_iterable =
        multi_stepper.componentIterable(mutable_multi_state);
    // Here test the const iterable
    auto const_state_iterable =
        multi_stepper.constComponentIterable(const_multi_state);

    auto mstate_it = mutable_state_iterable.begin();
    auto cstate_it = const_state_iterable.begin();
    for (; cstate_it != const_state_iterable.end(); ++mstate_it, ++cstate_it) {
      const auto mstate_cmp = *mstate_it;
      auto cstate_cmp = *cstate_it;

      BOOST_CHECK_EQUAL(static_cast<int>(mstate_cmp.status()),
                        1 + static_cast<int>(cstate_cmp.status()));

      CHECK_CLOSE_GENERIC(mstate_cmp.pathAccumulated(),
                          2.0 * cstate_cmp.pathAccumulated(), epsilon);
      CHECK_CLOSE_GENERIC(mstate_cmp.weight(), 2.0 * cstate_cmp.weight(),
                          epsilon);
      CHECK_CLOSE_GENERIC(mstate_cmp.pars(), 2.0 * cstate_cmp.pars(), epsilon);
      CHECK_CLOSE_GENERIC(mstate_cmp.cov(), 2.0 * cstate_cmp.cov(), epsilon);
      CHECK_CLOSE_GENERIC(mstate_cmp.jacTransport(),
                          2.0 * cstate_cmp.jacTransport(), epsilon);
      CHECK_CLOSE_GENERIC(mstate_cmp.derivative(),
                          2.0 * cstate_cmp.derivative(), epsilon);
      CHECK_CLOSE_GENERIC(mstate_cmp.jacobian(), 2.0 * cstate_cmp.jacobian(),
                          epsilon);
      CHECK_CLOSE_GENERIC(mstate_cmp.jacToGlobal(),
                          2.0 * cstate_cmp.jacToGlobal(), epsilon);
    }
  }

  /////////////////////////////////////////////
  // Test if the surface status update works
  /////////////////////////////////////////////
  template <typename multi_stepper_t>
  void test_multi_stepper_surface_status_update() const {
    using MultiState = typename multi_stepper_t::State;
    using MultiStepper = multi_stepper_t;

    auto start_surface = Acts::Surface::makeShared<Acts::PlaneSurface>(
        Vector3::Zero(), Vector3{1.0, 0.0, 0.0});

    auto right_surface = Acts::Surface::makeShared<Acts::PlaneSurface>(
        Vector3{1.0, 0.0, 0.0}, Vector3{1.0, 0.0, 0.0});

    std::vector<
        std::tuple<double, BoundVector, std::optional<BoundSquareMatrix>>>
        cmps(2, {0.5, BoundVector::Zero(), std::nullopt});
    std::get<BoundVector>(cmps[0])[eBoundTheta] = M_PI_2;
    std::get<BoundVector>(cmps[1])[eBoundTheta] = -M_PI_2;
    std::get<BoundVector>(cmps[0])[eBoundQOverP] = 1.0;
    std::get<BoundVector>(cmps[1])[eBoundQOverP] = 1.0;

    MultiComponentBoundTrackParameters multi_pars(start_surface, cmps,
                                                  particleHypothesis);

    BOOST_REQUIRE(std::get<1>(multi_pars[0])
                      .direction()
                      .isApprox(Vector3{1.0, 0.0, 0.0}, epsilon));
    BOOST_REQUIRE(std::get<1>(multi_pars[1])
                      .direction()
                      .isApprox(Vector3{-1.0, 0.0, 0.0}, epsilon));

    MultiState multi_state(geoCtx, magCtx, defaultNullBField, multi_pars,
                           defaultStepSize);
    SingleStepper::State single_state(
        geoCtx, defaultNullBField->makeCache(magCtx),
        std::get<1>(multi_pars[0]), defaultStepSize);

    MultiStepper multi_stepper(defaultNullBField);
    SingleStepper single_stepper(defaultNullBField);

    // Update surface status and check
    {
      auto status = multi_stepper.updateSurfaceStatus(
          multi_state, *right_surface, 0, Direction::Forward, false);

      BOOST_CHECK(status == Intersection3D::Status::reachable);

      auto cmp_iterable = multi_stepper.constComponentIterable(multi_state);

      BOOST_CHECK((*cmp_iterable.begin()).status() ==
                  Intersection3D::Status::reachable);
      BOOST_CHECK((*(++cmp_iterable.begin())).status() ==
                  Intersection3D::Status::missed);
    }

    // Step forward now
    {
      auto multi_prop_state = DummyPropState(Direction::Forward, multi_state);
      multi_stepper.step(multi_prop_state, mockNavigator);

      // Single stepper
      auto single_prop_state = DummyPropState(Direction::Forward, single_state);
      single_stepper.step(single_prop_state, mockNavigator);
    }

    // Update surface status and check again
    {
      auto status = multi_stepper.updateSurfaceStatus(
          multi_state, *right_surface, 0, Direction::Forward, false);

      BOOST_CHECK(status == Intersection3D::Status::onSurface);

      auto cmp_iterable = multi_stepper.constComponentIterable(multi_state);

      BOOST_CHECK((*cmp_iterable.begin()).status() ==
                  Intersection3D::Status::onSurface);
      BOOST_CHECK((*(++cmp_iterable.begin())).status() ==
                  Intersection3D::Status::missed);
    }

    // Start surface should be unreachable
    {
      auto status = multi_stepper.updateSurfaceStatus(
          multi_state, *start_surface, 0, Direction::Forward, false);

      BOOST_CHECK(status == Intersection3D::Status::unreachable);

      auto cmp_iterable = multi_stepper.constComponentIterable(multi_state);

      BOOST_CHECK((*cmp_iterable.begin()).status() ==
                  Intersection3D::Status::unreachable);
      BOOST_CHECK((*(++cmp_iterable.begin())).status() ==
                  Intersection3D::Status::unreachable);
    }
  }

  //////////////////////////////////
  // Test Bound state computations
  //////////////////////////////////
  template <typename multi_stepper_t>
  void test_component_bound_state() const {
    using MultiState = typename multi_stepper_t::State;
    using MultiStepper = multi_stepper_t;

    auto start_surface = Acts::Surface::makeShared<Acts::PlaneSurface>(
        Vector3::Zero(), Vector3{1.0, 0.0, 0.0});

    auto right_surface = Acts::Surface::makeShared<Acts::PlaneSurface>(
        Vector3{1.0, 0.0, 0.0}, Vector3{1.0, 0.0, 0.0});

    std::vector<
        std::tuple<double, BoundVector, std::optional<BoundSquareMatrix>>>
        cmps(2, {0.5, BoundVector::Zero(), std::nullopt});
    std::get<BoundVector>(cmps[0])[eBoundTheta] = M_PI_2;
    std::get<BoundVector>(cmps[1])[eBoundTheta] = -M_PI_2;
    std::get<BoundVector>(cmps[0])[eBoundQOverP] = 1.0;
    std::get<BoundVector>(cmps[1])[eBoundQOverP] = 1.0;

    MultiComponentBoundTrackParameters multi_pars(start_surface, cmps,
                                                  particleHypothesis);

    BOOST_REQUIRE(std::get<1>(multi_pars[0])
                      .direction()
                      .isApprox(Vector3{1.0, 0.0, 0.0}, 1.e-10));
    BOOST_REQUIRE(std::get<1>(multi_pars[1])
                      .direction()
                      .isApprox(Vector3{-1.0, 0.0, 0.0}, 1.e-10));

    MultiState multi_state(geoCtx, magCtx, defaultNullBField, multi_pars,
                           defaultStepSize);
    SingleStepper::State single_state(
        geoCtx, defaultNullBField->makeCache(magCtx),
        std::get<1>(multi_pars[0]), defaultStepSize);

    MultiStepper multi_stepper(defaultNullBField);
    SingleStepper single_stepper(defaultNullBField);

    // Step forward now
    {
      multi_stepper.updateSurfaceStatus(multi_state, *right_surface, 0,
                                        Direction::Forward, false);
      auto multi_prop_state = DummyPropState(Direction::Forward, multi_state);
      multi_stepper.step(multi_prop_state, mockNavigator);

      // Single stepper
      single_stepper.updateSurfaceStatus(single_state, *right_surface, 0,
                                         Direction::Forward, false);
      auto single_prop_state = DummyPropState(Direction::Forward, single_state);
      single_stepper.step(single_prop_state, mockNavigator);
    }

    // Check if on surface
    {
      auto sstatus = single_stepper.updateSurfaceStatus(
          single_state, *right_surface, 0, Direction::Forward, false);
      BOOST_CHECK(sstatus == Acts::Intersection3D::Status::onSurface);

      auto mstatus = multi_stepper.updateSurfaceStatus(
          multi_state, *right_surface, 0, Direction::Forward, false);
      BOOST_CHECK(mstatus == Acts::Intersection3D::Status::onSurface);
    }

    // Check component-wise bound-state
    {
      auto single_bound_state = single_stepper.boundState(
          single_state, *right_surface, true, FreeToBoundCorrection(false));
      BOOST_REQUIRE(single_bound_state.ok());

      auto cmp_iterable = multi_stepper.componentIterable(multi_state);

      auto ok_bound_state =
          (*cmp_iterable.begin())
              .boundState(*right_surface, true, FreeToBoundCorrection(false));
      BOOST_REQUIRE(ok_bound_state.ok());

      const auto &[spars, sjac, spl] = *single_bound_state;
      const auto &[mpars, mjac, mpl] = *ok_bound_state;

      BOOST_CHECK(spars.parameters().isApprox(mpars.parameters(), epsilon));
      BOOST_CHECK(spars.covariance() == std::nullopt);
      BOOST_CHECK(mpars.covariance() == std::nullopt);

      auto failed_bound_state =
          (*(++cmp_iterable.begin()))
              .boundState(*right_surface, true, FreeToBoundCorrection(false));
      BOOST_CHECK(not failed_bound_state.ok());
    }
  }

  template <typename multi_stepper_t>
  void test_combined_bound_state_function() const {
    using MultiState = typename multi_stepper_t::State;
    using MultiStepper = multi_stepper_t;

    auto surface = Acts::Surface::makeShared<Acts::PlaneSurface>(
        Vector3::Zero(), Vector3{1.0, 0.0, 0.0});

    // Use Ones() here, so that the angles are in correct range
    const auto pars = BoundVector::Ones().eval();
    const auto cov = []() {
      auto c = BoundSquareMatrix::Random().eval();
      c *= c.transpose();
      return c;
    }();

    std::vector<
        std::tuple<double, BoundVector, std::optional<BoundSquareMatrix>>>
        cmps(4, {0.25, pars, cov});

    MultiComponentBoundTrackParameters multi_pars(surface, cmps,
                                                  particleHypothesis);
    MultiState multi_state(geoCtx, magCtx, defaultBField, multi_pars,
                           defaultStepSize);
    MultiStepper multi_stepper(defaultBField);

    auto res = multi_stepper.boundState(multi_state, *surface, true,
                                        FreeToBoundCorrection(false));

    BOOST_REQUIRE(res.ok());

    const auto [bound_pars, jacobian, pathLength] = *res;

    CHECK_EIGEN_CLOSE(jacobian, decltype(jacobian)::Zero(), epsilon);
    BOOST_CHECK(pathLength == 0.0);
    CHECK_EIGEN_CLOSE(bound_pars.parameters(), pars, epsilon);
    CHECK_EIGEN_CLOSE((*bound_pars.covariance()), cov, epsilon);
  }

  //////////////////////////////////////////////////
  // Test the combined curvilinear state function
  //////////////////////////////////////////////////
  template <typename multi_stepper_t>
  void test_combined_curvilinear_state_function() const {
    using MultiState = typename multi_stepper_t::State;
    using MultiStepper = multi_stepper_t;

    auto surface = Acts::Surface::makeShared<Acts::PlaneSurface>(
        Vector3::Zero(), Vector3{1.0, 0.0, 0.0});

    // Use Ones() here, so that the angles are in correct range
    const auto pars = BoundVector::Ones().eval();
    const auto cov = []() {
      auto c = BoundSquareMatrix::Random().eval();
      c *= c.transpose();
      return c;
    }();

    std::vector<
        std::tuple<double, BoundVector, std::optional<BoundSquareMatrix>>>
        cmps(4, {0.25, pars, cov});
    BoundTrackParameters check_pars(surface, pars, cov, particleHypothesis);

    MultiComponentBoundTrackParameters multi_pars(surface, cmps,
                                                  particleHypothesis);
    MultiState multi_state(geoCtx, magCtx, defaultBField, multi_pars,
                           defaultStepSize);
    MultiStepper multi_stepper(defaultBField);

    const auto [curv_pars, jac, pathLength] =
        multi_stepper.curvilinearState(multi_state);

    BOOST_CHECK(curv_pars.fourPosition(multi_state.geoContext)
                    .isApprox(check_pars.fourPosition(multi_state.geoContext),
                              epsilon));
    BOOST_CHECK(
        curv_pars.direction().isApprox(check_pars.direction(), epsilon));
    BOOST_CHECK_CLOSE(curv_pars.absoluteMomentum(),
                      check_pars.absoluteMomentum(), epsilon);
    BOOST_CHECK_CLOSE(curv_pars.charge(), check_pars.charge(), epsilon);
  }

  ////////////////////////////////////
  // Test single component interface
  ////////////////////////////////////

  template <typename multi_stepper_t>
  void test_single_component_interface_function() const {
    using MultiState = typename multi_stepper_t::State;
    using MultiStepper = multi_stepper_t;

    std::vector<
        std::tuple<double, BoundVector, std::optional<BoundSquareMatrix>>>
        cmps;
    for (int i = 0; i < 4; ++i) {
      cmps.push_back(
          {0.25, BoundVector::Random(), BoundSquareMatrix::Random()});
    }

    auto surface = Acts::Surface::makeShared<Acts::PlaneSurface>(
        Vector3::Zero(), Vector3::Ones().normalized());

    MultiComponentBoundTrackParameters multi_pars(surface, cmps,
                                                  particleHypothesis);

    MultiState multi_state(geoCtx, magCtx, defaultBField, multi_pars,
                           defaultStepSize);

    MultiStepper multi_stepper(defaultBField);

    DummyPropState multi_prop_state(defaultNDir, multi_state);

    // Check at least some properties at the moment
    auto check = [&](auto cmp) {
      auto sstepper = cmp.singleStepper(multi_stepper);
      auto &sstepping = cmp.singleState(multi_prop_state).stepping;

      CHECK_CLOSE_GENERIC(sstepper.position(sstepping),
                          cmp.pars().template segment<3>(eFreePos0), epsilon);
      CHECK_CLOSE_GENERIC(sstepper.direction(sstepping),
                          cmp.pars().template segment<3>(eFreeDir0), epsilon);
      CHECK_CLOSE_GENERIC(sstepper.time(sstepping), cmp.pars()[eFreeTime],
                          epsilon);
      CHECK_CLOSE_GENERIC(sstepper.qOverP(sstepping), cmp.pars()[eFreeQOverP],
                          epsilon);
    };

    for (const auto cmp : multi_stepper.constComponentIterable(multi_state)) {
      check(cmp);
    }

    for (auto cmp : multi_stepper.componentIterable(multi_state)) {
      check(cmp);
    }
  }

  //////////////////////////////
  // Remove and add components
  //////////////////////////////

  template <typename multi_stepper_t>
  void remove_add_components_function() const {
    using MultiState = typename multi_stepper_t::State;
    using MultiStepper = multi_stepper_t;

    const auto multi_pars = makeDefaultBoundPars(true, 4);
    const auto &surface = multi_pars.referenceSurface();

    MultiState multi_state(geoCtx, magCtx, defaultBField, multi_pars,
                           defaultStepSize);

    MultiStepper multi_stepper(defaultBField);

    auto copy = [&](auto from, auto to) {
      const auto &[w, p, c] = from;

      to.weight() = w;
      to.pars() = Acts::detail::transformBoundToFreeParameters(
          surface, multi_state.geoContext, p);
      to.cov() = *c;
    };

    // Effectively add components
    {
      const auto new_pars = makeDefaultBoundPars(true, 6);

      const auto &cmps = new_pars.components();
      multi_stepper.update(multi_state, surface, cmps.begin(), cmps.end(),
                           copy);
      BOOST_CHECK_EQUAL(multi_stepper.numberComponents(multi_state), 6);
    }

    // Effectively remove components
    {
      const auto new_pars = makeDefaultBoundPars(true, 2);

      const auto &cmps = new_pars.components();
      multi_stepper.update(multi_state, surface, cmps.begin(), cmps.end(),
                           copy);
      BOOST_CHECK_EQUAL(multi_stepper.numberComponents(multi_state), 2);
    }

    // Clear
    {
      std::vector<int> empty;
      auto dummyCopy = [](auto /*from*/, auto /*to*/) {};
      multi_stepper.update(multi_state, surface, empty.begin(), empty.end(),
                           dummyCopy);
      BOOST_CHECK_EQUAL(multi_stepper.numberComponents(multi_state), 0);
    }
  }

  ////////////////////////////////////////////////////////////////////////
  // Compare the Multi-Stepper against the Eigen-Stepper for consistency
  ////////////////////////////////////////////////////////////////////////
  template <typename multi_stepper_t>
  void test_multi_stepper_vs_eigen_stepper() const {
    using MultiState = typename multi_stepper_t::State;
    using MultiStepper = multi_stepper_t;

    const BoundVector pars = BoundVector::Ones();
    const BoundSquareMatrix cov = BoundSquareMatrix::Identity();

    std::vector<
        std::tuple<double, BoundVector, std::optional<BoundSquareMatrix>>>
        cmps(4, {0.25, pars, cov});

    auto surface = Acts::Surface::makeShared<Acts::PlaneSurface>(
        Vector3::Zero(), Vector3::Ones().normalized());

    MultiComponentBoundTrackParameters multi_pars(surface, cmps,
                                                  particleHypothesis);
    BoundTrackParameters single_pars(surface, pars, cov, particleHypothesis);

    MultiState multi_state(geoCtx, magCtx, defaultNullBField, multi_pars,
                           defaultStepSize);
    SingleStepper::State single_state(geoCtx,
                                      defaultNullBField->makeCache(magCtx),
                                      single_pars, defaultStepSize);

    MultiStepper multi_stepper(defaultBField);
    SingleStepper single_stepper(defaultBField);

    for (auto cmp : multi_stepper.componentIterable(multi_state)) {
      cmp.status() = Acts::Intersection3D::Status::reachable;
    }

    // Do some steps and check that the results match
    for (int i = 0; i < 10; ++i) {
      // Single stepper
      auto single_prop_state = DummyPropState(defaultNDir, single_state);
      auto single_result =
          single_stepper.step(single_prop_state, mockNavigator);
      single_stepper.transportCovarianceToCurvilinear(single_state);

      // Multi stepper;
      auto multi_prop_state = DummyPropState(defaultNDir, multi_state);
      auto multi_result = multi_stepper.step(multi_prop_state, mockNavigator);
      multi_stepper.transportCovarianceToCurvilinear(multi_state);

      // Check equality
      BOOST_REQUIRE(multi_result.ok() == true);
      BOOST_REQUIRE(multi_result.ok() == single_result.ok());

      BOOST_CHECK_CLOSE(*single_result, *multi_result, epsilon);

      for (const auto cmp : multi_stepper.constComponentIterable(multi_state)) {
        CHECK_EIGEN_CLOSE(cmp.pars(), single_state.pars, epsilon);
        CHECK_EIGEN_CLOSE(cmp.cov(), single_state.cov, epsilon);
        CHECK_EIGEN_CLOSE(cmp.jacTransport(), single_state.jacTransport,
                          epsilon);
        CHECK_EIGEN_CLOSE(cmp.jacToGlobal(), single_state.jacToGlobal, epsilon);
        CHECK_EIGEN_CLOSE(cmp.derivative(), single_state.derivative, epsilon);
        BOOST_CHECK_CLOSE(cmp.pathAccumulated(), single_state.pathAccumulated,
                          epsilon);
      }
    }
  }

  //////////////////////////////////////////////////
  // Instantiate a Propagator with the MultiStepper
  //////////////////////////////////////////////////

  template <typename multi_stepper_t>
  void propagator_instatiation_test_function() const {
    auto bField = std::make_shared<NullBField>();
    multi_stepper_t multi_stepper(bField);
    Propagator<multi_stepper_t, Navigator> propagator(
        std::move(multi_stepper), Navigator{Navigator::Config{}});

    auto surface = Acts::Surface::makeShared<Acts::PlaneSurface>(
        Vector3::Zero(), Vector3{1.0, 0.0, 0.0});
    PropagatorOptions options(geoCtx, magCtx);

    std::vector<
        std::tuple<double, BoundVector, std::optional<BoundSquareMatrix>>>
        cmps(4, {0.25, BoundVector::Ones().eval(),
                 BoundSquareMatrix::Identity().eval()});
    MultiComponentBoundTrackParameters pars(surface, cmps, particleHypothesis);

    // This only checks that this compiles, not that it runs without errors
    // @TODO: Add test that checks the target aborter works correctly

    // Instantiate with target
    using type_a =
        decltype(propagator.template propagate<
                 decltype(pars), decltype(options), MultiStepperSurfaceReached>(
            pars, *surface, options));

    // Instantiate without target
    using tybe_b = decltype(propagator.propagate(pars, options));
  }
};
