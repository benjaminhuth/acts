// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <cstdint>

namespace Acts::detail {

/// Proxy class that redirects calls to multi-calls
template <typename simd_stepper_t>
struct MultiProxyStepper {
  using State = typename simd_stepper_t::State;
  using Stepper = simd_stepper_t;

  auto direction(const State& state) const {
    return Stepper::multiDirection(state);
  }
  auto position(const State& state) const {
    return Stepper::multiPosition(state);
  }
  auto charge(const State& state) const {
    return Stepper::charge_static(state);
  }
  auto absoluteMomentum(const State& state) const {
    return Stepper::multiAbsoluteMomentum(state);
  }
  auto reducedPosition(const State& state) const {
    return Stepper::Reducer::position(state);
  }
  auto qOverP(const State& state) const { return Stepper::multiQOverP(state); }
  const auto& particleHypothesis(const State& state) const {
    return state.particleHypothesis;
  }
};

/// Proxy stepper which acts of a specific component
template <typename simd_stepper_t>
class SingleProxyStepper {
  const std::size_t m_i = 0;
  const double m_olimit = 0.0;
  
public:
  using Stepper = simd_stepper_t;
  using State = typename Stepper::State;
  
  SingleProxyStepper(std::size_t i, double olimit) : m_i(i), m_olimit(olimit) {}

  auto direction(const State& state) const {
    return Stepper::direction(m_i, state);
  }
  auto position(const State& state) const {
    return Stepper::position(m_i, state);
  }
  auto charge(const State& state) const {
    return Stepper::charge_static(state);
  }
  auto absoluteMomentum(const State& state) const {
    return Stepper::absoluteMomentum(m_i, state);
  }
  auto qOverP(const State& state) const { return Stepper::qOverP(m_i, state); }
  auto time(const State& state) const { return Stepper::time(m_i, state); }
  auto overstepLimit(const State&) const { return m_olimit; }
  void setStepSize(State& state, double stepSize,
                   ConstrainedStep::Type stype = ConstrainedStep::actor,
                   bool release = true) const {
    state.stepSizes[m_i].update(stepSize, stype, release);
  }
  void releaseStepSize(State& state) const {
    state.stepSizes[m_i].release(ConstrainedStep::actor);
  };
  auto getStepSize(const State& state, ConstrainedStep::Type stype) const {
    return state.stepSizes[m_i].value(stype);
  };
};

/// A template class which contains all const member functions, that should be
/// available both in the mutable ComponentProxy and the ConstComponentProxy.
/// @tparam component_t Must be a const or mutable State::Component.
template <typename state_t, typename simd_stepper_t>
struct SimdComponentProxyBase {
  state_t& m_state;
  const std::size_t m_i;

 public:
  SimdComponentProxyBase(state_t& s, std::size_t i) : m_state(s), m_i(i) {
    // assert(i < NComponents && "Cannot create proxy: out of range");
  }

  auto index() const { return m_i; }
  auto weight() const { return m_state.weights[m_i]; }
  auto charge() const { return m_state.q; }
  auto pathAccumulated() const { return m_state.pathAccumulated; }
  auto status() const { return m_state.status[m_i]; }
  auto pars() const { return extract(m_state.pars, m_i); }
  auto derivative() const { return extract(m_state.derivative, m_i); }
  auto jacTransport() const { return extract(m_state.jacTransport, m_i); }
  const auto& cov() const { return m_state.covs[m_i]; }
  const auto& jacobian() const { return m_state.jacobians[m_i]; }
  const auto& jacToGlobal() const { return m_state.jacToGlobals[m_i]; }

  template <typename propagator_state_t>
  const auto& singleState(const propagator_state_t& state) const {
    return state;
  }

  auto singleStepper(const simd_stepper_t& stepper) const {
    // the stepper returns -olimit, so we invert sign again
    return SingleProxyStepper<simd_stepper_t>(m_i, -stepper.overstepLimit(m_state));
  }
};

/// A proxy struct which allows access to a single component of the
/// multi-component state. It has the semantics of a mutable reference, i.e.
/// it requires a mutable reference of the single-component state it
/// represents
template <typename state_t, typename simd_stepper_t>
struct SimdComponentProxy : SimdComponentProxyBase<state_t, simd_stepper_t> {
  using Base = SimdComponentProxyBase<state_t, simd_stepper_t>;

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

  SimdComponentProxy(state_t& s, std::size_t i) : Base(s, i) {}

  auto weight() { return m_state.weights[m_i]; }
  auto& charge() { return m_state.q; }
  auto& pathAccumulated() { return m_state.pathAccumulated; }
  auto& status() { return m_state.status[m_i]; }
  auto pars() { return extract(m_state.pars, m_i); }
  auto derivative() { return extract(m_state.derivative, m_i); }
  auto jacTransport() { return extract(m_state.jacTransport, m_i); }
  auto& cov() { return m_state.covs[m_i]; }
  auto& jacobian() { return m_state.jacobians[m_i]; }
  auto& jacToGlobal() { return m_state.jacToGlobals[m_i]; }

  template <typename propagator_state_t>
  auto& singleState(propagator_state_t& state) {
    return state;
  }

  auto boundState(const Surface& surface, bool transportCov,
                  const FreeToBoundCorrection& ftbc) {
    using R = Result<std::tuple<BoundTrackParameters, BoundMatrix, double>>;

    if (status() != Intersection3D::Status::onSurface) {
      return R{MultiStepperError::ComponentNotOnSurface};
    }

    // TODO template detail::bounState(...) on Eigen::MatrixBase<T> to allow
    // the Map types to go in directely
    Acts::FreeMatrix jacTransport = this->jacTransport();
    Acts::FreeVector derivative = this->derivative();
    Acts::FreeVector pars = this->pars();

    auto bs = detail::boundState(m_state.geoContext, cov(), jacobian(),
                                 jacTransport, derivative, jacToGlobal(), pars,
                                 m_state.particleHypothesis, m_state.covTransport && transportCov,
                                 m_state.pathAccumulated, surface, ftbc);

    this->jacTransport() = jacTransport;
    this->derivative() = derivative;
    this->pars() = pars;

    if (!bs.ok()) {
      return R{bs.error()};
    }

    return R{std::move(bs.value())};
  }
  

  auto curvilinearState(bool transportCov) {
    Acts::FreeMatrix jacTransport = this->jacTransport();
    Acts::FreeVector derivative = this->derivative();
    Acts::FreeVector pars = this->pars();

    auto bs = detail::curvilinearState(
        cov(), jacobian(), jacTransport, derivative, jacToGlobal(), pars,
        m_state.particleHypothesis, m_state.covTransport && transportCov,
        m_state.pathAccumulated);

    this->jacTransport() = jacTransport;
    this->derivative() = derivative;
    this->pars() = pars;

    return bs;
  }
};

}  // namespace Acts::detail
