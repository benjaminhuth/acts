// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Utilities/Logger.hpp"

namespace Acts {

template <typename E, typename R, typename A>
auto MultiEigenStepperLoop<E, R, A>::boundState(
    State& state, const Surface& surface, bool transportCov,
    const FreeToBoundCorrection& freeToBoundCorrection) const
    -> Result<BoundState> {
  assert(!state.components.empty());

  if (numberComponents(state) == 1) {
    return SingleStepper::boundState(state.components.front().state, surface,
                                     transportCov, freeToBoundCorrection);
  } else if (m_finalReductionMethod == FinalReductionMethod::eMaxWeight) {
    auto cmpIt = std::max_element(
        state.components.begin(), state.components.end(),
        [](const auto& a, const auto& b) { return a.weight < b.weight; });

    return SingleStepper::boundState(cmpIt->state, surface, transportCov,
                                     freeToBoundCorrection);
  } else {
    SmallVector<std::tuple<double, BoundVector, BoundSymMatrix>> states;
    double accumulatedPathLength = 0.0;
    int failedBoundTransforms = 0;

    for (auto i = 0ul; i < numberComponents(state); ++i) {
      auto bs = SingleStepper::boundState(state.components[i].state, surface,
                                          transportCov, freeToBoundCorrection);

      if (bs.ok()) {
        const auto& btp = std::get<BoundTrackParameters>(*bs);
        states.emplace_back(
            state.components[i].weight, btp.parameters(),
            btp.covariance().value_or(Acts::BoundSymMatrix::Zero()));
        accumulatedPathLength +=
            std::get<double>(*bs) * state.components[i].weight;
      } else {
        failedBoundTransforms++;
      }
    }

    if (states.empty()) {
      return MultiStepperError::AllComponentsConversionToBoundFailed;
    }

    if (failedBoundTransforms > 0) {
      return MultiStepperError::SomeComponentsConversionToBoundFailed;
    }

    auto [params, cov] =
        detail::angleDescriptionSwitch(surface, [&](const auto& desc) {
          return detail::combineGaussianMixture(states, Acts::Identity{}, desc);
        });

    std::optional<BoundSymMatrix> finalCov = std::nullopt;
    if (cov != BoundSymMatrix::Zero()) {
      finalCov = cov;
    }

    return BoundState{BoundTrackParameters(surface.getSharedPtr(), params, cov),
                      Jacobian::Zero(), accumulatedPathLength};
  }
}

template <typename E, typename R, typename A>
auto MultiEigenStepperLoop<E, R, A>::curvilinearState(State& state,
                                                      bool transportCov) const
    -> CurvilinearState {
  assert(!state.components.empty());

  if (numberComponents(state) == 1) {
    return SingleStepper::curvilinearState(state.components.front().state,
                                           transportCov);
  } else if (m_finalReductionMethod == FinalReductionMethod::eMaxWeight) {
    auto cmpIt = std::max_element(
        state.components.begin(), state.components.end(),
        [](const auto& a, const auto& b) { return a.weight < b.weight; });

    return SingleStepper::curvilinearState(cmpIt->state, transportCov);
  } else {
    Vector4 pos4 = Vector4::Zero();
    Vector3 dir = Vector3::Zero();
    ActsScalar qop = 0.0;
    BoundSymMatrix cov = BoundSymMatrix::Zero();
    ActsScalar pathLenth = 0.0;
    ActsScalar sumOfWeights = 0.0;

    for (auto i = 0ul; i < numberComponents(state); ++i) {
      const auto [cp, jac, pl] = SingleStepper::curvilinearState(
          state.components[i].state, transportCov);

      pos4 += state.components[i].weight * cp.fourPosition(state.geoContext);
      dir += state.components[i].weight * cp.unitDirection();
      qop += state.components[i].weight * (cp.charge() / cp.absoluteMomentum());
      if (cp.covariance()) {
        cov += state.components[i].weight * *cp.covariance();
      }
      pathLenth += state.components[i].weight * pathLenth;
      sumOfWeights += state.components[i].weight;
    }

    pos4 /= sumOfWeights;
    dir /= sumOfWeights;
    qop /= sumOfWeights;
    pathLenth /= sumOfWeights;
    cov /= sumOfWeights;

    std::optional<BoundSymMatrix> finalCov = std::nullopt;
    if (cov != BoundSymMatrix::Zero()) {
      finalCov = cov;
    }

    return CurvilinearState{
        CurvilinearTrackParameters(pos4, dir, qop, finalCov), Jacobian::Zero(),
        pathLenth};
  }
}

template <typename E, typename R, typename A>
std::size_t MultiEigenStepperLoop<E, R, A>::numberComponents(
    const State& state) const {
  return state.components.size();
}

template <typename E, typename R, typename A>
void MultiEigenStepperLoop<E, R, A>::removeMissedComponents(
    State& state) const {
  auto new_end = std::remove_if(
      state.components.begin(), state.components.end(), [](const auto& cmp) {
        return cmp.status == Intersection3D::Status::missed;
      });

  state.components.erase(new_end, state.components.end());
}

template <typename E, typename R, typename A>
void MultiEigenStepperLoop<E, R, A>::reweightComponents(State& state) const {
  ActsScalar sumOfWeights = 0.0;
  for (const auto& cmp : state.components) {
    sumOfWeights += cmp.weight;
  }
  for (auto& cmp : state.components) {
    cmp.weight /= sumOfWeights;
  }
}

template <typename E, typename R, typename A>
void MultiEigenStepperLoop<E, R, A>::clearComponents(State& state) const {
  state.components.clear();
}

template <typename E, typename R, typename A>
template <typename charge_t>
auto MultiEigenStepperLoop<E, R, A>::addComponent(
    State& state, const SingleBoundTrackParameters<charge_t>& pars,
    double weight) const -> Result<ComponentProxy> {
  state.components.push_back(
      {SingleState(state.geoContext,
                   SingleStepper::m_bField->makeCache(state.magContext), pars,
                   state.navDir),
       weight, Intersection3D::Status::onSurface});

  return ComponentProxy{state.components.back(), state};
}

template <typename E, typename R, typename A>
Intersection3D::Status MultiEigenStepperLoop<E, R, A>::updateSurfaceStatus(
    State& state, const Surface& surface, const BoundaryCheck& bcheck,
    const Logger& logger) const {
  using Status = Intersection3D::Status;

  std::array<int, 4> counts = {0, 0, 0, 0};

  for (auto& component : state.components) {
    component.status = detail::updateSingleSurfaceStatus<SingleStepper>(
        *this, component.state, surface, bcheck, logger);
    ++counts[static_cast<std::size_t>(component.status)];
  }

  // If at least one component is on a surface, we can remove all missed
  // components before the step. If not, we must keep them for the case that all
  // components miss and we need to retarget
  if (counts[static_cast<std::size_t>(Status::onSurface)] > 0) {
    removeMissedComponents(state);
    reweightComponents(state);
  }

  ACTS_VERBOSE("Component status wrt "
               << surface.geometryId() << " at {"
               << surface.center(state.geoContext).transpose() << "}:\t"
               << [&]() {
                    std::stringstream ss;
                    for (auto& component : state.components) {
                      ss << component.status << "\t";
                    }
                    return ss.str();
                  }());

  // Switch on stepCounter if one or more components reached a surface, but
  // some are still in progress of reaching the surface
  if (!state.stepCounterAfterFirstComponentOnSurface &&
      counts[static_cast<std::size_t>(Status::onSurface)] > 0 &&
      counts[static_cast<std::size_t>(Status::reachable)] > 0) {
    state.stepCounterAfterFirstComponentOnSurface = 0;
    ACTS_VERBOSE("started stepCounterAfterFirstComponentOnSurface");
  }

  // This is a 'any_of' criterium. As long as any of the components has a
  // certain state, this determines the total state (in the order of a
  // somewhat importance)
  if (counts[static_cast<std::size_t>(Status::reachable)] > 0) {
    return Status::reachable;
  } else if (counts[static_cast<std::size_t>(Status::onSurface)] > 0) {
    state.stepCounterAfterFirstComponentOnSurface.reset();
    return Status::onSurface;
  } else if (counts[static_cast<std::size_t>(Status::unreachable)] > 0) {
    return Status::unreachable;
  } else {
    return Status::missed;
  }
}

template <typename E, typename R, typename A>
template <typename object_intersection_t>
void MultiEigenStepperLoop<E, R, A>::updateStepSize(
    State& state, const object_intersection_t& oIntersection,
    bool release) const {
  const Surface& surface = *oIntersection.representation;

  for (auto& component : state.components) {
    auto intersection = surface.intersect(
        component.state.geoContext, SingleStepper::position(component.state),
        SingleStepper::direction(component.state), true);

    // We don't know whatever was done to manipulate the intersection before
    // (e.g. in Layer.ipp:240), so we trust and just adjust the sign
    if (std::signbit(oIntersection.intersection.pathLength) !=
        std::signbit(intersection.intersection.pathLength)) {
      intersection.intersection.pathLength *= -1;
    }

    if (std::signbit(oIntersection.alternative.pathLength) !=
        std::signbit(intersection.alternative.pathLength)) {
      intersection.alternative.pathLength *= -1;
    }

    SingleStepper::updateStepSize(component.state, intersection, release);
  }
}

template <typename E, typename R, typename A>
template <typename propagator_state_t>
Result<double> MultiEigenStepperLoop<E, R, A>::step(
    propagator_state_t& state) const {
  using Status = Acts::Intersection3D::Status;

  State& stepping = state.stepping;
  auto& components = stepping.components;

  // @TODO: This needs to be a real logger
  const Logger& logger = getDummyLogger();

  // Update step count
  stepping.steps++;

  // Check if we abort because of m_stepLimitAfterFirstComponentOnSurface
  if (stepping.stepCounterAfterFirstComponentOnSurface) {
    (*stepping.stepCounterAfterFirstComponentOnSurface)++;

    // If the limit is reached, remove all components which are not on a
    // surface, reweight the components, perform no step and return 0
    if (*stepping.stepCounterAfterFirstComponentOnSurface >=
        m_stepLimitAfterFirstComponentOnSurface) {
      for (auto& cmp : components) {
        if (cmp.status != Status::onSurface) {
          cmp.status = Status::missed;
        }
      }

      removeMissedComponents(stepping);
      reweightComponents(stepping);

      ACTS_VERBOSE("Stepper performed "
                   << m_stepLimitAfterFirstComponentOnSurface
                   << " after the first component hit a surface.");
      ACTS_VERBOSE(
          "-> remove all components not on a surface, perform no step");

      stepping.stepCounterAfterFirstComponentOnSurface.reset();

      return 0.0;
    }
  }

  // Flag indicating if we need to reweight in the end
  bool reweightNecessary = false;

  // If at least one component is on a surface, we can remove all missed
  // components before the step. If not, we must keep them for the case that all
  // components miss and we need to retarget
  const auto cmpsOnSurface =
      std::count_if(components.cbegin(), components.cend(), [&](auto& cmp) {
        return cmp.status == Intersection3D::Status::onSurface;
      });

  if (cmpsOnSurface > 0) {
    removeMissedComponents(stepping);
    reweightNecessary = true;
  }

  // Loop over all components and collect results in vector, write some
  // summary information to a stringstream
  SmallVector<std::optional<Result<double>>> results;
  double accumulatedPathLength = 0.0;
  std::size_t errorSteps = 0;

  // Type of the proxy single propagation state
  using ThisSinglePropState =
      SinglePropState<SingleState, decltype(state.navigation),
                      decltype(state.options), decltype(state.geoContext)>;

  // Lambda that performs the step for a component and returns false if the step
  // went ok and true if there was an error
  auto componentStep = [&](auto& component) {
    if (component.status == Status::onSurface) {
      // We need to add these, so the propagation does not fail if we have only
      // components on surfaces and failing states
      results.emplace_back(std::nullopt);
      return false;
    }

    ThisSinglePropState single_state(component.state, state.navigation,
                                     state.options, state.geoContext);

    results.emplace_back(SingleStepper::step(single_state));

    if (results.back()->ok()) {
      accumulatedPathLength += component.weight * results.back()->value();
      return false;
    } else {
      ++errorSteps;
      reweightNecessary = true;
      return true;
    }
  };

  // Loop over components and remove errorous components
  stepping.components.erase(
      std::remove_if(components.begin(), components.end(), componentStep),
      components.end());

  // Reweight if necessary
  if (reweightNecessary) {
    reweightComponents(stepping);
  }

  // Print the result vector to a string so we can log it
  auto summary = [](auto& result_vec) {
    std::stringstream ss;
    for (auto& optRes : result_vec) {
      if (not optRes) {
        ss << "on surface | ";
      } else if (optRes->ok()) {
        ss << optRes->value() << " | ";
      } else {
        ss << optRes->error() << " | ";
      }
    }
    auto str = ss.str();
    str.resize(str.size() - 3);
    return str;
  };

  // Print the summary
  if (errorSteps == 0) {
    ACTS_VERBOSE("Performed steps: " << summary(results));
  } else {
    ACTS_WARNING("Performed steps with errors: " << summary(results));
  }

  // Return error if there is no ok result
  if (stepping.components.empty()) {
    return MultiStepperError::AllComponentsSteppingError;
  }

  // Return the weighted accumulated path length of all successful steps
  stepping.pathAccumulated += accumulatedPathLength;
  return accumulatedPathLength;
}
}  // namespace Acts
