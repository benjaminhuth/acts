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
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Utilities/Logger.hpp"

#include <numeric>

namespace Acts {

namespace detail {

/// Stores meta information about the components
struct GsfComponentMetaCache {
  /// Where to find the parent component in the MultiTrajectory
  std::size_t parentIndex;

  /// Other quantities TODO are they really needed here? seems they are
  /// reinitialized to Identity etc.
  BoundMatrix jacobian;
  BoundToFreeMatrix jacToGlobal;
  FreeMatrix jacTransport;
  FreeVector derivative;

  /// We need to preserve the path length
  ActsScalar pathLength;
};

/// Stores parameters of a gaussian component
struct GsfComponentParameterCache {
  ActsScalar weight;
  BoundVector boundPars;
  std::optional<BoundSymMatrix> boundCov;
};

/// @brief A multi component state
using MultiComponentState =
    std::pair<std::shared_ptr<const Surface>,
              std::vector<std::tuple<ActsScalar, BoundVector,
                                     std::optional<BoundSymMatrix>>>>;

/// Functor which splits a component into multiple new components by creating a
/// gaussian mixture, and adds them to a cache with a customizable cache_maker
template <typename bethe_heitler_t>
struct ComponentSplitter {
  const bethe_heitler_t &betheHeitler;

  ComponentSplitter(const bethe_heitler_t &bh) : betheHeitler(bh) {}

  template <typename propagator_state_t, typename component_cache_t>
  void operator()(const propagator_state_t &state,
                  const BoundTrackParameters &old_bound,
                  const double old_weight,
                  const GsfComponentMetaCache &metaCache,
                  std::vector<component_cache_t> &componentCaches) const {
    const auto p_prev = old_bound.absoluteMomentum();
    const auto slab =
        state.navigation.currentSurface->surfaceMaterial()->materialSlab(
            old_bound.position(state.stepping.geoContext),
            state.stepping.navDir, MaterialUpdateStage::fullUpdate);

    const auto mixture = betheHeitler.mixture(slab.thicknessInX0());

    // Create all possible new components
    for (const auto &gaussian : mixture) {
      // compute delta p from mixture and update parameters
      auto new_pars = old_bound.parameters();

      const auto delta_p = [&]() {
        if (state.stepping.navDir == NavigationDirection::forward)
          return p_prev * (gaussian.mean - 1.);
        else
          return p_prev * (1. / gaussian.mean - 1.);
      }();

      throw_assert(p_prev + delta_p > 0.,
                   "new momentum after bethe-heitler must be > 0");
      new_pars[eBoundQOverP] = old_bound.charge() / (p_prev + delta_p);

      // compute inverse variance of p from mixture and update covariance
      auto new_cov = std::move(old_bound.covariance());

      if (new_cov.has_value()) {
        const auto varInvP = [&]() {
          if (state.stepping.navDir == NavigationDirection::forward) {
            const auto f = 1. / (p_prev * gaussian.mean);
            return f * f * gaussian.var;
          } else {
            return gaussian.var / (p_prev * p_prev);
          }
        }();

        (*new_cov)(eBoundQOverP, eBoundQOverP) += varInvP;
      }
      // Here we combine the new child weight with the parent weight.
      // However, this must be later re-adjusted
      const auto new_weight = gaussian.weight * old_weight;

      // Set the remaining things and push to vector
      componentCaches.push_back(
          {GsfComponentParameterCache{new_weight, new_pars, new_cov},
           metaCache});
    }
  }
};

/// Functor whicht forwards components to the component cache and does not
/// splitting therefore
struct ComponentForwarder {
  template <typename propagator_state_t, typename component_cache_t,
            typename meta_cache_t>
  void operator()(const propagator_state_t &,
                  const BoundTrackParameters &old_bound,
                  const double old_weight, const meta_cache_t &metaCache,
                  std::vector<component_cache_t> &componentCaches) const {
    componentCaches.push_back(
        {GsfComponentParameterCache{old_weight, old_bound.parameters(),
                                    old_bound.covariance()},
         metaCache});
  }
};

/// Function that updates the stepper with the component Cache
template <typename propagator_state_t, typename stepper_t, typename component_t,
          typename projector_t>
Result<void> updateStepper(propagator_state_t &state, const stepper_t &stepper,
                           const std::vector<component_t> &componentCache,
                           const projector_t &proj) {
  //   const auto &logger = state.options.logger;
  const auto &surface = *state.navigation.currentSurface;
  stepper.clearComponents(state.stepping);

  for (const auto &[variant, meta] : componentCache) {
    const auto &[weight, pars, cov] = proj(variant);

    auto res = stepper.addComponent(
        state.stepping, BoundTrackParameters(surface.getSharedPtr(), pars, cov),
        weight);

    if (!res.ok()) {
      return res.error();
    }

    auto &cmp = *res;
    cmp.jacobian() = meta.jacobian;
    cmp.jacToGlobal() = meta.jacToGlobal;
    cmp.pathLength() = meta.pathLength;
    cmp.derivative() = meta.derivative;
    cmp.jacTransport() = meta.jacTransport;
  }

  return Result<void>::success();
}

/// @brief Expands all existing components to new components by using a
/// gaussian-mixture approximation for the Bethe-Heitler distribution.
///
/// @return a std::vector with all new components (parent tip, weight,
/// parameters, covariance)
/// TODO We could make propagator_state const here if the component proxy
/// of the stepper would accept it
template <typename propagator_state_t, typename stepper_t,
          typename component_t, typename component_processor_t>
void extractComponents(propagator_state_t &state, const stepper_t &stepper,
                       const std::vector<std::size_t> &parentTrajectoryIdxs,
                       const component_processor_t &componentProcessor,
                       const bool doCovTransport,
                       std::vector<component_t> &componentCache) {
  // Some shortcuts
  auto &stepping = state.stepping;
  const auto &logger = state.options.logger;
  const auto &surface = *state.navigation.currentSurface;

  // Adjust qop to account for lost energy due to lost components
  // TODO do we need to adjust variance?
  double sumW_loss = 0.0, sumWeightedQOverP_loss = 0.0;
  double initialQOverP = 0.0;

  for (auto i = 0ul; i < stepper.numberComponents(state.stepping); ++i) {
    typename stepper_t::ComponentProxy cmp(state.stepping, i);

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
      auto &weight = cmp.weight();
      auto &qop = cmp.pars()[eFreeQOverP];

      weight /= (1.0 - sumW_loss);
      qop = qop * (1.0 - sumW_loss) + sumWeightedQOverP_loss;

      checkWeightSum += weight;
      checkQOverPSum += weight * qop;
    }
  }

  throw_assert(std::abs(checkQOverPSum - initialQOverP) < 1.e-8,
               "momentum mismatch, initial: " << initialQOverP
                                              << ", final: " << checkQOverPSum);
  throw_assert(std::abs(checkWeightSum - 1.0) < 1.e-8,
               "must sum up to 1 but is " << checkWeightSum);

  // Approximate bethe-heitler distribution as gaussian mixture
  for (auto i = 0ul; i < stepper.numberComponents(state.stepping); ++i) {
    typename stepper_t::ComponentProxy old_cmp(state.stepping, i);

    if (old_cmp.status() != Intersection3D::Status::onSurface) {
      ACTS_VERBOSE("Skip component which is not on surface");
      continue;
    }

    auto boundState = old_cmp.boundState(surface, doCovTransport);

    if (!boundState.ok()) {
      ACTS_ERROR("Failed to compute boundState: " << boundState.error());
      continue;
    }

    const auto &[old_bound, jac, pathLength] = boundState.value();

    detail::GsfComponentMetaCache metaCache{
        parentTrajectoryIdxs[i], jac,
        old_cmp.jacToGlobal(),   old_cmp.jacTransport(),
        old_cmp.derivative(),    pathLength};

    componentProcessor(state, old_bound, old_cmp.weight(), metaCache,
                       componentCache);
  }
}

/// @brief reweight MultiComponentState
inline void normalizeMultiComponentState(MultiComponentState &state) {
  ActsScalar sum{0.0};

  for (const auto &[weight, pars, cov] : state.second) {
    sum += weight;
  }

  for (auto &[weight, pars, cov] : state.second) {
    weight /= sum;
  }
}

struct Identity {
  template <typename T>
  auto operator()(T &&v) const {
    return std::forward<T>(v);
  }
};

/// @brief Combine multiple components into one representative track state
/// object. The function takes iterators to allow for arbitrary ranges to be
/// combined
/// TODO replace Identity with std::identity on C++20
/// @tparam component_iterator_t An iterator of a range of components
/// @tparam projector_t A projector, which maps the component to a std::tuple< ActsScalar, BoundVector, std::optional< BoundSymMatrix > >
template <typename component_iterator_t, typename projector_t = Identity>
auto combineComponentRange(const component_iterator_t begin,
                           const component_iterator_t end,
                           projector_t &&projector = projector_t{},
                           bool checkIfNormalized = false) {
  using ret_type = std::tuple<BoundVector, std::optional<BoundSymMatrix>>;
  BoundVector mean = BoundVector::Zero();
  BoundSymMatrix cov1 = BoundSymMatrix::Zero();
  BoundSymMatrix cov2 = BoundSymMatrix::Zero();
  double sumOfWeights{0.0};

  const auto &[begin_weight, begin_pars, begin_cov] = projector(*begin);

  if (std::distance(begin, end) == 1) {
    return ret_type{begin_pars, *begin_cov};
  }

  const double referencePhi = begin_pars[eBoundPhi];

  // clang-format off
  // x = \sum_{l} w_l * x_l
  // C = \sum_{l} w_l * C_l + \sum_{l} \sum_{m>l} w_l * w_m * (x_l - x_m)(x_l - x_m)^T
  // clang-format on
  for (auto l = begin; l != end; ++l) {
    const auto &[weight_l, pars_l, cov_l] = projector(*l);
    throw_assert(cov_l, "we require a covariance here");

    sumOfWeights += weight_l;
    mean += weight_l * pars_l;
    cov1 += weight_l * *cov_l;

    // Avoid problems with cyclic phi
    const double deltaPhi = referencePhi - pars_l[eBoundPhi];

    if (deltaPhi > M_PI) {
      mean[eBoundPhi] += (2 * M_PI) * weight_l;
    } else if (deltaPhi < -M_PI) {
      mean[eBoundPhi] -= (2 * M_PI) * weight_l;
    }

    for (auto m = std::next(l); m != end; ++m) {
      const auto &[weight_m, pars_m, cov_m] = projector(*m);
      throw_assert(cov_m, "we require a covariance here");

      const BoundVector diff = pars_l - pars_m;
      cov2 += weight_l * weight_m * diff * diff.transpose();
    }
  }

  if (checkIfNormalized) {
    throw_assert(std::abs(sumOfWeights - 1.0) < 1.e-8,
                 "weights are not normalized");
  }

  return ret_type{mean / sumOfWeights,
                  cov1 / sumOfWeights + cov2 / (sumOfWeights * sumOfWeights)};
}

/// @brief Function that reduces the number of components. at the moment,
/// this is just by erasing the components with the lowest weights.
/// Finally, the components are reweighted so the sum of the weights is
/// still 1
/// TODO If we create new components here to preserve the mean or if we do some
/// component merging, how can this be applied in the MultiTrajectory
template <typename component_t>
void reduceNumberOfComponents(std::vector<component_t> &components,
                              std::size_t maxRemainingComponents) {
  // The elements should be sorted by weight (high -> low)
  std::sort(begin(components), end(components),
            [](const auto &a, const auto &b) { return a.weight > b.weight; });

  // Remove elements by resize
  components.erase(begin(components) + maxRemainingComponents, end(components));

  // Reweight after removal
  const auto sum_of_weights = std::accumulate(
      begin(components), end(components), 0.0,
      [](auto sum, const auto &cmp) { return sum + cmp.weight; });

  for (auto &cmp : components) {
    cmp.weight /= sum_of_weights;
  }
}

/// @brief Reweight the components according to `R. Frühwirth, "Track fitting
/// with non-Gaussian noise"`. See also the implementation in Athena at
/// PosteriorWeightsCalculator.cxx
/// Expects that the projector maps the component to something like a
/// std::pair< trackProxy&, double& > so that it can be extracted with std::get
template <typename component_t, typename projector_t>
void reweightComponents(std::vector<component_t> &cmps,
                        const projector_t &proj) {
  // Helper Function to compute detR
  auto computeDetR = [](const auto &trackState) -> ActsScalar {
    const auto predictedCovariance = trackState.predictedCovariance();

    return visit_measurement(
        trackState.calibrated(), trackState.calibratedCovariance(),
        trackState.calibratedSize(),
        [&](const auto calibrated, const auto calibratedCovariance) {
          constexpr size_t kMeasurementSize =
              decltype(calibrated)::RowsAtCompileTime;
          const auto H =
              trackState.projector()
                  .template topLeftCorner<kMeasurementSize, eBoundSize>()
                  .eval();

          return (H * predictedCovariance * H.transpose() +
                  calibratedCovariance)
              .determinant();
        });
  };

  // Find minChi2, this can be used to factor some things later in the
  // exponentiation
  const auto minChi2 =
      std::get<0>(proj(*std::min_element(cmps.begin(), cmps.end(),
                                         [&](const auto &a, const auto &b) {
                                           return std::get<0>(proj(a)).chi2() <
                                                  std::get<0>(proj(b)).chi2();
                                         })))
          .chi2();

  // Compute new weights and reweight
  double sumOfWeights = 0.;

  for (auto &cmp : cmps) {
    const double chi2 = std::get<0>(proj(cmp)).chi2() - minChi2;
    const double detR = computeDetR(std::get<0>(proj(cmp)));

    std::get<1>(proj(cmp)) *= std::sqrt(1. / detR) * std::exp(-0.5 * chi2);
    sumOfWeights += std::get<1>(proj(cmp));
  }

  throw_assert(sumOfWeights > 0.,
               "The sum of the weights needs to be positive");

  for (auto &cmp : cmps) {
    std::get<1>(proj(cmp)) *= (1. / sumOfWeights);
  }
}

/// @brief Combine a forward pass and a backward pass to a smoothed trajectory.
/// This is part of the Weighted-Mean-Smoother implementation for the GSF
inline std::vector<BoundTrackParameters> combineForwardAndBackwardPass(
    const std::vector<BoundTrackParameters> &forward,
    const std::vector<BoundTrackParameters> &backward) {
  throw_assert(forward.size() == backward.size(),
               "forward and backwards pass must match in size");

  std::vector<BoundTrackParameters> ret;
  ret.reserve(forward.size());

  for (auto i = 0ul; i < forward.size(); ++i) {
    throw_assert(forward[i].referenceSurface().geometryId() ==
                     backward[i].referenceSurface().geometryId(),
                 "ID must be equal");

    //     const BoundSymMatrix covFwdInv = forward[i].covariance()->inverse();
    //     const BoundSymMatrix covBwdInv = backward[i].covariance()->inverse();
    //
    //     const BoundSymMatrix covInv = covFwdInv + covBwdInv;
    //
    //     const BoundVector params = covInv * (covFwdInv *
    //     forward[i].parameters() +
    //                                          covBwdInv *
    //                                          backward[i].parameters());
    //         ret.push_back(
    //         BoundTrackParameters(forward[i].referenceSurface().getSharedPtr(),
    //                              params, covInv.inverse()));

    // Where do these equations come from???
    const BoundSymMatrix covSummed =
        *forward[i].covariance() + *backward[i].covariance();
    const BoundSymMatrix K = *forward[i].covariance() * covSummed.inverse();
    const BoundSymMatrix newCov = K * *backward[i].covariance();

    const BoundVector xNew =
        forward[i].parameters() +
        K * (backward[i].parameters() - forward[i].parameters());

    ret.push_back(BoundTrackParameters(
        forward[i].referenceSurface().getSharedPtr(), xNew, newCov));
  }

  return ret;
}

/// Enumeration type used in extractMultiComponentStates(...)
enum class StatesType { ePredicted, eFiltered, eSmoothed };

inline std::ostream &operator<<(std::ostream &os, StatesType type) {
  constexpr static std::array names = {"predicted", "filtered", "smoothed"};
  os << names[static_cast<int>(type)];
  return os;
}

/// @brief Extracts a MultiComponentState from a MultiTrajectory and a given list of indices
template <typename source_link_t>
auto extractMultiComponentState(const MultiTrajectory<source_link_t> &traj,
                                const std::vector<size_t> &tips,
                                const std::map<size_t, ActsScalar> &weights,
                                StatesType type) {
  throw_assert(
      !tips.empty(),
      "need at least one component to extract trajectory of type " << type);

  std::vector<std::tuple<double, BoundVector, BoundSymMatrix>> cmps;
  std::shared_ptr<const Surface> surface;

  for (auto &tip : tips) {
    const auto proxy = traj.getTrackState(tip);

    throw_assert(weights.find(tip) != weights.end(),
                 "Could not find weight for idx " << tip);

    switch (type) {
      case StatesType::ePredicted:
        cmps.push_back(
            {weights.at(tip), proxy.predicted(), proxy.predictedCovariance()});
        break;
      case StatesType::eFiltered:
        cmps.push_back(
            {weights.at(tip), proxy.filtered(), proxy.filteredCovariance()});
        break;
      case StatesType::eSmoothed:
        cmps.push_back(
            {weights.at(tip), proxy.smoothed(), proxy.smoothedCovariance()});
    }

    if (!surface) {
      surface = proxy.referenceSurface().getSharedPtr();
    } else {
      throw_assert(
          surface->geometryId() == proxy.referenceSurface().geometryId(),
          "surface mismatch");
    }
  }

  return MultiComponentBoundTrackParameters<SinglyCharged>(surface, cmps);
}

/// @brief This function applies the bayesian smoothing by combining a
/// forward MultiComponentState and a backward MultiComponentState into a new
/// MultiComponentState. The result is not normalized, and also not component
/// reduction is done
inline auto bayesianSmoothing(const MultiComponentState &fwd,
                       const MultiComponentState &bwd) {
  MultiComponentState smoothedState;
  std::get<1>(smoothedState)
      .reserve(std::get<1>(fwd).size() + std::get<1>(bwd).size());

  throw_assert(std::get<0>(fwd)->geometryId() == std::get<0>(bwd)->geometryId(),
               "surface mismatch");
  std::get<0>(smoothedState) = std::get<0>(fwd);

  for (const auto &[weight_a, pars_a, cov_a] : std::get<1>(fwd)) {
    throw_assert(cov_a, "for now we require a covariance here");

    for (const auto &[weight_b, pars_b, cov_b] : std::get<1>(bwd)) {
      throw_assert(cov_b, "for now we require a covariance here");

      const auto summedCov = *cov_a + *cov_b;
      const auto K = *cov_a * summedCov.inverse();
      const auto new_pars = (pars_a + K * (pars_b - pars_a)).eval();
      const auto new_cov = (K * *cov_b).eval();

      const auto diff = pars_a - pars_b;
      const ActsScalar exponent = diff.transpose() * summedCov.inverse() * diff;

      const auto new_weight = std::exp(-0.5 * exponent) * weight_a * weight_b;

      std::get<1>(smoothedState).push_back({new_weight, new_pars, new_cov});
    }
  }

  return smoothedState;
}

/// @brief Smoothing function, which takes two ranges of
/// MultiTrajectory-indices and the corresponding projectors.
template <typename component_iterator_t, typename fwd_projector_t = Identity,
          typename bwd_projector_t = Identity>
auto bayesianSmoothing(component_iterator_t fwdBegin,
                       component_iterator_t fwdEnd,
                       component_iterator_t bwdBegin,
                       component_iterator_t bwdEnd,
                       fwd_projector_t fwdProjector = fwd_projector_t{},
                       bwd_projector_t bwdProjector = bwd_projector_t{}) {
  std::vector<std::tuple<double, BoundVector, std::optional<BoundSymMatrix>>>
      smoothedState;

  for (auto fwd = fwdBegin; fwd != fwdEnd; ++fwd) {
    const auto &[weight_a, pars_a, cov_a] = fwdProjector(*fwd);
    throw_assert(cov_a, "for now we require a covariance here");

    for (auto bwd = bwdBegin; bwd != bwdEnd; ++bwd) {
      const auto &[weight_b, pars_b, cov_b] = bwdProjector(*bwd);
      throw_assert(cov_b, "for now we require a covariance here");

      const auto summedCov = *cov_a + *cov_b;
      const auto K = *cov_a * summedCov.inverse();
      const auto new_pars = (pars_a + K * (pars_b - pars_a)).eval();
      const auto new_cov = (K * *cov_b).eval();

      const auto diff = pars_a - pars_b;
      const ActsScalar exponent = diff.transpose() * summedCov.inverse() * diff;

      const auto new_weight = std::exp(-0.5 * exponent) * weight_a * weight_b;

      smoothedState.push_back({new_weight, new_pars, new_cov});
    }
  }

  return smoothedState;
}

/// @brief Projector type which maps a MultiTrajectory-Index to a tuple of
/// [weight, parameters, covariance]. Therefore, it contains a MultiTrajectory
/// and for now a std::map for the weights
template <StatesType type, typename source_link_t>
struct MultiTrajectoryProjector {
  const MultiTrajectory<source_link_t> &mt;
  const std::map<std::size_t, double> &weights;

  auto operator()(std::size_t idx) const {
    const auto proxy = mt.getTrackState(idx);
    switch (type) {
      case StatesType::ePredicted:
        return std::make_tuple(weights.at(idx), proxy.predicted(),
                               std::optional{proxy.predictedCovariance()});
      case StatesType::eFiltered:
        return std::make_tuple(weights.at(idx), proxy.filtered(),
                               std::optional{proxy.filteredCovariance()});
      case StatesType::eSmoothed:
        return std::make_tuple(weights.at(idx), proxy.smoothed(),
                               std::optional{proxy.smoothedCovariance()});
    }
  }
};

/// @brief This function takes two MultiTrajectory objects and corresponding
/// index lists (one of the backward pass, one of the forward pass), combines
/// them, applies smoothing, and returns a new, single-component MultiTrajectory
template <typename source_link_t>
auto smoothAndCombineTrajectories(
    const MultiTrajectory<source_link_t> &fwd,
    const std::vector<std::size_t> &fwdStartTips,
    const std::map<std::size_t, double> &fwdWeights,
    const MultiTrajectory<source_link_t> bwd,
    const std::vector<std::size_t> &bwdStartTips,
    const std::map<std::size_t, double> &bwdWeights) {
  // Use backward trajectory as basic trajectory, so that final trajectory is
  // ordered correctly. We ensure also that they are unique.
  std::vector<std::size_t> bwdTips = bwdStartTips;
  std::sort(bwdTips.begin(), bwdTips.end());
  bwdTips.erase(std::unique(bwdTips.begin(), bwdTips.end()), bwdTips.end());

  MultiTrajectory<source_link_t> finalTrajectory;

  std::size_t lastTip = SIZE_MAX;

  // MultiTrajectory uses uint16_t internally TODO is none_of here correct?
  while (std::none_of(bwdTips.begin(), bwdTips.end(), [](auto i) {
    return i == std::numeric_limits<uint16_t>::max();
  })) {
    const auto firstBwdState = bwd.getTrackState(bwdTips.front());

    lastTip = finalTrajectory.addTrackState(TrackStatePropMask::All, lastTip);
    auto proxy = finalTrajectory.getTrackState(lastTip);

    // This way I hope we copy all relevant flags and the calibrated field
    proxy.copyFrom(firstBwdState);

    // Search corresponding forward tips
    std::vector<std::size_t> fwdTips;

    for (const auto tip : fwdStartTips) {
      fwd.visitBackwards(tip, [&](const auto &state) {
        if (state.referenceSurface().geometryId() ==
            firstBwdState.referenceSurface().geometryId()) {
          fwdTips.push_back(state.index());
        }
      });
    }

    std::sort(fwdTips.begin(), fwdTips.end());
    fwdTips.erase(std::unique(fwdTips.begin(), fwdTips.end()), fwdTips.end());

    // Evaluate the predicted, filtered and smoothed state
    using PredProjector =
        MultiTrajectoryProjector<StatesType::ePredicted, source_link_t>;
    using FiltProjector =
        MultiTrajectoryProjector<StatesType::eFiltered, source_link_t>;

    const auto [fwdMeanPred, fwdCovPred] = combineComponentRange(
        fwdTips.begin(), fwdTips.end(), PredProjector{fwd, fwdWeights});
    proxy.predicted() = fwdMeanPred;
    proxy.predictedCovariance() = fwdCovPred.value();

    const auto [bwdMeanFilt, bwdCovFilt] = combineComponentRange(
        bwdTips.begin(), bwdTips.end(), FiltProjector{bwd, bwdWeights});
    proxy.filtered() = bwdMeanFilt;
    proxy.filteredCovariance() = bwdCovFilt.value();

    const auto smoothedState = bayesianSmoothing(
        fwdTips.begin(), fwdTips.end(), bwdTips.begin(), bwdTips.end(),
        PredProjector{fwd, fwdWeights}, FiltProjector{bwd, bwdWeights});
    const auto [smoothedMean, smoothedCov] =
        combineComponentRange(smoothedState.begin(), smoothedState.end());
    proxy.smoothed() = smoothedMean;
    proxy.smoothedCovariance() = smoothedCov.value();

    throw_assert(proxy.typeFlags().test(Acts::TrackStateFlag::MeasurementFlag),
                 "must be a measurment");

    // Update bwdTips to the next state
    for (auto &tip : bwdTips) {
      const auto p = bwd.getTrackState(tip);
      tip = p.previous();
    }

    std::sort(bwdTips.begin(), bwdTips.end());
    bwdTips.erase(std::unique(bwdTips.begin(), bwdTips.end()), bwdTips.end());
  }

  return std::make_tuple(finalTrajectory, lastTip);
}

template <typename source_link_t>
auto multiTrajectoryToMultiComponentParameters(
    const std::vector<std::size_t> &tips,
    const MultiTrajectory<source_link_t> &mt,
    const std::map<std::size_t, double> weights, StatesType type) {
  std::shared_ptr<const Surface> surface =
      mt.getTrackState(tips.front()).referenceSurface().getSharedPtr();

  using Tuple = std::tuple<double, BoundVector, std::optional<BoundSymMatrix>>;
  std::vector<Tuple> comps;

  for (const auto tip : tips) {
    const auto &state = mt.getTrackState(tip);

    throw_assert(state.referenceSurface().geometryId() == surface->geometryId(),
                 "surface mismatch");

    switch (type) {
      case StatesType::ePredicted: {
        comps.push_back(Tuple{weights.at(tip), state.predicted(),
                              state.predictedCovariance()});
      } break;

      case StatesType::eFiltered: {
        comps.push_back(Tuple{weights.at(tip), state.filtered(),
                              state.filteredCovariance()});
      } break;

      case StatesType::eSmoothed: {
        comps.push_back(Tuple{weights.at(tip), state.smoothed(),
                              state.smoothedCovariance()});
      }
    }
  }

  return MultiComponentBoundTrackParameters<SinglyCharged>(surface, comps);
}

}  // namespace detail

}  // namespace Acts
