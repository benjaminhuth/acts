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

template <typename component_t, typename projector_t>
bool componentWeightsAreNormalized(const std::vector<component_t> &cmps,
                                   const projector_t &proj,
                                   double tol = 1.e-8) {
  double sum_of_weights = 0.0;

  for (const auto &cmp : cmps) {
    sum_of_weights += proj(cmp);
  }

  if (std::abs(sum_of_weights - 1.0) < tol) {
    return true;
  } else {
    std::cout << "diff: " << std::setprecision(10)
              << std::abs(sum_of_weights - 1.0) << "\n";
    std::cout << "weights = ";
    for (const auto &cmp : cmps) {
      std::cout << proj(cmp) << " ";
    }
    std::cout << "]\n";
    return false;
  }
}

template <typename component_t, typename projector_t>
void normalizeWeights(std::vector<component_t> &cmps, const projector_t &proj) {
  double sum_of_weights = 0.0;

  for (auto &cmp : cmps) {
    sum_of_weights += proj(cmp);
  }

  for (auto &cmp : cmps) {
    proj(cmp) /= sum_of_weights;
  }
}

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
  const double weightCutoff;

  ComponentSplitter(const bethe_heitler_t &bh, double cutoff)
      : betheHeitler(bh), weightCutoff(cutoff) {}

  template <typename propagator_state_t, typename component_cache_t>
  void operator()(const propagator_state_t &state,
                  const BoundTrackParameters &old_bound,
                  const double old_weight,
                  const GsfComponentMetaCache &metaCache,
                  std::vector<component_cache_t> &componentCaches) const {
    const auto &logger = state.options.logger;
    const auto p_prev = old_bound.absoluteMomentum();
    const auto slab =
        state.navigation.currentSurface->surfaceMaterial()->materialSlab(
            old_bound.position(state.stepping.geoContext),
            state.stepping.navDir, MaterialUpdateStage::fullUpdate);

    const auto mixture = betheHeitler.mixture(slab.thicknessInX0());

    // Create all possible new components
    for (const auto &gaussian : mixture) {
      // Here we combine the new child weight with the parent weight.
      // However, this must be later re-adjusted
      const auto new_weight = gaussian.weight * old_weight;

      if (new_weight < weightCutoff) {
        ACTS_VERBOSE("Skip component with weight " << new_weight);
        continue;
      }

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
/// @note components with weight 0 are ignored and not added to the stepper.
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

    if (weight == 0.0) {
      continue;
    }

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
template <typename propagator_state_t, typename stepper_t, typename component_t,
          typename component_processor_t>
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

  constexpr int prec = std::numeric_limits<long double>::digits10 + 1;

  throw_assert(
      std::abs(checkQOverPSum - initialQOverP) < 1.e-4,
      "momentum mismatch, initial: "
          << std::setprecision(prec) << initialQOverP
          << ", final: " << checkQOverPSum << ", component summary:\n"
          << [&]() {
               std::stringstream ss;
               for (auto i = 0ul; i < stepper.numberComponents(stepping); ++i) {
                 typename stepper_t::ComponentProxy cmp(stepping, i);
                 ss << "  #" << i << ": qop = " << std::setprecision(prec)
                    << cmp.pars()[eFreeQOverP] << "\t(" << cmp.status()
                    << ")\n";
               }
               return ss.str();
             }());

  throw_assert(
      std::abs(checkWeightSum - 1.0) < 1.e-4,
      "must sum up to 1 but is "
          << checkWeightSum
          << ", difference: " << std::abs(checkWeightSum - 1.0)
          << ", component summary: " << [&]() {
               std::stringstream ss;
               for (auto i = 0ul; i < stepper.numberComponents(stepping); ++i) {
                 typename stepper_t::ComponentProxy cmp(stepping, i);
                 ss << "  #" << i << ": weight = " << cmp.weight() << "\t("
                    << cmp.status() << ")\n";
               }
               return ss.str();
             }());

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
  throw_assert(std::distance(begin, end) > 0, "empty component range");

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
void reweightComponents(std::vector<component_t> &cmps, const projector_t &proj,
                        const double weightCutoff) {
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

    if (std::isnan(chi2) || std::isnan(detR)) {
      sumOfWeights += std::get<1>(proj(cmp));
    } else {
      const auto newWeight = std::sqrt(1. / detR) * std::exp(-0.5 * chi2);

      if (newWeight < weightCutoff) {
        std::get<1>(proj(cmp)) = 0;
      } else {
        std::get<1>(proj(cmp)) *= newWeight;
        sumOfWeights += std::get<1>(proj(cmp));
      }
    }
  }

  throw_assert(
      sumOfWeights > 0.,
      "The sum of the weights needs to be positive, but is " << sumOfWeights);

  for (auto &cmp : cmps) {
    std::get<1>(proj(cmp)) *= (1. / sumOfWeights);
  }
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
