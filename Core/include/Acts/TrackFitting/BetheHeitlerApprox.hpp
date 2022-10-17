// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/TrackFitting/detail/GsfUtils.hpp"

#include <array>
#include <fstream>
#include <random>

#include <boost/math/quadrature/trapezoidal.hpp>

namespace Acts {

inline ActsScalar logistic_sigmoid(ActsScalar x) {
  return 1. / (1 + std::exp(-x));
}

struct GaussianComponent {
  ActsScalar weight, mean, var;
};

/// This class approximates the Bethe-Heitler distribution as a gaussian
/// mixture. To enable an approximation for continous input variables, the
/// weights, means and variances are internally parametrized as a Nth order
/// polynomial.
template <int NComponents, int PolyDegree>
class AtlasBetheHeitlerApprox {
  static_assert(NComponents > 0);
  static_assert(PolyDegree > 0);

 public:
  struct PolyData {
    std::array<ActsScalar, PolyDegree + 1> weightCoeffs, meanCoeffs, varCoeffs;
  };

  using Data = std::array<PolyData, NComponents>;

  constexpr static double noChangeLimit = 0.0001;
  constexpr static double singleGaussianLimit = 0.002;
  constexpr static double lowerLimit = 0.10;
  constexpr static double higherLimit = 0.20;

 private:
  Data m_low_data;
  Data m_high_data;
  bool m_low_transform;
  bool m_high_transform;

 public:
  /// Construct the Bethe-Heitler approximation description
  ///
  /// @param low_data data for the lower x/x0 range
  /// @param high_data data for the higher x/x0 range
  /// @param transform wether the data need to be transformed (see Atlas code)
  constexpr AtlasBetheHeitlerApprox(const Data &low_data, const Data &high_data,
                                    bool low_transform, bool high_transform)
      : m_low_data(low_data),
        m_high_data(high_data),
        m_low_transform(low_transform),
        m_high_transform(high_transform) {}

  /// Returns the number of components the returned mixture will have
  constexpr auto numComponents() const { return NComponents; }

  /// Checks if an input is valid for the parameterization
  ///
  /// @param x input in terms of x/x0
  constexpr bool validXOverX0(ActsScalar x) const { return x < higherLimit; }

  /// Generates the mixture from the polynomials and reweights them, so
  /// that the sum of all weights is 1
  ///
  /// @param x The input in terms of x/x0 (pathlength in terms of radiation length)
  auto mixture(ActsScalar x) const {
    // Build a polynom
    auto poly = [](ActsScalar xx,
                   const std::array<ActsScalar, PolyDegree + 1> &coeffs) {
      ActsScalar sum{0.};
      for (const auto c : coeffs) {
        sum = xx * sum + c;
      }
      throw_assert(std::isfinite(sum), "polynom result not finite");
      return sum;
    };

    // Lambda which builds the components
    auto make_mixture = [&](const Data &data, double xx, bool transform) {
      // Value initialization should garanuee that all is initialized to zero
      std::array<GaussianComponent, NComponents> ret{};
      ActsScalar weight_sum = 0;
      for (int i = 0; i < NComponents; ++i) {
        // These transformations must be applied to the data according to ATHENA
        // (TrkGaussianSumFilter/src/GsfCombinedMaterialEffects.cxx:79)
        if (transform) {
          ret[i].weight = logistic_sigmoid(poly(xx, data[i].weightCoeffs));
          ret[i].mean = logistic_sigmoid(poly(xx, data[i].meanCoeffs));
          ret[i].var = std::exp(poly(xx, data[i].varCoeffs));
        } else {
          ret[i].weight = poly(xx, data[i].weightCoeffs);
          ret[i].mean = poly(xx, data[i].meanCoeffs);
          ret[i].var = poly(xx, data[i].varCoeffs);
        }

        weight_sum += ret[i].weight;
      }

      for (int i = 0; i < NComponents; ++i) {
        ret[i].weight /= weight_sum;
      }

      return ret;
    };

    // Return no change
    if (x < noChangeLimit) {
      std::array<GaussianComponent, NComponents> ret{};

      ret[0].weight = 1.0;
      ret[0].mean = 1.0;  // p_initial = p_final
      ret[0].var = 0.0;

      return ret;
    }
    // Return single gaussian approximation
    if (x < singleGaussianLimit) {
      std::array<GaussianComponent, NComponents> ret{};

      ret[0].weight = 1.0;
      ret[0].mean = std::exp(-1. * x);
      ret[0].var =
          std::exp(-1. * x * std::log(3.) / std::log(2.)) - std::exp(-2. * x);

      return ret;
    }
    // Return a component representation for lower x0
    if (x < lowerLimit) {
      return make_mixture(m_low_data, x, m_low_transform);
    }
    // Return a component representation for higher x0
    else {
      // Cap the x because beyond the parameterization goes wild
      const auto high_x = std::min(higherLimit, x);
      return make_mixture(m_high_data, high_x, m_high_transform);
    }
  }

  /// Loads a parameterization from a file according to the Atlas file
  /// description
  ///
  /// @param low_parameters_path Path to the foo.par file that stores
  /// the parameterization for low x/x0
  /// @param high_parameters_path Path to the foo.par file that stores
  /// the parameterization for high x/x0
  static auto loadFromFile(const std::string &low_parameters_path,
                           const std::string &high_parameters_path) {
    auto read_file = [](const std::string &filepath) {
      std::ifstream file(filepath);

      if (!file) {
        throw std::invalid_argument("Could not open '" + filepath + "'");
      }

      std::size_t n_cmps, degree;
      bool transform_code;

      file >> n_cmps >> degree >> transform_code;

      if (NComponents != n_cmps) {
        throw std::invalid_argument("Wrong number of components in '" +
                                    filepath + "'");
      }

      if (PolyDegree != degree) {
        throw std::invalid_argument("Wrong wrong polynom order in '" +
                                    filepath + "'");
      }

      if (!transform_code) {
        throw std::invalid_argument("Transform-code is required in '" +
                                    filepath + "'");
      }

      Data data;

      for (auto &cmp : data) {
        for (auto &coeff : cmp.weightCoeffs) {
          file >> coeff;
        }
        for (auto &coeff : cmp.meanCoeffs) {
          file >> coeff;
        }
        for (auto &coeff : cmp.varCoeffs) {
          file >> coeff;
        }
      }

      return std::make_tuple(data, transform_code);
    };

    const auto [low_data, low_transform] = read_file(low_parameters_path);
    const auto [high_data, high_transform] = read_file(high_parameters_path);

    return AtlasBetheHeitlerApprox(low_data, high_data, low_transform,
                                   high_transform);
  }
};

/// This class approximates the Bethe-Heitler with only one component
struct BetheHeitlerApproxSingleCmp {
  /// Returns the number of components the returned mixture will have
  constexpr auto numComponents() const { return 1; }

  /// Checks if an input is valid for the parameterization. Since this is for
  /// debugging, it always returns false
  ///
  /// @param x input in terms of x/x0
  constexpr bool validXOverX0(ActsScalar) const { return false; }

  /// Returns array with length 1
  auto mixture(const ActsScalar x) const {
    std::array<GaussianComponent, 1> ret{};

    ret[0].weight = 1.0;
    ret[0].mean = std::exp(-1. * x);
    ret[0].var =
        std::exp(-1. * x * std::log(3.) / std::log(2.)) - std::exp(-2. * x);

    return ret;
  }
};

/// This class does the approximation by minimizing the CDF distance
/// individually for each point. Probably very slow, but good vor validating.
template <std::size_t NComponents>
class BetheHeitlerSimulatedAnnealingMinimizer {
  std::vector<double> m_temperatures;
  mutable std::mt19937 m_gen; // TODO how to solve this mutable?
  std::array<GaussianComponent, NComponents> m_start_value;

  /// Compute the value of the gaussian mixture at x
  class GaussianMixtureModel {
    std::array<GaussianComponent, NComponents> m_components;

   public:
    GaussianMixtureModel(
        const std::array<GaussianComponent, NComponents> &components)
        : m_components(components) {}

    double operator()(double x) {
      auto gaussian = [&](double mu, double sigma) -> double {
        return (1. / std::sqrt(sigma * sigma * 2 * M_PI)) *
               std::exp((-(x - mu) * (x - mu)) / (2 * sigma * sigma));
      };

      double sum = 0;

      for (const auto [w, m, s] : m_components) {
        sum += w * gaussian(m, s);
      }

      return sum;
    }
  };

  /// Compute the Bethe-Heitler Distribution on a value x
  class BetheHeitlerDistribution {
    double m_thickness;

   public:
    BetheHeitlerDistribution(double thicknessInX0)
        : m_thickness(thicknessInX0) {}

    double operator()(double x) {
      if (x <= 0.0 || x >= 1.0) {
        return 0.0;
      }

      auto c = m_thickness / std::log(2);
      return std::pow(-std::log(x), c - 1) / std::tgamma(c);
    }
  };

  /// Integrand for the CDF distance
  class CDFIntegrant {
    GaussianMixtureModel m_mixture;
    BetheHeitlerDistribution m_distribution;

   public:
    CDFIntegrant(const GaussianMixtureModel &mixture,
                 const BetheHeitlerDistribution &dist)
        : m_mixture(mixture), m_distribution(dist) {}

    double operator()(double x) {
      return std::abs(m_mixture(x) - m_distribution(x));
    }
  };

 public:
  BetheHeitlerSimulatedAnnealingMinimizer(
      const std::vector<double> &temperatures,
      std::mt19937 gen = std::mt19937{42})
      : m_temperatures(temperatures), m_gen(gen) {
    // Distribute means with to geometric series
    std::array<GaussianComponent, NComponents> mixture{};
    double m = 0.;
    for (auto i = 0ul; i < mixture.size(); ++i) {
      m += std::pow(0.5, i + 1);

      mixture[i].weight = 1. / (mixture.size() - i);
      mixture[i].mean = m;
      mixture[i].var = 0.05 / (i + 1);
    }

    detail::normalizeWeights(mixture,
                             [](auto &a) -> double& { return a.weight; });
    m_start_value = mixture;
  }

  /// Returns the number of components the returned mixture will have
  constexpr auto numComponents() const { return NComponents; }

  /// Checks if an input is valid for the parameterization. Since we do the fit
  /// on-the-fly, always return true (but of course the Bethe-Heitler model does
  /// not apply to arbitrary thick materials)
  ///
  /// @param x input in terms of x/x0
  constexpr bool validXOverX0(ActsScalar) const { return true; }

  /// Performes a simulated-annealing minimization of the CDF distance at the
  /// given x/x0.
  ///
  /// @param x The input in terms of x/x0 (pathlength in terms of radiation length)
  auto mixture(const ActsScalar x) const {
    // Helper function to do the integration
    auto integrate = [&](const auto &mixture) {
      return boost::math::quadrature::trapezoidal(
          CDFIntegrant{GaussianMixtureModel{mixture},
                       BetheHeitlerDistribution{x}},
          -0.5, 1.5);
    };

    // Transform to coordinates defined in [-inf, inf]
    auto transform = [](const std::array<GaussianComponent, NComponents> &cs) {
      auto ret = std::array<double, 3 * NComponents>{};

      auto it = ret.begin();
      for (const auto &[w, m, s] : cs) {
        *it = std::log(w) - std::log(1 - w);
        ++it;
        *it = std::log(m) - std::log(1 - m);
        ++it;
        *it = std::log(s);
        ++it;
      }

      return ret;
    };

    // Transform from coordinates defined in [-inf, inf]
    auto inv_transform = [](const std::array<double, 3 * NComponents> &cs) {
      auto ret = std::array<GaussianComponent, NComponents>{};

      auto it = cs.cbegin();
      for (auto &[w, m, s] : ret) {
        w = 1. / (1 + std::exp(-*it));
        ++it;
        m = 1. / (1 + std::exp(-*it));
        ++it;
        s = std::exp(*it);
        ++it;
      }

      return ret;
    };

    // How to pick a new configuration
    auto next = [&](auto ps) {
      const double range = 0.3;
      auto val_dist = std::uniform_real_distribution{-range, range};
      for (auto &p : ps) {
        p += val_dist(m_gen);
      }

      return ps;
    };

    // Initialize state
    auto current_distance = integrate(m_start_value);
    auto current_params = transform(m_start_value);

    // seperately keep track of best solution
    auto best_distance = current_distance;
    auto best_params = m_start_value;

    for (auto T : m_temperatures) {
      for (int i = 0; i < 10; ++i) {
        const auto new_params = next(current_params);
        auto new_params_transformed = inv_transform(new_params);
        detail::normalizeWeights(new_params_transformed,
                                 [](auto &a) -> double& { return a.weight; });
        const double new_distance = integrate(new_params_transformed);

        if (not std::isfinite(new_distance)) {
          continue;
        }

        const double p = std::exp(-(new_distance - current_distance) / T);

        if (new_distance < best_distance) {
          best_distance = new_distance;
          best_params = new_params_transformed;
        }

        if (new_distance < current_distance or
            p < std::uniform_real_distribution{0., 1.0}(m_gen)) {
          current_distance = new_distance;
          current_params = new_params;
        }

        break;
      }
    }

    return best_params;
  }
};

/// These data are from ATLAS and allow using the GSF without loading files.
/// However, this might not be the optimal parameterization. These data come
/// this file in Athena:
/// Tracking/TrkFitter/TrkGaussianSumFilterUtils/Data/BetheHeitler_cdf_nC6_O5.par
/// These data must be transformed, so construct the AtlasBetheHeitlerApprox
/// with transforms = true
// clang-format off
constexpr static AtlasBetheHeitlerApprox<6, 5>::Data bh_cdf_cmps6_order5_data = {{
    // Component #1
    {
        {{3.74397e+004,-1.95241e+004, 3.51047e+003,-2.54377e+002, 1.81080e+001,-3.57643e+000}},
        {{3.56728e+004,-1.78603e+004, 2.81521e+003,-8.93555e+001,-1.14015e+001, 2.55769e-001}},
        {{3.73938e+004,-1.92800e+004, 3.21580e+003,-1.46203e+002,-5.65392e+000,-2.78008e+000}}
    },
    // Component #2
    {
        {{-4.14035e+004, 2.31883e+004,-4.37145e+003, 2.44289e+002, 1.13098e+001,-3.21230e+000}},
        {{-2.06936e+003, 2.65334e+003,-1.01413e+003, 1.78338e+002,-1.85556e+001, 1.91430e+000}},
        {{-5.19068e+004, 2.55327e+004,-4.22147e+003, 1.90227e+002, 9.34602e+000,-4.80961e+000}}
    },
    // Component #3
    {
        {{2.52200e+003,-4.86348e+003, 2.11942e+003,-3.84534e+002, 2.94503e+001,-2.83310e+000}},
        {{1.80405e+003,-1.93347e+003, 6.27196e+002,-4.32429e+001,-1.43533e+001, 3.58782e+000}},
        {{-4.61617e+004, 1.78221e+004,-1.95746e+003,-8.80646e+001, 3.43153e+001,-7.57830e+000}}
    },
    // Component #4
    {
        {{4.94537e+003,-2.08737e+003, 1.78089e+002, 2.29879e+001,-5.52783e+000,-1.86800e+000}},
        {{4.60220e+003,-1.62269e+003,-1.57552e+002, 2.01796e+002,-5.01636e+001, 6.47438e+000}},
        {{-9.50373e+004, 4.05517e+004,-5.62596e+003, 4.58534e+001, 6.70479e+001,-1.22430e+001}}
    },
    // Component #5
    {
        {{-1.04129e+003, 1.15222e+002,-2.70356e+001, 3.18611e+001,-7.78800e+000,-1.50242e+000}},
        {{-2.71361e+004, 2.00625e+004,-6.19444e+003, 1.10061e+003,-1.29354e+002, 1.08289e+001}},
        {{3.15252e+004,-3.31508e+004, 1.20371e+004,-2.23822e+003, 2.44396e+002,-2.09130e+001}}
    },
    // Component #6
    {
        {{1.27751e+004,-6.79813e+003, 1.24650e+003,-8.20622e+001,-2.33476e+000, 2.46459e-001}},
        {{3.64336e+005,-2.08457e+005, 4.33028e+004,-3.67825e+003, 4.22914e+001, 1.42701e+001}},
        {{-1.79298e+006, 1.01843e+006,-2.10037e+005, 1.82222e+004,-4.33573e+002,-2.72725e+001}}
    },
}};
// clang-format on

}  // namespace Acts
