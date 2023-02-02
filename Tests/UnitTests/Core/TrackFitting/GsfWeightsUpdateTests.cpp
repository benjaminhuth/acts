// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <boost/test/unit_test.hpp>

#include <Acts/Definitions/Units.hpp>
#include <Acts/EventData/VectorMultiTrajectory.hpp>
#include <Acts/TrackFitting/detail/GsfUtils.hpp>

using namespace Acts::UnitLiterals;

struct Component1D {
  double weight;
  double predicted;
  double predictedStddev;
};

struct Gaussian {
  double m_mean;
  double m_std;

  Gaussian(double mean, double std) : m_mean(mean), m_std(std) {}

  double operator()(double x) {
    auto exp = (x - m_mean) / m_std;
    return std::sqrt(2 * M_PI * m_std * m_std) * std::exp(-0.5 * exp * exp);
  }
};

template <std::size_t NComponents>
void compute1DimensionalCase(std::array<Component1D, NComponents> &components,
                             double measValue, double measStd) {
  double sum = 0.0;
  for (auto &[weight, pred, predStd] : components) {
    auto pdf = Gaussian(pred, std::sqrt(measStd*measStd + predStd*predStd));
    // std::cout << "factor: " << pdf(measValue) << " pred: " << pred
    //           << " diff: " << std::abs(pred - measValue) << "\n";
    weight *= pdf(measValue);
    sum += weight;
  }

  for (auto &[weight, pred, predStd] : components) {
    weight /= sum;
  }
}

BOOST_AUTO_TEST_CASE(MultivariateGaussianPDF) {
  const auto step = 0.1;
  const auto symrange = 3.0;
  std::vector<double> xvals(static_cast<std::size_t>(2 * symrange / step));
  std::generate(xvals.begin(), xvals.end(), [&, n = -symrange]() mutable {
    n += step;
    return n;
  });

  auto mean = -0.5;
  auto stddev = 0.3;

  auto pdf1D = Gaussian(mean, stddev);
  auto pdfMD = Acts::detail::MultivariateNormalPDF<1>(
      Acts::ActsVector<1>{mean}, Acts::ActsSymMatrix<1>{stddev * stddev});

  for (auto x : xvals) {
    BOOST_CHECK_CLOSE(pdf1D(x), pdfMD(Acts::ActsVector<1>{x}), 1.e-8);
  }
}

BOOST_AUTO_TEST_CASE(WeightsUpdate1D) {
  constexpr static std::size_t nComponents = 4;

  std::array<Component1D, nComponents> components = {
      {{1. / static_cast<double>(nComponents), 1.5_mm, 0.2_mm},
       {1. / static_cast<double>(nComponents), 2.5_mm, 0.1_mm},
       {1. / static_cast<double>(nComponents), 3.1_mm, 0.1_mm},
       {1. / static_cast<double>(nComponents), 3.5_mm, 0.2_mm}}};

  const double measValue = 3_mm;
  const double measStddev = 0.1_mm;

  Acts::VectorMultiTrajectory mtj;
  std::map<Acts::MultiTrajectoryTraits::IndexType, double> weights;
  std::vector<Acts::MultiTrajectoryTraits::IndexType> tips;

  for (auto i = 0ul; i < nComponents; ++i) {
    tips.push_back(mtj.addTrackState());
    auto proxy = mtj.getTrackState(tips.back());

    weights[tips.back()] = components[i].weight;

    Acts::Measurement<Acts::BoundIndices, 1> measurement(
        Acts::SourceLink({}, nullptr),
        std::array<Acts::BoundIndices, 1>{{Acts::eBoundLoc0}},
        Acts::ActsVector<1>{measValue},
        Acts::ActsSymMatrix<1>{measStddev * measStddev});

    proxy.allocateCalibrated(1);
    proxy.setCalibrated(measurement);

    proxy.predicted() = Acts::BoundVector::Ones() * components[i].predicted;
    proxy.predictedCovariance() = Acts::BoundSymMatrix::Identity() *
                                  components[i].predictedStddev *
                                  components[i].predictedStddev;
  }

  Acts::detail::computePosteriorWeights(mtj, tips, weights);
  Acts::detail::normalizeWeights(
      tips, [&](auto tip) -> double & { return weights.at(tip); });

  compute1DimensionalCase(components, measValue, measStddev);

  for (auto i = 0ul; i < nComponents; ++i) {
    BOOST_CHECK_CLOSE(components[i].weight, weights.at(tips.at(i)), 1.e-4);
  }
}
