// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/TrackFitting/detail/GsfUtils.hpp"

#include "Acts/EventData/MeasurementHelpers.hpp"

namespace Acts {
namespace detail {

using TrackStateTraits =
    TrackStateTraits<MultiTrajectoryTraits::MeasurementSizeMax, true>;

ActsScalar calculateDeterminant(
    const double* fullCalibrated, const double* fullCalibratedCovariance,
    TrackStateTraits::Covariance predictedCovariance,
    TrackStateTraits::Projector projector, unsigned int calibratedSize) {
  return visit_measurement(calibratedSize, [&](auto N) {
    constexpr size_t kMeasurementSize = decltype(N)::value;

    typename Acts::TrackStateTraits<kMeasurementSize, true>::Measurement
        calibrated{fullCalibrated};

    typename Acts::TrackStateTraits<
        kMeasurementSize, true>::MeasurementCovariance calibratedCovariance{
        fullCalibratedCovariance};

    const auto H =
        projector.template topLeftCorner<kMeasurementSize, eBoundSize>().eval();

    return (H * predictedCovariance * H.transpose() + calibratedCovariance)
        .determinant();
  });
}

ActsScalar calculateFactor(const double* fullMeas, const double* fullMeasCov,
                           TrackStateTraits::Parameters predicted,
                           TrackStateTraits::Covariance predictedCov,
                           TrackStateTraits::Projector projector,
                           unsigned int calibratedSize) {
  return visit_measurement(calibratedSize, [&](auto N) {
    constexpr size_t K = decltype(N)::value;

    typename Acts::TrackStateTraits<K, true>::Measurement meas{fullMeas};
    typename Acts::TrackStateTraits<K, true>::MeasurementCovariance measCov{
        fullMeasCov};

    auto H = projector.template topLeftCorner<K, eBoundSize>().eval();

    const auto pdf = MultivariateNormalPDF<K>(
        H * predicted, measCov + H * predictedCov * H.transpose());

    // std::cout << "measCov + predictedCov = " << measCov + H * predictedCov * H.transpose() << "\n";
    //
    // std::cout << "factor Core: " << pdf(meas) << " - pred " << predicted.transpose() << "\n";

    return pdf(meas);
  });
}

}  // namespace detail
}  // namespace Acts
