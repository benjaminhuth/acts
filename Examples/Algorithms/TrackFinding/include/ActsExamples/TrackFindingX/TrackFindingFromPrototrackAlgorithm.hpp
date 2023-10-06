// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/Measurement.hpp"
#include "Acts/EventData/MultiTrajectory.hpp"
#include "Acts/EventData/VectorMultiTrajectory.hpp"
#include "Acts/MagneticField/MagneticFieldProvider.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/TrackFinding/MeasurementSelector.hpp"
#include "Acts/TrackFitting/GainMatrixSmoother.hpp"
#include "Acts/TrackFitting/GainMatrixUpdater.hpp"
#include "Acts/Utilities/Zip.hpp"
#include "ActsExamples/EventData/IndexSourceLink.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/Trajectories.hpp"
#include "ActsExamples/Framework/IAlgorithm.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/TrackFinding/TrackFindingAlgorithm.hpp"

#include <string>
#include <vector>

namespace ActsExamples {

class TrackFindingFromPrototrackAlgorithm final : public IAlgorithm {
 public:
  struct Config {
    /// Input prototracks collection.
    std::string inputProtoTracks;

    /// Input measurements
    std::string inputMeasurements;

    /// Input source links
    std::string inputSourceLinks;

    /// Input track parameters
    std::string inputInitialTrackParameters;

    /// Output protoTracks collection.
    std::string outputTracks;

    /// CKF measurement selector config
    Acts::MeasurementSelector::Config measurementSelectorCfg;

    /// CKF function
    std::shared_ptr<TrackFindingAlgorithm::TrackFinderFunction> findTracks;

    /// Tracking Geometry
    std::shared_ptr<const Acts::TrackingGeometry> trackingGeometry;

    /// Magnetic field
    std::shared_ptr<const Acts::MagneticFieldProvider> magneticField;
  };

  /// Constructor of the track finding algorithm
  ///
  /// @param cfg is the config struct to configure the algorithm
  /// @param level is the logging level
  TrackFindingFromPrototrackAlgorithm(Config cfg, Acts::Logging::Level lvl);

  virtual ~TrackFindingFromPrototrackAlgorithm() {}

  /// Filter the measurements
  ///
  /// @param ctx is the algorithm context that holds event-wise information
  /// @return a process code to steer the algorithm flow
  ActsExamples::ProcessCode execute(
      const ActsExamples::AlgorithmContext& ctx) const final;
      
  ActsExamples::ProcessCode finalize() override;

  const Config& config() const { return m_cfg; }

 private:
  Config m_cfg;
  
  mutable std::mutex m_mutex;
  mutable std::vector<unsigned> m_nTracksPerSeeds;

  ReadDataHandle<ProtoTrackContainer> m_inputProtoTracks{this,
                                                         "InputProtoTracks"};
  ReadDataHandle<MeasurementContainer> m_inputMeasurements{this,
                                                           "InputMeasurements"};
  ReadDataHandle<IndexSourceLinkContainer> m_inputSourceLinks{
      this, "InputSourceLinks"};
  ReadDataHandle<TrackParametersContainer> m_inputInitialTrackParameters{
      this, "InputInitialTrackParameters"};

  WriteDataHandle<ConstTrackContainer> m_outputTracks{this, "OutputTracks"};
};

}  // namespace ActsExamples
