// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/EventData/Charge.hpp"
#include "Acts/EventData/MultiComponentBoundTrackParameters.hpp"
#include "Acts/Geometry/CuboidVolumeBuilder.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingGeometryBuilder.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Surfaces/RectangleBounds.hpp"
#include "Acts/Utilities/Logger.hpp"
#include "Acts/Visualization/GeometryView3D.hpp"
#include "Acts/Visualization/ObjVisualization3D.hpp"

#include "MultiEigenStepper.hpp"
#include "MultiEigenStepperSIMD.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

using MultiStepper = Acts::MultiEigenStepperSIMD<3>;
using MultiPropagator = Acts::Propagator<MultiStepper, Acts::Navigator>;

using ActionList = Acts::ActionList<Acts::MaterialInteractor>;
using AbortList = Acts::AbortList<Acts::EndOfWorldReached>;
using MultiPropagatorOptions =
    Acts::DenseStepperPropagatorOptions<ActionList, AbortList>;

using PropagatorState = MultiPropagator::State<MultiPropagatorOptions>;

int main() {
  auto magField = std::make_shared<Acts::ConstantBField>(
      Acts::Vector3(0.0, 0.0, 2.0 * Acts::UnitConstants::T));
  
  // Dirty workaround to create a PropagatorState object (this is just used to check if everything compiles)
  auto prop_state_ptr = (PropagatorState *)std::malloc(sizeof(PropagatorState));
  
  MultiStepper stepper(magField);

  const auto result = stepper.step(*prop_state_ptr);

  if (result.ok())
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}
