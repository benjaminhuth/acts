// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Utilities/Definitions.hpp"
#include "Acts/Utilities/ParameterDefinitions.hpp"
#include <Acts/Utilities/Helpers.hpp>

namespace Acts {

class Surface;

namespace detail {

/// Convert free track parameters to bound track parameters.
///
/// @param freeParams Free track parameters vector
/// @param surface Surface onto which the parameters are bound
/// @param geoCtx Geometry context for the global-to-local transformation
/// @return Bound track parameters vector on the given surface
///
/// @warning The position is assumed to be on the surface. If this is not
///          the case, the behaviour is undefined.
BoundVector transformFreeToBoundParameters(const FreeVector& freeParams,
                                           const Surface& surface,
                                           const GeometryContext& geoCtx);

/// templated implementation for the Free-to-bound transform. Used for autodiff
template<typename T>
inline ActsVector<T, eBoundSize> transformFreeToBoundParametersImpl(const ActsVector<T, eFreeSize> &freeParams,
                                                             const ActsVector<T, 2> &localPosition)
{
  // initialize the bound vector
  ActsVector<T, eBoundSize> bp = ActsVector<T, eBoundSize>::Zero();
  
  auto direction = freeParams.template segment<3>(eFreeDir0);
  
  bp[eBoundLoc0] = localPosition[ePos0];
  bp[eBoundLoc1] = localPosition[ePos1];
  bp[eBoundTime] = freeParams[eFreeTime];
  bp[eBoundPhi] = VectorHelpers::phi(direction);
  bp[eBoundTheta] = VectorHelpers::theta(direction);
  bp[eBoundQOverP] = freeParams[eFreeQOverP];
  
  return bp;
}

/// Convert position and direction to bound track parameters.
///
/// @param position Global track three-position
/// @param time Global track time
/// @param direction Global direction three-vector; normalization is ignored.
/// @param qOverP Charge-over-momentum-like parameter
/// @param surface Surface onto which the parameters are bound
/// @param geoCtx Geometry context for the global-to-local transformation
/// @return Equivalent bound parameters vector on the given surface
///
/// @warning The position is assumed to be on the surface. If this is not
///          the case, the behaviour is undefined.
BoundVector transformFreeToBoundParameters(
    const Vector3D& position, FreeScalar time, const Vector3D& direction,
    FreeScalar qOverP, const Surface& surface, const GeometryContext& geoCtx);

/// Convert direction to curvilinear track parameters.
///
/// @param time Global track time
/// @param direction Global direction three-vector; normalization is ignored.
/// @param qOverP Charge-over-momentum-like parameter
/// @return Equivalent bound parameters vector on the curvilinear surface
///
/// @note The parameters are assumed to be defined at the origin of the
///       curvilinear frame derived from the direction vector. The local
///       coordinates are zero by construction.
BoundVector transformFreeToCurvilinearParameters(FreeScalar time,
                                                 const Vector3D& direction,
                                                 FreeScalar qOverP);

/// Convert direction angles to curvilinear track parameters.
///
/// @param time Global track time
/// @param phi Global transverse direction angle
/// @param theta Global longitudinal direction angle
/// @param qOverP Charge-over-momentum-like parameter
/// @return Equivalent bound parameters vector on the curvilinear surface
///
/// @note The parameters are assumed to be defined at the origin of the
///       curvilinear frame derived from the direction angles. The local
///       coordinates are zero by construction.
BoundVector transformFreeToCurvilinearParameters(FreeScalar time,
                                                 FreeScalar phi,
                                                 FreeScalar theta,
                                                 FreeScalar qOverP);

}  // namespace detail
}  // namespace Acts
