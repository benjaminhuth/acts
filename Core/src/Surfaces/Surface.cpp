// This file is part of the Acts project.
//
// Copyright (C) 2016-2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Surfaces/Surface.hpp"

#include <autodiff/forward.hpp>

#include "Acts/Surfaces/detail/AlignmentHelper.hpp"

#include <iomanip>
#include <iostream>
#include <utility>

Acts::Surface::Surface(const Transform3D& transform)
    : GeometryObject(), m_transform(transform) {}

Acts::Surface::Surface(const DetectorElementBase& detelement)
    : GeometryObject(), m_associatedDetElement(&detelement) {}

Acts::Surface::Surface(const Surface& other)
    : GeometryObject(other),
      std::enable_shared_from_this<Surface>(),
      m_transform(other.m_transform),
      m_surfaceMaterial(other.m_surfaceMaterial) {}

Acts::Surface::Surface(const GeometryContext& gctx, const Surface& other,
                       const Transform3D& shift)
    : GeometryObject(),
      m_transform(shift * other.transform(gctx)),
      m_associatedLayer(nullptr),
      m_surfaceMaterial(other.m_surfaceMaterial) {}

Acts::Surface::~Surface() = default;

bool Acts::Surface::isOnSurface(const GeometryContext& gctx,
                                const Vector3D& position,
                                const Vector3D& momentum,
                                const BoundaryCheck& bcheck) const {
  // global to local transformation
  auto lpResult = globalToLocal(gctx, position, momentum);
  if (lpResult.ok()) {
    return bcheck ? bounds().inside(lpResult.value(), bcheck) : true;
  }
  return false;
}

Acts::AlignmentToBoundMatrix Acts::Surface::alignmentToBoundDerivative(
    const GeometryContext& gctx, const FreeVector& parameters,
    const FreeVector& pathDerivative) const {
  // 1) Calculate the derivative of bound parameter local position w.r.t.
  // alignment parameters without path length correction
  const auto alignToBoundWithoutCorrection =
      alignmentToBoundDerivativeWithoutCorrection(gctx, parameters);
  // 2) Calculate the derivative of path length w.r.t. alignment parameters
  const auto alignToPath = alignmentToPathDerivative(gctx, parameters);
  // 3) Calculate the jacobian from free parameters to bound parameters
  FreeToBoundMatrix jacToLocal = jacobianGlobalToLocal(gctx, parameters);
  // 4) The derivative of bound parameters w.r.t. alignment
  // parameters is alignToBoundWithoutCorrection +
  // jacToLocal*pathDerivative*alignToPath
  
  std::cout << "By-Hand: alignToPath = \n" << alignToPath << std::endl;
  
  AlignmentToBoundMatrix alignToBound =
      alignToBoundWithoutCorrection + jacToLocal * pathDerivative * alignToPath;

  return alignToBound;
}

Acts::AlignmentToBoundMatrix
Acts::Surface::alignmentToBoundDerivativeWithoutCorrection(
    const GeometryContext& gctx, const FreeVector& parameters) const {
  // The global posiiton
  const auto position = parameters.segment<3>(eFreePos0);
  // The vector between position and center
  const ActsRowVector<AlignmentScalar, 3> pcRowVec =
      (position - center(gctx)).transpose();
  // The local frame rotation
  const auto& rotation = transform(gctx).rotation();
  // The axes of local frame
  const Vector3D localXAxis = rotation.col(0);
  const Vector3D localYAxis = rotation.col(1);
  const Vector3D localZAxis = rotation.col(2);
  // Calculate the derivative of local frame axes w.r.t its rotation
  const auto [rotToLocalXAxis, rotToLocalYAxis, rotToLocalZAxis] =
      detail::rotationToLocalAxesDerivative(rotation);
  // Calculate the derivative of local 3D Cartesian coordinates w.r.t.
  // alignment parameters (without path correction)
  AlignmentToLocalCartesianMatrix alignToLoc3D =
      AlignmentToLocalCartesianMatrix::Zero();
  alignToLoc3D.block<1, 3>(eX, eAlignmentCenter0) = -localXAxis.transpose();
  alignToLoc3D.block<1, 3>(eY, eAlignmentCenter0) = -localYAxis.transpose();
  alignToLoc3D.block<1, 3>(eZ, eAlignmentCenter0) = -localZAxis.transpose();
  alignToLoc3D.block<1, 3>(eX, eAlignmentRotation0) =
      pcRowVec * rotToLocalXAxis;
  alignToLoc3D.block<1, 3>(eY, eAlignmentRotation0) =
      pcRowVec * rotToLocalYAxis;
  alignToLoc3D.block<1, 3>(eZ, eAlignmentRotation0) =
      pcRowVec * rotToLocalZAxis;
  // The derivative of bound local w.r.t. local 3D Cartesian coordinates
  LocalCartesianToBoundLocalMatrix loc3DToBoundLoc =
      localCartesianToBoundLocalDerivative(gctx, position);
  // Initialize the derivative of bound parameters w.r.t. alignment
  // parameters without path correction
  AlignmentToBoundMatrix alignToBound = AlignmentToBoundMatrix::Zero();
  // It's only relevant with the bound local position without path correction
  alignToBound.block<2, eAlignmentSize>(eBoundLoc0, eAlignmentCenter0) =
      loc3DToBoundLoc * alignToLoc3D;
  return alignToBound;
}

namespace
{
    auto anglesToTransform(const autodiff::Vector3dual &angles)
    {        
        const auto phi = angles(0);   // x
        const auto theta = angles(1); // y
        const auto psi = angles(2);   // z
        
        using std::cos, std::sin;
        autodiff::Matrix3dual R;
        R <<  cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi) - cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi),
              cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi),
             -sin(theta),          sin(phi)*cos(theta),                              cos(phi)*cos(theta);
             
        return R;
    }
}


/// This in fact corresponds to Surface::alignmentToBoundDerivativeWithoutCorrection, but it is anyways just for testing
/// Works only for PlaneSurface now I think
Acts::AlignmentToBoundMatrix Acts::Surface::alignmentToBoundDerivativeAutodiff(
    const Acts::GeometryContext& gctx, 
    const Acts::FreeVector& free_params, 
    const Acts::FreeVector& pathDerivative) const
{
    if( type() != SurfaceType::Plane )
        throw std::runtime_error("alignmentToBoundDerivativeAutodiff is just available for Plane surfaces");
    
    using namespace autodiff::forward;
    using FreeVectorAutodiff = ActsVector<autodiff::dual,eFreeSize>;
    using AlignmentVectorAutodiff = ActsVector<autodiff::dual,eAlignmentSize>;
    using BoundVectorAutodiff = ActsVector<autodiff::dual,eBoundSize>;    
    
    // The transformation function: free params + alignment params -> bound_parameters
    auto f = [](const AlignmentVectorAutodiff &alignment_params, const FreeVectorAutodiff &track_params)
    {
        using std::atan2, std::sqrt;
        
        const auto center = alignment_params.segment<3>(eAlignmentCenter0);
        const auto angles = alignment_params.segment<3>(eAlignmentRotation0);
        const auto pos = track_params.segment<3>(eFreePos0);
        const auto dir = track_params.segment<3>(eFreeDir0);
        
        const auto R = anglesToTransform(angles);
        const auto loc_pos = R.transpose() * (pos - center);
        
        const auto phi = atan2(dir[1], dir[0]);
        const auto theta = atan2(sqrt(dir[0] * dir[0] + dir[1] * dir[1]), dir[2]);
        
        // NOTE this is not generic, just for plane/disk-surface
        BoundVectorAutodiff bound;
        bound(eBoundLoc0) = loc_pos(0);
        bound(eBoundLoc1) = loc_pos(1);
        bound(eBoundPhi) = phi;
        bound(eBoundTheta) = theta;
        bound(eBoundQOverP) = track_params(eFreeQOverP);
        bound(eBoundTime) = track_params(eFreeTime);
        
        return bound;
    };
    
    // The transformation function: free params + alignment params -> path correction
    auto g = [](const AlignmentVectorAutodiff &alignment_params, const FreeVectorAutodiff &track_params)
    {
        const auto center = alignment_params.segment<3>(eAlignmentCenter0);
        const auto angles = alignment_params.segment<3>(eAlignmentRotation0);
        
        const auto pos = track_params.segment<3>(eFreePos0);
        const auto dir = track_params.segment<3>(eFreeDir0);
        const auto R = anglesToTransform(angles);
        const auto ez = R.col(2);
        
        return autodiff::dual{ (center - pos).dot(ez) / dir.dot(ez) };
    };
    
    // Get alignment parameters
    autodiff::Vector3dual center = this->center(gctx).cast<autodiff::dual>();
    autodiff::Vector3dual rot = this->transform(gctx).rotation().eulerAngles(2,1,0).cast<autodiff::dual>();
    
    // reorder angles to get correct order in jacobian
    AlignmentVectorAutodiff alignment_params;
    alignment_params << center, rot(2), rot(1), rot(0); 
    FreeVectorAutodiff ad_free_params = free_params.cast<autodiff::dual>();
    
    // 1) alignToBound
    AlignmentToBoundMatrix alignToBound = 
        jacobian(f, wrt(alignment_params), at(alignment_params, ad_free_params)).cast<double>();
        
    // 2) jacToLocal
    FreeToBoundMatrix jacToLocal = 
        jacobian(f, wrt(ad_free_params), at(alignment_params, ad_free_params)).cast<double>();
        
    // 3) alignToPath
    AlignmentRowVector alignToPath = 
        gradient(g, wrt(alignment_params), at(alignment_params, ad_free_params)).cast<double>();
    
    // Combine the results
    alignToBound += jacToLocal * pathDerivative * alignToPath;
    
    return alignToBound;
}


Acts::AlignmentRowVector Acts::Surface::alignmentToPathDerivative(
    const GeometryContext& gctx, const FreeVector& parameters) const {
  // The global posiiton
  const auto position = parameters.segment<3>(eFreePos0);
  // The direction
  const auto direction = parameters.segment<3>(eFreeDir0);
  // The vector between position and center
  const ActsRowVector<AlignmentScalar, 3> pcRowVec =
      (position - center(gctx)).transpose();
  // The local frame rotation
  const auto& rotation = transform(gctx).rotation();
  // The local frame z axis
  const Vector3D localZAxis = rotation.col(2);
  // Cosine of angle between momentum direction and local frame z axis
  const double dz = localZAxis.dot(direction);
  // Calculate the derivative of local frame axes w.r.t its rotation
  const auto [rotToLocalXAxis, rotToLocalYAxis, rotToLocalZAxis] =
      detail::rotationToLocalAxesDerivative(rotation);
  // Initialize the derivative of propagation path w.r.t. local frame
  // translation (origin) and rotation      
  AlignmentRowVector alignToPath = AlignmentRowVector::Zero();
  alignToPath.segment<3>(eAlignmentCenter0) = localZAxis.transpose() / dz;
  alignToPath.segment<3>(eAlignmentRotation0) =
      -pcRowVec * rotToLocalZAxis / dz;
  return alignToPath;
}

std::shared_ptr<Acts::Surface> Acts::Surface::getSharedPtr() {
  return shared_from_this();
}

std::shared_ptr<const Acts::Surface> Acts::Surface::getSharedPtr() const {
  return shared_from_this();
}

Acts::Surface& Acts::Surface::operator=(const Surface& other) {
  if (&other != this) {
    GeometryObject::operator=(other);
    // detector element, identifier & layer association are unique
    m_transform = other.m_transform;
    m_associatedLayer = other.m_associatedLayer;
    m_surfaceMaterial = other.m_surfaceMaterial;
    m_associatedDetElement = other.m_associatedDetElement;
  }
  return *this;
}

bool Acts::Surface::operator==(const Surface& other) const {
  // (a) fast exit for pointer comparison
  if (&other == this) {
    return true;
  }
  // (b) fast exit for type
  if (other.type() != type()) {
    return false;
  }
  // (c) fast exit for bounds
  if (other.bounds() != bounds()) {
    return false;
  }
  // (d) compare  detector elements
  if (m_associatedDetElement != other.m_associatedDetElement) {
    return false;
  }
  // (e) compare transform values
  if (!m_transform.isApprox(other.m_transform, 1e-9)) {
    return false;
  }
  // (f) compare material
  if (m_surfaceMaterial != other.m_surfaceMaterial) {
    return false;
  }

  // we should be good
  return true;
}

// overload dump for stream operator
std::ostream& Acts::Surface::toStream(const GeometryContext& gctx,
                                      std::ostream& sl) const {
  sl << std::setiosflags(std::ios::fixed);
  sl << std::setprecision(4);
  sl << name() << std::endl;
  const Vector3D& sfcenter = center(gctx);
  sl << "     Center position  (x, y, z) = (" << sfcenter.x() << ", "
     << sfcenter.y() << ", " << sfcenter.z() << ")" << std::endl;
  Acts::RotationMatrix3D rot(transform(gctx).matrix().block<3, 3>(0, 0));
  Acts::Vector3D rotX(rot.col(0));
  Acts::Vector3D rotY(rot.col(1));
  Acts::Vector3D rotZ(rot.col(2));
  sl << std::setprecision(6);
  sl << "     Rotation:             colX = (" << rotX(0) << ", " << rotX(1)
     << ", " << rotX(2) << ")" << std::endl;
  sl << "                           colY = (" << rotY(0) << ", " << rotY(1)
     << ", " << rotY(2) << ")" << std::endl;
  sl << "                           colZ = (" << rotZ(0) << ", " << rotZ(1)
     << ", " << rotZ(2) << ")" << std::endl;
  sl << "     Bounds  : " << bounds();
  sl << std::setprecision(-1);
  return sl;
}

bool Acts::Surface::operator!=(const Acts::Surface& sf) const {
  return !(operator==(sf));
}
