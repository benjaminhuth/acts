#pragma once

#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Utilities/ParameterDefinitions.hpp"
#include "Acts/Surfaces/PlaneSurface.hpp"
#include "Acts/Surfaces/DiscSurface.hpp"
#include "Acts/Surfaces/CylinderSurface.hpp"
#include "Acts/Surfaces/ConeSurface.hpp"

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>


auto make_plane_surface_and_track(Acts::GeometryContext &gctx)
{
    // Just a random surface
    auto normal = Acts::Vector3D::Random().normalized();
    auto center = Acts::Vector3D::Random();
    auto surface = Acts::Surface::makeShared<Acts::PlaneSurface>(center,normal);
    
    // Shift the track away from the surface center
    auto track_shift = normal.cross(Acts::Vector3D::Random().normalized());
    
    // Track at the center of the surface, with some arbitrary momentum and qop
    Acts::FreeVector parameters = Acts::FreeVector::Zero();
    parameters.segment<3>(Acts::eFreePos0) = surface->center(gctx) + track_shift;
    parameters(Acts::eFreeTime) = 0.0;
    parameters.segment<3>(Acts::eFreeDir0) = Acts::Vector3D::Ones().normalized();
    parameters(Acts::eFreeQOverP) = 1.0;
    
    // Create path derivative
    Acts::FreeVector pathDerivatives = Acts::FreeVector::Zero();
    pathDerivatives.head<3>() = parameters.segment<3>(4);
    
    return std::make_tuple(surface, parameters, pathDerivatives);    
}

auto make_cylinder_surface_and_track(Acts::GeometryContext &gctx)
{
    Eigen::Quaterniond rot = Eigen::Quaterniond::UnitRandom();
    Eigen::Vector3d trans = Eigen::Vector3d::Random();
    Acts::Transform3D trafo = Acts::Transform3D::Identity();
    trafo.fromPositionOrientationScale(trans, rot, Eigen::Vector3d::Ones());
    double radius = 50;
    double halfz = 50;
    
    auto surface = Acts::Surface::makeShared<Acts::CylinderSurface>(trafo, radius, halfz);
    
    const auto cyl_surface = static_cast<Acts::CylinderSurface *>(surface.get());
    const auto sym_axis = cyl_surface->rotSymmetryAxis(gctx).normalized();
    const auto random_normal = sym_axis.cross(Eigen::Vector3d::Random().normalized()).normalized();
    
    Acts::FreeVector parameters = Acts::FreeVector::Zero();
    parameters.segment<3>(Acts::eFreePos0) = surface->center(gctx) + radius * random_normal;
    parameters(Acts::eFreeTime) = 0.0;
    parameters.segment<3>(Acts::eFreeDir0) = Acts::Vector3D::Ones().normalized();
    parameters(Acts::eFreeQOverP) = 1.0;
    
    // Create path derivative
    Acts::FreeVector pathDerivatives = Acts::FreeVector::Zero();
    pathDerivatives.head<3>() = parameters.segment<3>(4);
    
    return std::make_tuple(surface, parameters, pathDerivatives);    
}
