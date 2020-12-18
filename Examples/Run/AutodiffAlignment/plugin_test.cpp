#include <Acts/Surfaces/PlaneSurface.hpp>
#include <Acts/Geometry/GeometryContext.hpp>
#include <Acts/Utilities/Definitions.hpp>
#include <iomanip>
#include <cstdlib>
#include <chrono>

#include <Acts/Plugins/Autodiff/AutodiffAlignmentDerivatives.hpp>

#include "common.hpp"

int main()
{
    Acts::GeometryContext gctx;    
    
//     auto [surface, parameters, pathDerivatives] = make_plane_surface_and_track(gctx);
    auto [surface, parameters, pathDerivatives] = make_cylinder_surface_and_track(gctx);
    
    Acts::Vector3D free_pos = parameters.segment<3>(Acts::eFreePos0);
    Acts::Vector3D free_dir = parameters.segment<3>(Acts::eFreeDir0);
    auto loc_pos = surface->globalToLocal(gctx, free_pos, free_dir);
    
    // check if the generated track is on the surface
    if( !loc_pos.ok() )
        throw std::runtime_error("in main: track not ok: " + loc_pos.error().message());
    else
        std::cout << "track seems to be on surface!" << std::endl;
    
    // STANDARD METHOD
    auto alignToBound = surface->alignmentToBoundDerivative(gctx, parameters, pathDerivatives);
    
//     std::cout << "STANDARD METHOD\n";
//     std::cout << std::setprecision(3) << alignToBound << "\n" << std::endl;
    
    // AUTODIFF PLUGIN METHOD
    auto alignToBoundAutodiff = Acts::alignmentToBoundDerivative(surface.get(), gctx, parameters, pathDerivatives,
                                                                 Acts::AlignmentToBoundDerivativeVisitorAutodiff());
    
//     std::cout << "AUTODIFF METHOD\n";
//     std::cout << alignToBoundAutodiff << "\n" << std::endl;
    
    // DIFFERENCE
    Acts::AlignmentToBoundMatrix diff = (alignToBound - alignToBoundAutodiff).cwiseAbs();
    
    // some tolerance for numerical errors
    for(auto p = diff.data(); p != diff.data()+diff.size(); ++p)
        *p = *p < 1.e-5 ? 0.0 : *p;
    
    std::cout << "DIFFERENCE\n";
    std::cout << diff << "\n";
}
