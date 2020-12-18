#include <Acts/Surfaces/PlaneSurface.hpp>
#include <Acts/Geometry/GeometryContext.hpp>
#include <Acts/Utilities/Definitions.hpp>
#include <iomanip>
#include <cstdlib>
#include <chrono>

#include "common.hpp"

int main()
{    
    Acts::GeometryContext gctx;    
    
    // First do correctness check
    auto [surface, parameters, pathDerivatives] = make_plane_surface_and_track(gctx);
    
    // STANDARD METHOD
    auto alignToBound = surface->alignmentToBoundDerivative(gctx, parameters, pathDerivatives);
    
    std::cout << "STANDARD METHOD\n";
    std::cout << std::setprecision(3) << alignToBound << "\n\n";
    
    // AUTODIFF METHOD
    auto alignToBoundAutodiff = surface->alignmentToBoundDerivativeAutodiff(gctx, parameters, pathDerivatives);
    
    std::cout << "AUTODIFF METHOD\n";
    std::cout << alignToBoundAutodiff << "\n\n";
    
    // DIFFERENCE
    Acts::AlignmentToBoundMatrix diff = (alignToBound - alignToBoundAutodiff).cwiseAbs();
    
    // some tolerance for numerical errors
    for(auto p = diff.data(); p != diff.data()+diff.size(); ++p)
        *p = *p < 1.e-5 ? 0.0 : *p;
    
    std::cout << "DIFFERENCE\n";
    std::cout << diff << "\n";
    
    
    // Now benchmark
    std::vector<decltype(make_plane_surface_and_track(gctx))> samples;
    
    for(std::size_t i=0ul; i<1000; ++i)
        samples.push_back(make_plane_surface_and_track(gctx));
    
    std::vector<Acts::AlignmentToBoundMatrix> normal_results;
    normal_results.reserve(samples.size());
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    for(auto &[surf, pars, pathDev] : samples)
        normal_results.push_back(surf->alignmentToBoundDerivative(gctx, pars, pathDev));
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    const auto t_normal = std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(t1-t0);
    
    
    t0 = std::chrono::high_resolution_clock::now();
    
    for(auto &[surf, pars, pathDev] : samples)
        normal_results.push_back(surf->alignmentToBoundDerivativeAutodiff(gctx, pars, pathDev));
    
    t1 = std::chrono::high_resolution_clock::now();
    
    const auto t_autodiff = std::chrono::duration_cast<std::chrono::duration<double,std::milli>>(t1-t0);
    
    std::cout << "t_autodiff: " << t_autodiff.count() << std::endl;
    std::cout << "t_normal: " << t_normal.count() << std::endl;
    std::cout << "ratio: " << t_autodiff.count() / t_normal.count() << std::endl;
}
