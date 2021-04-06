#pragma once

#include <Acts/Surfaces/Surface.hpp>
#include <Acts/Surfaces/PlaneSurface.hpp>
#include <Acts/Surfaces/DiscSurface.hpp>
#include <Acts/Surfaces/ConeSurface.hpp>
#include <Acts/Surfaces/CylinderSurface.hpp>
#include <Acts/Surfaces/LineSurface.hpp>
#include <Acts/EventData/detail/TransformationFreeToBound.hpp>

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

#include <tuple>

namespace Acts {
    
using Vector3Autodiff = ActsVector<autodiff::dual, 3>;
using Vector2Autodiff = ActsVector<autodiff::dual, 2>;
using FreeVectorAutodiff = ActsVector<autodiff::dual,eFreeSize>;
using AlignmentVectorAutodiff = ActsVector<autodiff::dual,eAlignmentSize>;
using BoundVectorAutodiff = ActsVector<autodiff::dual,eBoundSize>;
using Transform3Autodiff = Eigen::Transform<autodiff::dual, 3, Eigen::Affine>;
    
namespace detail {
    
inline auto alignmentParamsToTransform(const AlignmentVectorAutodiff &ap)
{        
    Transform3Autodiff transform = Transform3Autodiff::Identity();
        
    transform.rotate(Eigen::AngleAxis<autodiff::dual>(ap(eAlignmentRotation2), Vector3Autodiff::UnitZ())
                   * Eigen::AngleAxis<autodiff::dual>(ap(eAlignmentRotation1), Vector3Autodiff::UnitY())
                   * Eigen::AngleAxis<autodiff::dual>(ap(eAlignmentRotation0), Vector3Autodiff::UnitX()));
    
    transform.matrix().block<3,1>(0,3) = ap.segment<3>(eAlignmentCenter0);
            
    return transform;
}

// Metafunction to check if type T is contained in one of the types U...
template<typename T, typename... U>
struct is_in
{
    static constexpr bool value = (std::is_same_v<T,U> || ...);
};

}


struct AlignmentToBoundDerivativeVisitorAutodiff
{
    template<typename Surface>
    auto operator()(const Surface &surface, const GeometryContext &gctx, 
                    const FreeVector &trackParams, const FreeVector &pathDerivative) const
    {
        using surface_t = std::remove_cv_t<Surface>;
        static_assert( detail::is_in<surface_t, CylinderSurface, ConeSurface, LineSurface, PlaneSurface, DiscSurface>::value );
        
        // The transformation function: free params + alignment params -> bound_parameters
        auto f = [&surface](const auto &alignmentParams, const auto &freeParams)
        {
            const auto transform = detail::alignmentParamsToTransform(alignmentParams);
            
            const Vector3Autodiff pos = freeParams.template segment<3>(eFreePos0);
            const Vector3Autodiff dir = freeParams.template segment<3>(eFreeDir0);
            
            auto locPos = Result<Vector2Autodiff>::success({0.0, 0.0});
            
            if constexpr( std::is_same_v<surface_t, CylinderSurface> || std::is_same_v<surface_t, ConeSurface> )
                locPos = surface_t::globalToLocalImpl(pos, transform, surface.bounds(), s_onSurfaceTolerance);
            else if constexpr( std::is_same_v<surface_t, LineSurface> )
                locPos = surface_t::globalToLocalImpl(pos, dir, transform);
            else if constexpr( std::is_same_v<surface_t, PlaneSurface> || std::is_same_v<surface_t, DiscSurface> )
                locPos = surface_t::globalToLocalImpl(pos, transform, s_onSurfaceTolerance);
            
            if( !locPos.ok() )
                throw std::runtime_error("result not ok: " + locPos.error().message());
            else
                return detail::transformFreeToBoundParametersImpl(freeParams, locPos.value());
        };    
        
        // Get alignment parameters
        auto g = [&surface](const AlignmentVectorAutodiff &alignmentParams, const FreeVectorAutodiff &freeParams)
        {            
            const auto center = alignmentParams.segment<3>(eAlignmentCenter0);
            const auto pos = freeParams.segment<3>(eFreePos0);
            const auto dir = freeParams.segment<3>(eFreeDir0);
            const auto R = detail::alignmentParamsToTransform(alignmentParams).rotation();
            
            const auto ex = R.col(0);
            const auto ey = R.col(1);
            const auto ez = R.col(2);
            
            const auto diff = center - pos;
            const auto denom = 1 - dir.dot(ez)*dir.dot(ez);
            
            autodiff::dual delta_s = 0;
            
            if constexpr( std::is_same_v<surface_t, PlaneSurface> || std::is_same_v<surface_t, DiscSurface> )
                delta_s = diff.dot(ez) / dir.dot(ez);
            else if constexpr( std::is_same_v<surface_t, LineSurface> )
                delta_s = diff.dot( dir - ez * dir.dot(ez)) / denom;
            else if constexpr( std::is_same_v<surface_t, CylinderSurface> )
                delta_s = 2*( diff.dot(ex) * dir.dot(ex) + diff.dot(ey) * dir.dot(ey) ) / denom;
            else if constexpr( std::is_same_v<surface_t, ConeSurface> )
            {
                const auto tanAlphaSq = std::pow(surface.bounds().tanAlpha(), 2);
                delta_s = 2*(  dir.dot(ez) * diff.dot(ez) * tanAlphaSq + 
                    diff.dot(ex)*dir.dot(ex) + diff.dot(ey)*dir.dot(ey) ) / ( denom * (1 + tanAlphaSq) );
            }
            
            return delta_s;
        };
        
        autodiff::Vector3dual center = surface.center(gctx).template cast<autodiff::dual>();
        autodiff::Vector3dual rot = surface.transform(gctx).rotation().eulerAngles(2,1,0).template cast<autodiff::dual>();
        
        // Insert angles in a way, that order is x-y-z
        AlignmentVectorAutodiff ap;
        ap << center, rot(2), rot(1), rot(0); 
        
        FreeVectorAutodiff fp = trackParams.template cast<autodiff::dual>();
        
        using namespace autodiff::forward;
        
        // 1) alignToBound
        AlignmentToBoundMatrix alignToBound = 
            jacobian(f, wrt(ap), at(ap, fp)).template cast<double>();
            
        // 2) jacToLocal
        FreeToBoundMatrix jacToLocal = 
            jacobian(f, wrt(fp), at(ap, fp)).template cast<double>();
            
        // 3) alignToPath
        AlignmentRowVector alignToPath = 
            gradient(g, wrt(ap), at(ap, fp)).template cast<double>();
            
        std::cout << "Autodiff: alignToPath = \n" << alignToPath << std::endl;
            
        alignToBound += jacToLocal * pathDerivative * alignToPath;
        
        return alignToBound;
    }
};  
    
template<typename Visitor /* = DefaultDerivativeEvaluator */ >
auto alignmentToBoundDerivative(Surface *surface, const GeometryContext &gctx, const FreeVector &trackParams, 
                                const FreeVector &pathDerivative, const Visitor &visitor)
{
    AlignmentToBoundMatrix res = AlignmentToBoundMatrix::Zero();
    
    switch(surface->type())
    {
        case Surface::SurfaceType::Cylinder:
            res = visitor(*static_cast<CylinderSurface *>(surface), gctx, trackParams, pathDerivative);
            break;
        case Surface::SurfaceType::Cone:
            res = visitor(*static_cast<ConeSurface *>(surface), gctx, trackParams, pathDerivative);
            break;
        case Surface::SurfaceType::Plane:
            res = visitor(*static_cast<PlaneSurface *>(surface), gctx, trackParams, pathDerivative);
            break;
        case Surface::SurfaceType::Disc:
            res = visitor(*static_cast<DiscSurface *>(surface), gctx, trackParams, pathDerivative);
            break;
        default:
            throw std::runtime_error("unsupported surface type");
    }
    
    return res;
}

} // namespace Acts                                              
                                                                  
