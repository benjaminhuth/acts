#include <iostream>

#include <Eigen/Geometry>

#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

#include "Acts/Surfaces/detail/AlignmentHelper.hpp"

int main()
{
    enum class axes { x, y, z };
    
    auto rand_rot = Eigen::Quaterniond::UnitRandom().matrix();
    
    auto f = [&rand_rot](const auto &euler_angles, axes ax) -> autodiff::Vector3dual
    {
        autodiff::Matrix3dual R;
        
        R = Eigen::AngleAxis<autodiff::dual>(euler_angles(2), autodiff::Vector3dual::UnitZ())
          * Eigen::AngleAxis<autodiff::dual>(euler_angles(1), autodiff::Vector3dual::UnitY())
          * Eigen::AngleAxis<autodiff::dual>(euler_angles(0), autodiff::Vector3dual::UnitX());
          
        if( !rand_rot.isApprox(R.cast<double>()) )
            throw std::runtime_error("wrong rotation matrix");
          
        switch(ax)
        {
            case axes::x:
                return R.col(0);
            case axes::y:
                return R.col(1);
            case axes::z:
                return R.col(2);
        }
    };
    
    
    autodiff::Vector3dual euler_angles = rand_rot.eulerAngles(2,1,0).cast<autodiff::dual>().reverse();
    
    
    const auto [rotToLocalXAxis, rotToLocalYAxis, rotToLocalZAxis] =
        Acts::detail::rotationToLocalAxesDerivative(rand_rot);
        
    using namespace autodiff::forward;
    bool error = false;
    
    // X axis
    const auto x = jacobian(f, wrt(euler_angles), at(euler_angles, axes::x)).cast<double>();
    if( !rotToLocalXAxis.isApprox(x) )
    {
        std::cout << "x axis diff = \n" << rotToLocalXAxis - x << "\n\n";
        error = true;
    }
    
    // Y axis
    const auto y = jacobian(f, wrt(euler_angles), at(euler_angles, axes::y)).cast<double>();
    if( !rotToLocalYAxis.isApprox(y) )
    {
        std::cout << "y axis diff = \n" << rotToLocalYAxis - y << "\n\n";
        error = true;
    }
    
    // Z axis
    const auto z = jacobian(f, wrt(euler_angles), at(euler_angles, axes::z)).cast<double>();
    if( !rotToLocalZAxis.isApprox(z) )
    {
        std::cout << "z axis diff = \n" << rotToLocalZAxis - z << "\n\n";
        error = true;
    }
    
    if( error )
        std::cout << "Errors occured!\n";
    else
        std::cout << "No errors occured!\n";
}
