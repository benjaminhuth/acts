#include "Acts/Geometry/GeometryIdentifier.hpp"

#include <iostream>


int main(int argc, char **argv)
{
    if( argc != 2 )
    {
        std::cerr << "Usage: " << argv[0] << " <Acts Geometry ID>" << std::endl;
        return EXIT_FAILURE;
    }

    Acts::GeometryIdentifier::Value v = std::stoul(argv[1]);

    std::cout << Acts::GeometryIdentifier(v) << std::endl;
    return EXIT_SUCCESS;
}
