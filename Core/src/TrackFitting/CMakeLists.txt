target_sources(
  ActsCore
  PRIVATE
    KalmanFitterError.cpp
    GainMatrixUpdater.cpp
    GainMatrixSmoother.cpp
    GsfError.cpp
    GsfUtils.cpp
)

set(matplotlib_cpp_DIR "/home/benjamin/Documents/matplotlib-cpp/lib64/matplotlib_cpp/cmake")
find_package(matplotlib_cpp REQUIRED)
find_package(fmt REQUIRED)

add_executable(TestBetheHeitlerApprox TestBetheHeitlerApprox.cpp)


set_target_properties(TestBetheHeitlerApprox
    PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS off
)

target_link_libraries(TestBetheHeitlerApprox ActsCore Boost::headers matplotlib_cpp::matplotlib_cpp fmt::fmt-header-only)
