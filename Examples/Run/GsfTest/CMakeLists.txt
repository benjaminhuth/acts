################################################
# Library providing the GSF algorithm function #
################################################
add_library(GsfAlgorithmFunctionLib SHARED
  GsfAlgorithmFunctionStandard.cpp
  GsfAlgorithmFunctionDirect.cpp
  GsfAlgorithmFunctionSettings.cpp
  GsfInfrastructure/MultiStepperError.cpp
  GsfInfrastructure/GsfError.cpp)

target_include_directories(
  GsfAlgorithmFunctionLib
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(
  GsfAlgorithmFunctionLib
  PUBLIC
  ActsCore
  ActsExamplesTrackFitting)


#######################
# Example executables #
#######################

set(COMMON_LIBS
    ActsCore
    ActsExamplesFramework
    ActsExamplesGenerators
    ActsExamplesCommon
    ActsExamplesDigitizationCommon
    ActsExamplesFatras
    ActsExamplesDetectorTelescope
    ActsExamplesTrackFitting
    ActsExamplesIoPerformance
    ActsExamplesIoRoot
    ActsExamplesTrackFinding
    ActsExamplesIoObj
    dfelibs
    GsfAlgorithmFunctionLib
)

add_library(GsfTestLib TestGsfGeneric.cpp)
target_link_libraries(GsfTestLib ${COMMON_LIBS})

# Telescope
add_executable(GsfTelescopeTest TestGsfTelescope.cpp)
target_link_libraries(GsfTelescopeTest ${COMMON_LIBS} GsfTestLib)

# DD4hep
add_executable(GsfDD4hepTest TestGsfDD4hep.cpp)
target_link_libraries(GsfDD4hepTest 
    ${COMMON_LIBS}
    ActsPluginDD4hep
    ActsExamplesRecTracksCommon
    ActsExamplesDetectorDD4hep
    Boost::program_options
    GsfTestLib)

#add_executable(MultiStepperTest TestMultiStepper.cpp)
#target_link_libraries(MultiStepperTest ActsCore)

#add_executable(TargetSurfaceTest TestTargetSurface.cpp)
#target_link_libraries(TargetSurfaceTest ${COMMON_LIBS})

add_executable(GeoIdDecoder DecodeGeoId.cpp)
target_link_libraries(GeoIdDecoder ActsCore)

#########
# Tests #
#########

add_executable(TestComponentMerging TestComponentMerging.cpp)
target_link_libraries(TestComponentMerging ActsCore Boost::unit_test_framework)
