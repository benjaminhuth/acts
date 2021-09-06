#add_executable(MultiStepperTest TestMultiStepper.cpp)
add_executable(GsfTelescopeTest TestGsfTelescope.cpp GsfAlgorithmFunction.cpp)
#add_executable(TargetSurfaceTest TestTargetSurface.cpp)
add_executable(GsfDD4hepTest TestGsfDD4hep.cpp GsfAlgorithmFunction.cpp)

set(EXAMPLES_LIBS
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
)

#target_link_libraries(MultiStepperTest ActsCore)
target_link_libraries(GsfTelescopeTest ActsCore ${EXAMPLES_LIBS} dfelibs)
target_link_libraries(GsfDD4hepTest ActsCore ActsPluginDD4hep ActsExamplesRecTracksCommon ${EXAMPLES_LIBS} ActsExamplesDetectorDD4hep dfelibs Boost::program_options)
#target_link_libraries(TargetSurfaceTest ActsCore ActsExamplesFramework ActsExamplesGenerators ActsExamplesCommon ActsExamplesDigitizationCommon ActsExamplesFatras ActsExamplesDetectorTelescope ActsExamplesTrackFitting)

add_executable(GeoIdDecoder DecodeGeoId.cpp)
target_link_libraries(GeoIdDecoder ActsCore)


# Tests
add_executable(TestComponentMerging TestComponentMerging.cpp)
target_link_libraries(TestComponentMerging ActsCore Boost::unit_test_framework)
