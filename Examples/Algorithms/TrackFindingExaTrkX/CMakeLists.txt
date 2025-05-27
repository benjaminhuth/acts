add_library(
    ActsExamplesTrackFindingExaTrkX
    SHARED
    src/TrackFindingAlgorithmExaTrkX.cpp
    src/PrototracksToParameters.cpp
    src/TrackFindingFromPrototrackAlgorithm.cpp
    src/TruthGraphBuilder.cpp
    src/createFeatures.cpp
)

target_include_directories(
    ActsExamplesTrackFindingExaTrkX
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(
    ActsExamplesTrackFindingExaTrkX
    PUBLIC ActsPluginExaTrkX ActsExamplesFramework ActsExamplesTrackFinding
)

if(ACTS_BUILD_PLUGIN_TRACCC)
    target_sources(
        ActsExamplesTrackFindingExaTrkX
        PUBLIC src/GNNTracccFullChainAlgorithm.cpp
    )

    target_link_libraries(ActsExamplesTrackFindingExaTrkX PUBLIC traccc::core)
endif()

install(
    TARGETS ActsExamplesTrackFindingExaTrkX
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
