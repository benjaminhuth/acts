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

if(ACTS_BUILD_PLUGIN_TRACCC AND ACTS_EXATRKX_ENABLE_TRACCC)
    target_sources(
        ActsExamplesTrackFindingExaTrkX
        PRIVATE src/GNNTracccFullChainAlgorithm.cpp
    )

    target_link_libraries(
        ActsExamplesTrackFindingExaTrkX
        PUBLIC ActsExamplesTraccc
    )
endif()

install(
    TARGETS ActsExamplesTrackFindingExaTrkX
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
