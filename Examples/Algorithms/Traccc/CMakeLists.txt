add_library(ActsExamplesTraccc SHARED src/EdmConversion.cpp)

target_include_directories(
    ActsExamplesTraccc
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(
    ActsExamplesTraccc
    PUBLIC
        ActsCore
        ActsExamplesFramework
        ActsExamplesPropagation
        ActsPluginDetray
        traccc::core
        traccc::cuda
        vecmem::cuda
)

acts_compile_headers(ActsExamplesTraccc GLOB "include/**/*.hpp")

install(TARGETS ActsExamplesTraccc LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
