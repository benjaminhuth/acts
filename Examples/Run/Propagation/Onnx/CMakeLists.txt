add_executable(
    ActsExamplePropagationWithMLNavigation
    PropagationWithMLNavigation.cpp
)

target_link_libraries(ActsExamplePropagationWithMLNavigation
  ActsCore
  ActsPluginOnnx
  ActsExamplesFramework
  ActsExamplesPropagation
  ActsExamplesCommon
  ActsExamplesMagneticField
  ActsExamplesIoObj
  ActsExamplesIoJson
  ActsExamplesIoRoot
  ActsExamplesDetectorGeneric
  Boost::program_options
  Boost::filesystem
  dfelibs
)

install(TARGETS ActsExamplePropagationWithMLNavigation
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
