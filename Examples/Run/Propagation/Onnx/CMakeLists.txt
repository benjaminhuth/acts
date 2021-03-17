add_executable(
    ActsExamplePropagationMLNavigation
    PropagationMLNavigation.cpp
)

target_link_libraries(ActsExamplePropagationMLNavigation
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

install(TARGETS ActsExamplePropagationMLNavigation
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
