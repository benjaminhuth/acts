set(unittest_extra_libraries ActsPluginOnnx)

add_executable(ActsUnitTestKDTree KDTreeTest.cpp)
target_link_libraries(ActsUnitTestKDTree PRIVATE ActsPluginOnnx Boost::unit_test_framework)
  
add_executable(ActsUnitTestOnnxModel OnnxModelTest.cpp)
target_link_libraries(ActsUnitTestOnnxModel PRIVATE ActsPluginOnnx Boost::unit_test_framework Boost::filesystem)

file(COPY test_model.onnx DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
