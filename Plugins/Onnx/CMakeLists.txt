add_library(
  ActsPluginOnnx SHARED
  # header files
  include/Acts/Plugins/Onnx/OnnxRuntimeBase.hpp
  include/Acts/Plugins/Onnx/MLTrackClassifier.hpp
  include/Acts/Plugins/Onnx/MLNavigator.hpp
  include/Acts/Plugins/Onnx/OnnxModel.hpp
  include/Acts/Plugins/Onnx/KDTree.hpp
  include/Acts/Plugins/Onnx/PairwiseScoreModel.hpp
  include/Acts/Plugins/Onnx/TargetPredModel.hpp
  include/Acts/Plugins/Onnx/beampipe_split.hpp
  # source files
  src/OnnxRuntimeBase.cpp
  src/MLTrackClassifier.cpp)

target_include_directories(
  ActsPluginOnnx
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(
  ActsPluginOnnx
  PUBLIC ActsCore OnnxRuntime)

install(
  TARGETS ActsPluginOnnx
  EXPORT ActsPluginOnnxTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
  DIRECTORY include/Acts
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
