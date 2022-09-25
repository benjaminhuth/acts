set(SOURCES "")

if(ACTS_EXATRKX_ENABLE_ONNX OR ACTS_EXATRKX_ENABLE_TORCH)
  list(APPEND SOURCES
    src/buildEdgesFRNN.cpp
  )
endif()

if(ACTS_EXATRKX_ENABLE_ONNX)
  list(APPEND SOURCES 
    src/ExaTrkXTrackFindingOnnx.cpp
  )
endif()

if(ACTS_EXATRKX_ENABLE_TORCH)
  list(APPEND SOURCES 
    src/ExaTrkXTrackFindingTorch.cpp
  )
endif()

if(ACTS_EXATRKX_ENABLE_CPU)
  list(APPEND SOURCES
    src/buildEdgesTBB.cpp
    src/ExaTrkXTrackFindingCpu.cpp
  )
endif()

add_library(
  ActsPluginExaTrkX SHARED
  ${SOURCES}
)

set_target_properties(ActsPluginExaTrkX
  PROPERTIES  
    BUILD_RPATH                         "\$ORIGIN"
    INSTALL_RPATH                       "\$ORIGIN"
    CXX_STANDARD                        17
    CXX_STANDARD_REQUIRED               ON
    CUDA_STANDARD                       17
    CUDA_STANDARD_REQUIRED              ON
    INTERFACE_POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION          ON
)

target_compile_options(ActsPluginExaTrkX
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${EXATRKX_CXX_FLAGS}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:${EXATRKX_CUDA_FLAGS}>")
            
target_compile_definitions(ActsPluginExaTrkX PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
target_compile_definitions(ActsPluginExaTrkX PUBLIC TRITON_ENABLE_GPU)

target_include_directories(
  ActsPluginExaTrkX
  PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(
  ActsPluginExaTrkX
  PUBLIC
    ActsCore
    Boost::boost
  PRIVATE
    ${TORCH_LIBRARIES}
)

if(ACTS_EXATRKX_ENABLE_ONNX OR ACTS_EXATRKX_ENABLE_TORCH)
  target_link_libraries(
    ActsPluginExaTrkX
    PRIVATE
      frnn
  )
endif()

if(ACTS_EXATRKX_ENABLE_ONNX)
  target_compile_definitions(ActsPluginExaTrkX PUBLIC ACTS_EXATRKX_ONNX_BACKEND)

  target_link_libraries(
    ActsPluginExaTrkX
    PRIVATE
      OnnxRuntime
      cugraph::cugraph
  )
endif()

if(ACTS_EXATRKX_ENABLE_TORCH)
  target_compile_definitions(ActsPluginExaTrkX PUBLIC ACTS_EXATRKX_TORCH_BACKEND)

  target_link_libraries(
    ActsPluginExaTrkX
    PRIVATE
      TorchScatter::TorchScatter
  )

  # Should not discard TorchScatter even if its not needed at this point 
  # since we need the scatter_max operation in the torch script later
  target_link_options(
    ActsPluginExaTrkX
    PUBLIC
      "-Wl,-no-as-needed"
  )
endif()

if(ACTS_EXATRKX_ENABLE_CPU)
  target_compile_definitions(ActsPluginExaTrkX PUBLIC ACTS_EXATRKX_CPU_BACKEND)

  target_link_libraries(
    ActsPluginExaTrkX
    PRIVATE
      TBB::tbb
  )

  target_link_libraries(
    ActsPluginExaTrkX
    PRIVATE
      TorchScatter::TorchScatter
  )

  # Should not discard TorchScatter even if its not needed at this point
  # since we need the scatter_max operation in the torch script later
  target_link_options(
    ActsPluginExaTrkX
    PUBLIC
      "-Wl,-no-as-needed"
  )
endif()



install(
  TARGETS ActsPluginExaTrkX
  EXPORT ActsPluginExaTrkXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(
  DIRECTORY include/Acts
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
