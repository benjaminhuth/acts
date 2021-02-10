set(unittest_extra_libraries ActsPluginOnnx)

add_executable(ActsUnitTestKDTree
  KDTreeTest.cpp)
target_link_libraries(ActsUnitTestKDTree
  PRIVATE ActsPluginOnnx)
