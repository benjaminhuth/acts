add_executable(GsfTest GsfTest.cpp)

find_package(Qt5 COMPONENTS Core DataVisualization REQUIRED)

target_link_libraries(GsfTest ActsCore)
