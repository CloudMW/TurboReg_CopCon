# bindings/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

file(GLOB PYBIND_SOURCES
    ${CMAKE_SOURCE_DIR}/turboreg/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pybind.cpp
)

add_library(turboreg_gpu SHARED ${PYBIND_SOURCES})

target_include_directories(turboreg_gpu PRIVATE
    ${CMAKE_SOURCE_DIR}/turboreg/include
    ${EIGEN3_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
)

target_link_libraries(turboreg_gpu PRIVATE ${TORCH_LIBRARIES} {PCL_LIBRARIES})
set_target_properties(turboreg_gpu PROPERTIES PREFIX "")