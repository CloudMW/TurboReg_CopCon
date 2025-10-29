set(CMAKE_TOOLCHAIN_FILE "D:\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake")
set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "")
cmake_minimum_required(VERSION 3.10)
project(TurboReg )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# 强制静态链接（针对MSVC）
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options(/MT$<$<CONFIG:Debug>:d>)
endif()

option(BUILD_DEMO_CPP "Build C++ demo" ON)
option(BUILD_PYBIND "Build Python bindings with pybind11" ON)


#message(STATUS "VCPKG_ROOT 环境变量的值: $ENV{VCPKG_ROOT}")

#vcpkg
#if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
#
#        CACHE STRING "")
#endif()

set(VCPKG_TARGET_TRIPLET "x64-windows-static" CACHE STRING "VCPKG target triplet")
find_package(PCL REQUIRED)
set(TORCH_CMAKE_PREFIX_PATH "D:\\envs\\anaconda3\\py310\\lib\\site-packages\\torch\\share\\cmake")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

# LibTorch
find_package(Torch REQUIRED)


# Eigen
find_package(Eigen3 REQUIRED)

set(EIGEN3_INCLUDE_DIR "D:\\vcpkg\\installed\\x64-windows\\include\\eigen3")
message(STATUS "EIGEN3_INCLUDE_DIR : ${EIGEN3_INCLUDE_DIR}")
include_directories(${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS})

# TurboReg Core Library
include_directories(turboreg/include)
add_subdirectory(turboreg)

# Demos
if (BUILD_DEMO_CPP)
    add_subdirectory(demo_cpp)
endif()

message(STATUS "---- All CMake Variables ----")
get_cmake_property(_vars VARIABLES)
foreach(_var ${_vars})
    message(STATUS "${_var} = ${${_var}}")
endforeach()
message(STATUS "-----------------------------")