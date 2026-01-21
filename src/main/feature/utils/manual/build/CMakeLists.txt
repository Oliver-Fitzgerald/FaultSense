# Project
cmake_minimum_required(VERSION 3.10)
project(FaultSense_)

# Set policies
cmake_policy(SET CMP0072 NEW)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_OBJECT_DETECTION "Build object-detection executable with debug symbols" OFF)

# Packages
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Compilation
add_executable(hsv ../configure-hsv.cpp)
target_link_libraries(hsv ${OpenCV_LIBS} stdc++fs)

add_executable(edge ../configure-edge.cpp)
target_link_libraries(edge ${OpenCV_LIBS} stdc++fs)
