# Project
cmake_minimum_required(VERSION 3.10)
project(FaultSense)

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
add_executable(gabor ../gabor-filter-beta.cpp)
target_link_libraries(gabor ${OpenCV_LIBS} stdc++fs)

add_executable(texture ../texture-beta.cpp 
    ../../generic-utils.cpp
    ../../pre-processing-utils.cpp
    ../../../object-detection.cpp)
target_link_libraries(texture ${OpenCV_LIBS} stdc++fs)

add_executable(shadow ../shadow-detection-removal-beta.cpp 
    ../../generic-utils.cpp
    ../../pre-processing-utils.cpp
    ../../../object-detection.cpp)
target_link_libraries(shadow ${OpenCV_LIBS} stdc++fs)

add_executable(user-select ../user-select-beta.cpp)
target_link_libraries(user-select ${OpenCV_LIBS} stdc++fs)
