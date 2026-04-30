# Project
cmake_minimum_required(VERSION 3.10)
project(fault-sense-manual-configuration)

# Set policies
cmake_policy(SET CMP0072 NEW)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
include(FetchContent)

# Options

# Packages
FetchContent_Declare(
    simpleini
    GIT_REPOSITORY https://github.com/brofield/simpleini
    GIT_TAG        v4.25
)
FetchContent_MakeAvailable(simpleini)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Compilation
add_executable(manual-edge-configuration configure-edge.cpp)
target_link_libraries(manual-edge-configuration PRIVATE ${OpenCV_LIBS} stdc++fs)

add_executable(manual-hsv-configuration configure-hsv.cpp ../pre-processing/pre-processing.cpp ../pre-processing/utils/pre-processing-utils.cpp)
target_link_libraries(manual-hsv-configuration PRIVATE ${OpenCV_LIBS} stdc++fs)
