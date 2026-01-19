# Project
cmake_minimum_required(VERSION 4.1.1)
project(FaultSense)

# Set policies
cmake_policy(SET CMP0072 NEW)

# Set C++ version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)

# Packages
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

# Compilation
add_executable(data-preperation ../src/main/data-preperation/generate-masks.cpp
                                ../src/main/feature/utils/generic-utils.cpp)
target_link_libraries(data-preperation ${OpenCV_LIBS})
