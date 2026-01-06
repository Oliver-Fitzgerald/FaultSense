# Project
cmake_minimum_required(VERSION 3.10)
project(fault-sense-test)

# Set policies
cmake_policy(SET CMP0072 NEW)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)

# Options

# Packages
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)
FetchContent_MakeAvailable(Catch2)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Compilation
add_executable(generic-utils-test ../feature/utils/generic-utils-test.cpp)
target_link_libraries(generic-utils-test PRIVATE Catch2::Catch2WithMain ${OpenCV_LIBS} stdc++fs)
