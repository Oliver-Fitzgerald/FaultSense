# Project
cmake_minimum_required(VERSION 4.2.1)
project(FaultSense)

# Set policies
cmake_policy(SET CMP0072 NEW)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)

# Packages
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

FetchContent_Declare(
    cli11_proj
    QUIET
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.6.0
)
FetchContent_MakeAvailable(cli11_proj)


# Compilation

add_executable(testing-cli ../cli/testing-cli.cpp
                           ../../feature/pre-processing.cpp
                           ../../feature/object-detection.cpp
                           ../../feature/utils/generic-utils.cpp
                           ../../feature/utils/pre-processing-utils.cpp
                           ../../feature/utils/beta/texture-beta.cpp)
target_link_libraries(testing-cli ${OpenCV_LIBS})

add_executable(fault-sense-cli ../cli/fault-sense-cli.cpp
                           ../../feature/pre-processing.cpp
                           ../../feature/object-detection.cpp
                           ../../feature/utils/generic-utils.cpp
                           ../../feature/utils/pre-processing-utils.cpp
                           ../../feature/utils/beta/texture-beta.cpp)
                       target_link_libraries(fault-sense-cli PRIVATE  ${OpenCV_LIBS} CLI11::CLI11)
