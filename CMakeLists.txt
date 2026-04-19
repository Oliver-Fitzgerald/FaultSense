# Project
cmake_minimum_required(VERSION 4.1.1)
project(FaultSense)

# Set policies
cmake_policy(SET CMP0072 NEW)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
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

FetchContent_Declare(
    simpleini
    GIT_REPOSITORY https://github.com/brofield/simpleini
    GIT_TAG        v4.25
)
FetchContent_MakeAvailable(simpleini)

# Compilation
add_executable(fault-sense-cli ../src/main/frontend/cli/fault-sense-cli.cpp
                           ../src/main/frontend/cli/image-viewer/image-viewer.cpp
                           ../src/main/frontend/cli/evaluation/evaluation-interface.cpp
                           ../src/main/frontend/cli/training/training-interface.cpp
                           ../src/main/feature/feature-extraction.cpp
                           ../src/main/pre-processing/object-detection.cpp
                           ../src/main/pre-processing/pre-processing.cpp
                           ../src/main/pre-processing/utils/pre-processing-utils.cpp
                           ../src/main/general/generic-utils.cpp
                           ../src/main/training/train.cpp
                           ../src/main/general/file-operations/feature-file-operations.cpp
                           ../src/main/general/file-operations/image-file-operations.cpp
                           ../src/main/evaluation/evaluation.cpp
                           ../src/main/evaluation/evaluation-utils.cpp)
target_include_directories(fault-sense-cli PRIVATE ${CMAKE_BINARY_DIR}/_deps/simpleini-src)
target_link_libraries(fault-sense-cli PRIVATE ${OpenCV_LIBS} CLI11::CLI11)

add_executable(data-managment-cli ../src/main/frontend/cli/data-managment-cli.cpp
                           ../src/main/general/file-operations/image-file-operations.cpp
                           ../src/main/general/generic-utils.cpp
                           ../src/main/data-preperation/generate-masks.cpp
                           ../src/main/data-preperation/synthetic-features-data-generation.cpp
                           ../src/main/data-preperation/synthetic-pre-processing-data-generation.cpp)
target_link_libraries(data-managment-cli PRIVATE  ${OpenCV_LIBS} CLI11::CLI11)
