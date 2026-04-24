# Project
cmake_minimum_required(VERSION 4.1.1)
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
add_executable(fault-sense-cli ../src/main/frontend/cli/fault-sense-cli.cpp
                           ../src/main/feature/pre-processing.cpp
                           ../src/main/feature/object-detection.cpp
                           ../src/main/feature/pre-processing.cpp
                           ../src/main/feature/feature-extraction.cpp
                           ../src/main/feature/utils/generic-utils.cpp
                           ../src/main/feature/utils/pre-processing-utils.cpp
                           ../src/main/training/train.cpp
                           ../src/main/general/file-operations/training-data.cpp
                           ../src/main/evaluation/evaluation.cpp)
target_link_libraries(fault-sense-cli PRIVATE  ${OpenCV_LIBS} CLI11::CLI11)

add_executable(data-managment-cli ../src/main/frontend/cli/data-managment-cli.cpp
                           ../src/main/general/file-operations/generic-read-write.cpp
                           ../src/main/data-preperation/generate-masks.cpp
                           ../src/main/data-preperation/synthetic-data-generation.cpp)
target_link_libraries(data-managment-cli PRIVATE  ${OpenCV_LIBS} CLI11::CLI11)
