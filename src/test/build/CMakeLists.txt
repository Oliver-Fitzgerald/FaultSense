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

add_executable(train-test ../training/train-test.cpp
                          ../../main/general/generic-utils.cpp
                          ../../main/feature/feature-extraction.cpp
                          ../../main/feature/utils/pre-processing-utils.cpp
                          ../../main/feature/object-detection.cpp
                          ../../main/general/file-operations/generic-file-operations.cpp
                          ../../main/training/train.cpp)
target_link_libraries(train-test PRIVATE Catch2::Catch2WithMain ${OpenCV_LIBS} stdc++fs)

add_executable(training-data-tests ../general/file-operations/training-data-tests.cpp
                          ../../main/general/file-operations/training-data.cpp)
target_link_libraries(training-data-tests PRIVATE Catch2::Catch2WithMain ${OpenCV_LIBS} stdc++fs)

add_executable(ground-truth-tests ../general/file-operations/ground-truth-tests.cpp
                          ../../main/general/file-operations/ground-truth.cpp)
target_link_libraries(ground-truth-tests PRIVATE Catch2::Catch2WithMain ${OpenCV_LIBS} stdc++fs)

add_executable(evaluation-tests ../evaluation/evaluation-tests.cpp
                          ../../main/training/train.cpp
                          ../../main/general/generic-utils.cpp
                          ../../main/feature/utils/pre-processing-utils.cpp
                          ../../main/general/file-operations/generic-file-operations.cpp
                          ../../main/feature/pre-processing.cpp
                          ../../main/feature/feature-extraction.cpp
                          ../../main/feature/object-detection.cpp
                          ../../main/feature/utils/pre-processing-utils.cpp
                          ../../main/feature/object-detection.cpp
                          ../../main/evaluation/evaluation.cpp)
target_link_libraries(evaluation-tests PRIVATE Catch2::Catch2WithMain ${OpenCV_LIBS} stdc++fs)

add_executable(pre-processing-utils-tests ../feature/utils/pre-processing-utils-tests.cpp
                          ../../main/frontend/cli/image-viewer-ui/image-viewer.cpp
                          ../../main/general/generic-utils.cpp
                          ../../main/feature/utils/pre-processing-utils.cpp)
target_link_libraries(pre-processing-utils-tests PRIVATE Catch2::Catch2WithMain ${OpenCV_LIBS} stdc++fs)
