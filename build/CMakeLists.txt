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

# Set OpenGL preference to GLVND
set(OpenGL_GL_PREFERENCE GLVND)

# Packages
find_package(OpenCV REQUIRED)
find_package(FLTK REQUIRED)
find_package(Fontconfig REQUIRED)
find_package(X11 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} ${FLTK_INCLUDE_DIR} )

# Compilation
add_executable(gui ../src/frontend/main-menu.cc)
target_link_libraries(gui 
    ${FLTK_LIBRARIES} 
    ${OpenCV_LIBS}
    Fontconfig::Fontconfig
    ${X11_LIBRARIES}
    ${X11_Xft_LIB}
    ${X11_Xrender_LIB}
    ${X11_Xinerama_LIB}
    ${X11_Xfixes_LIB}
    ${X11_Xcursor_LIB}
)

add_executable(cli ../src/frontend/main.cc)
target_link_libraries(cli ${OpenCV_LIBS})

add_executable(configureHSV ../src/feature/configure-hsv.cc)
target_link_libraries(configureHSV ${OpenCV_LIBS})

add_executable(configureEdge ../src/feature/configure-edge.cc)
target_link_libraries(configureEdge ${OpenCV_LIBS})

add_executable(gabor-filter ../src/feature/gabor-filter.cc)
target_link_libraries(gabor-filter ${OpenCV_LIBS})

# Optional object-detection target
if(BUILD_OBJECT_DETECTION)
    add_executable(object-detection ../src/feature/object-detection.cc ../src/feature/features.cc)
    target_link_libraries(object-detection ${OpenCV_LIBS})
    target_include_directories(object-detection PRIVATE ${OpenCV_INCLUDE_DIRS})
    
    # Add debug symbols
    target_compile_options(object-detection PRIVATE -g)
    
    # Link filesystem library (needed for GCC < 9)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(object-detection stdc++fs)
    endif()
endif()

# Apply filesystem support to all targets that need it
set(TARGETS_NEEDING_FS cli configureHSV configureEdge gabor-filter)
foreach(target ${TARGETS_NEEDING_FS})
    # Link filesystem library for older GCC versions
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(${target} stdc++fs)
    endif()
endforeach()
