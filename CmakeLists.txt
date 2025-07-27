cmake_minimum_required(VERSION 3.12...3.18)

project(graphAlgorithms LANGUAGES CXX)

if(UNIX)
    set(CMAKE_CXX_FLAGS "-std=c++17 -fopenmp")
endif()

set(SOURCE_EXE src/main.cpp)
add_library(grAlgoLib STATIC
            include/matrix.h
            include/matrix_la.h
            include/graph_algorithms.h
            include/matrix_utils.h
            include/mmio.h
            src/mmio.cpp
            src/graph_algorithms.cpp)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g)
    link_libraries(-g)
endif()
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
    link_libraries(-O3)
endif()

target_include_directories(grAlgoLib PUBLIC include)

add_executable(grAlgo ${SOURCE_EXE})

target_link_libraries(grAlgo PUBLIC grAlgoLib)

message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
