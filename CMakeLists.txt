cmake_minimum_required(VERSION 3.12)

project(graphAlgorithms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

option(WITH_RVV "Enable RISC-V Vector intrinsics" OFF)

add_library(grAlgoLib STATIC
    include/matrix.h
    include/matrix_la.h
    include/graph_algorithms.h
    include/matrix_utils.h
    include/mmio.h
    src/mmio.cpp
    src/graph_algorithms.cpp
)

target_include_directories(grAlgoLib PUBLIC include)

target_link_libraries(grAlgoLib PUBLIC OpenMP::OpenMP_CXX)

if(WITH_RVV)
    target_compile_definitions(grAlgoLib PUBLIC USE_RVV)
endif()

add_executable(grAlgo
    src/main.cpp
)

target_link_libraries(grAlgo PRIVATE grAlgoLib)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")