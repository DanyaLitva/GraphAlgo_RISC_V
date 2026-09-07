cmake_minimum_required(VERSION 3.12)

project(graphAlgorithms LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

option(WITH_RVV "Enable RISC-V Vector intrinsics" OFF)
set(MCA_LMUL "1" CACHE STRING "RVV LMUL value (1, 2, 4)")

if(WITH_RVV AND NOT MCA_LMUL MATCHES "^(1|2|4)$")
    message(FATAL_ERROR "MCA_LMUL must be 1, 2, or 4")
endif()

add_library(grAlgoLib STATIC
    include/matrix.h
    include/matrix_la.h
    include/graph_algorithms.h
    include/matrix_utils.h
    include/mmio.h
    src/mmio.cpp
    src/graph_algorithms.cpp
    src/floatMult.cpp
)

target_include_directories(grAlgoLib PUBLIC include)

target_link_libraries(grAlgoLib PUBLIC OpenMP::OpenMP_CXX)


#-DMCA_LMUL=1,2,4
if(WITH_RVV)
    target_compile_definitions(grAlgoLib PUBLIC USE_RVV)
    if(MCA_LMUL EQUAL 1)
        target_compile_definitions(grAlgoLib PUBLIC MCA_LMUL1)
    elseif(MCA_LMUL EQUAL 2)
        target_compile_definitions(grAlgoLib PUBLIC MCA_LMUL2)
    elseif(MCA_LMUL EQUAL 4)
        target_compile_definitions(grAlgoLib PUBLIC MCA_LMUL4)
    endif()
endif()

add_executable(grAlgo
    src/main.cpp
)

add_executable(floatMult
    src/floatMult.cpp
)

add_executable(k_truss_test
    src/k_truss_test.cpp
)

add_executable(triangle_test
    src/triangle_test.cpp
)

add_executable(mxm_test
    src/mxm_test.cpp
)

target_link_libraries(grAlgo PRIVATE grAlgoLib)
target_link_libraries(floatMult PRIVATE grAlgoLib)
target_link_libraries(k_truss_test PRIVATE grAlgoLib)
target_link_libraries(triangle_test PRIVATE grAlgoLib)
target_link_libraries(mxm_test PRIVATE grAlgoLib)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "RVV: ${WITH_RVV}, MCA_LMUL: ${MCA_LMUL}")