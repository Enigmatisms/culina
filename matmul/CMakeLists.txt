cmake_minimum_required(VERSION 3.10)

project(SGEMM CUDA CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O0")
ELSE()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
ENDIF()
SET(COMPILE_CUDA True)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
  set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
  message(STATUS ${OpenMP_CXX_FLAGS})
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/../
)

add_executable(matmul ./matmul.cu)
target_link_libraries(matmul pthread OpenMP::OpenMP_CXX)

add_executable(vecmat ./matvec.cu)
target_link_libraries(vecmat pthread OpenMP::OpenMP_CXX)

add_executable(dotprod ./dot_product.cu)
target_link_libraries(dotprod pthread OpenMP::OpenMP_CXX)

add_executable(trans ./transpose.cu)
target_link_libraries(trans pthread OpenMP::OpenMP_CXX)
