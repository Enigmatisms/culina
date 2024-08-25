cmake_minimum_required(VERSION 3.11.0)
project(Task)
set(CMAKE_BUILD_TYPE "release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -g -O0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g -O0")
ELSE()
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
ENDIF()

include_directories(.)

add_executable(tp main.cc
    thread_pool.cc
)
add_executable(new_tp new_thread_pool.cc)

add_executable(bf double_buffer.cc)