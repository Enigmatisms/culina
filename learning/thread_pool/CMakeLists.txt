cmake_minimum_required(VERSION 3.11.0)
project(Task)
set(CMAKE_BUILD_TYPE "release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories(.)

add_executable(tp main.cc
    thread_pool.cc
)

target_link_libraries(tp pthread)