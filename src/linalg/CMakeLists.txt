cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
# CUDA_ARCHITECTURES native is available in CMake since 3.24

set(PROJ_NAME "culina")
project(${PROJ_NAME} LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_TOOLCHAIN_FILE "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\VC\\vcpkg\\scripts\\buildsystems\\vcpkg.cmake")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
set(VCPKG_TARGET_TRIPLET "x64-windows")

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall /Od /Zi / /RTC1")
  message(STATUS "Building project in DEBUG mode.")
  set(CMAKE_BUILD_TYPE Debug)
ELSE()
  # to check register usage, please set "-Xptxas -v"
  # to cap the register usage to boost occupancy, set ""
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /bigobj")
  message(STATUS "Building project in RELEASE mode.")
  set(CMAKE_BUILD_TYPE Release)
ENDIF()

list(APPEND CMAKE_MODULE_PATH "C:/Program Files/OpenVDB/lib/cmake/OpenVDB/")

find_package(Boost REQUIRED COMPONENTS iostreams regex)
find_package(TBB REQUIRED)
find_package(Blosc REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
find_package(nlohmann_json 3.2.0 REQUIRED)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP is found.")
else()
    message(FATAL_ERROR "OpenMP can not be found but it is required by mesh2vol.cpp. Manual overriding is required to skip this exception.")
endif()

set(OpenVDB_INCLUDE_DIR "C:/Program Files/OpenVDB/include/openvdb")
find_package(OpenVDB
    REQUIRED             # Fail with error if OpenVDB is not found
    COMPONENTS openvdb nanovdb
)

# For SGGX splatting viewer
find_package(glad CONFIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

include_directories(
    "C:/Program Files/OpenVDB/include"
    ${EIGEN3_INCLUDE_DIRS}
)

message(STATUS "Qianyue He: ${EIGEN3_INCLUDE_DIRS}")

add_executable(vol2cpp vol2cpp.cpp)
add_executable(grid_extract grid_extract.cpp)
add_executable(synthesis grid_synthesis.cpp)
add_executable(fast_test fast_test.cpp)
add_executable(mesh2vol mesh2vol.cpp)

add_executable(splat_viewer splat_viewer.cpp)
# add_executable(color_viewer color_viewer.cpp)
target_include_directories(splat_viewer PRIVATE ${OPENGL_INCLUDE_DIR})
# target_include_directories(color_viewer PRIVATE ${OPENGL_INCLUDE_DIR})

target_link_libraries(vol2cpp OpenVDB::openvdb OpenVDB::nanovdb)
target_link_libraries(grid_extract OpenVDB::openvdb OpenVDB::nanovdb)
target_link_libraries(synthesis OpenVDB::openvdb OpenVDB::nanovdb)
target_link_libraries(mesh2vol OpenVDB::openvdb OpenVDB::nanovdb nlohmann_json::nlohmann_json)

target_link_libraries(splat_viewer PRIVATE ${OPENGL_LIBRARIES} glfw glad::glad nlohmann_json::nlohmann_json)
# target_link_libraries(color_viewer PRIVATE ${OPENGL_LIBRARIES} glfw glad::glad)