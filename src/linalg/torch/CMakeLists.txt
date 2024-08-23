cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cpp_end)

set(CMAKE_PREFIX_PATH "E:\\libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(cpp_end ./cpp/cpp_end.cpp)
target_link_libraries(cpp_end "${TORCH_LIBRARIES}")
set_property(TARGET cpp_end PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET cpp_end
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:cpp_end>)
endif (MSVC)