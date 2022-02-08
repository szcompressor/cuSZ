cmake_minimum_required(VERSION 3.18)

project(cusz CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(BUILD_SHARED_LIBS OFF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

## This is a feature as of cmake 3.18.
## check `cmake --help-policy CMP0104` for more detail.
## The maximum possible in compatibility can be set in CMakeLists.txt using
## `set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86)`.
## For command line, use, for example `cmake -DCMAKE_CUDA_ARCHITECTURES="75" ..` 
## to specify CUDA arch.

#include_directories(src)
#include_directories(src/pSZ)

## TODO flag only add to a specific library, e.g. suppressing deprecation on CUDA10 cuSPARSE
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations")

add_library(pq src/wrapper/extrap_lorenzo.cu)
add_library(argp src/context.cc)
add_library(sp src/wrapper/csr11.cu src/wrapper/spgs.cu)
target_link_libraries(sp -lcusparse)
add_library(huff src/wrapper/huffman_parbook.cu src/wrapper/huffman_coarse.cu)
set_target_properties(huff PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_library(nvgpusz src/default_path.cu src/base_cusz.cu)

add_executable(cusz src/cusz.cu)
target_link_libraries(cusz nvgpusz argp huff sp pq)


include(GNUInstallDirs)

install(TARGETS pq argp sp huff nvgpusz cusz EXPORT cusz)
export(EXPORT cusz NAMESPACE CUSZ::
  FILE ${CMAKE_CURRENT_BINARY_DIR}/cuSZ-targets.cmake)
install(EXPORT cusz
  FILE cuSZ-targets.cmake
  NAMESPACE CUSZ::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuSZ)
