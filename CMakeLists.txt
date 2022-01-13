cmake_minimum_required(VERSION 3.18)

project(cusz_asap CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

## This is a feature as of cmake 3.18.
## check `cmake --help-policy CMP0104` for more detail.
## The maximum possible in compatibility can be set in CMakeLists.txt using
## `set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 80 86)`.
## For command line, use, for example `cmake -DCMAKE_CUDA_ARCHITECTURES="75" ..` 
## to specify CUDA arch.

enable_language(CUDA)

#include_directories(src)
#include_directories(src/pSZ)

## TODO flag only add to a specific library, e.g. suppressing deprecation on CUDA10 cuSPARSE
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations")

set(LIB_TYPE STATIC)

add_library(pq ${LIB_TYPE} src/wrapper/extrap_lorenzo.cu src/wrapper/interp_spline3.cu)
add_library(argp ${LIB_TYPE} src/context.cc)
add_library(sp ${LIB_TYPE} src/wrapper/csr11.cu src/wrapper/spgs.cu)
target_link_libraries(sp -lcusparse)
add_library(huff ${LIB_TYPE} src/wrapper/huffman_parbook.cu src/wrapper/huffman_coarse.cu)
set_target_properties(huff PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_library(nvgpusz ${LIB_TYPE} src/default_path.cu src/base_compressor.cu src/sp_path.cu)

add_executable(cusz src/cusz.cu)
target_link_libraries(cusz nvgpusz argp huff sp pq)
