cmake_minimum_required(VERSION 3.16)

project(cusz_asap CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75)

enable_language(CUDA)

include_directories(src)
include_directories(src/pSZ)

## TODO flag only add to a specific library, e.g. suppressing deprecation on CUDA10 cuSPARSE
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-declarations")

set(LIB_TYPE STATIC)

## new for 0.3
add_library(newlossless ${LIB_TYPE}
        src/wrapper/lossless_huffman.cu
        src/wrapper/handle_sparsity.cu
        )
set_target_properties(newlossless PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(newpredictor ${LIB_TYPE}
        src/wrapper/extrap_lorenzo.cu
        # src/wrapper/interp_spline.cu
        )

add_library(argparse ${LIB_TYPE}
        src/argparse.cc
        src/argparse.hh
        )

## to link with parhuffman
add_library(oldlossless ${LIB_TYPE}
        src/wrapper/deprecated_lossless_huffman.cu
        src/wrapper/deprecated_sparsity.cu
        src/wrapper/par_huffman.cu
        )
target_link_libraries(oldlossless
        -lcusparse
        )
set_target_properties(oldlossless PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(cusz_cmake
        src/cusz.cu
        src/cusz_interface.cu
        )

set_target_properties(cusz_cmake
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cusz_cmake
        -lcusparse
        argparse
        oldlossless
        )
