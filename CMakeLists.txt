cmake_minimum_required(VERSION 3.16)

project(cusz_cmake CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

enable_language(CUDA)

include_directories(src)
include_directories(src/pSZ)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

set(LIB_TYPE STATIC)

add_library(parhuffbook ${LIB_TYPE}
        src/par_huffman_sortbyfreq.cu
        src/par_merge.cuh
        src/par_merge.cu
        src/par_huffman.cuh
        src/par_huffman.cu)
set_property(TARGET parhuffbook PROPERTY CUDA_ARCHITECTURES 75)
set_target_properties(parhuffbook PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(psz
        src/format.hh
        src/format.cc
        src/constants.hh
        src/constants.cc
        src/SDRB.hh
        src/SDRB.cc
        src/verify.hh
        src/verify.cc
        src/types.hh
        src/types.cc
        src/io.hh
        src/analysis_utils.hh
        src/analysis_utils.cc
        src/pSZ/psz_14.hh
        src/pSZ/psz_14blocked.hh
        src/pSZ/psz_dualquant.hh
        src/pSZ/psz_exe.cc
        src/pSZ/psz_workflow.hh
        )

## to link with parhuffman
add_library(gpuhuffman ${LIB_TYPE}
        src/dbg_gpu_printing.cuh
#        src/gather_scatter.cu
#        src/gather_scatter.cuh
        src/histogram.cu
        src/histogram.cuh
        src/huffman.cu
        src/huffman.cuh
        src/huffman_codec.cu
        src/huffman_codec.cuh
        src/huffman_workflow.cu
        src/huffman_workflow.cuh
        src/histogram.cu
        src/histogram.cuh
        )
add_library(utils ${LIB_TYPE}
        src/analysis.hh
        src/analysis_utils.cc
        src/analysis_utils.hh
        src/cuda_error_handling.cuh
        src/cuda_mem.cu
        src/cuda_mem.cuh
        src/verify.cc
        src/verify.hh
        src/autotune.cu
        src/autotune.h
        src/format.cc
        src/format.hh
        )


add_library(filter
        src/filter.cu
        src/filter.cuh
        )

add_executable(cusz_cmake
        src/cusz.cu
        src/argparse.cc
        src/argparse.hh
        src/constants.cc
        src/constants.hh
        src/cusz_dryrun.cu
        src/cusz_dryrun.cuh
        src/cusz_dualquant.cu
        src/cusz_dualquant.cuh
        src/cusz_workflow.cu
        src/cusz_workflow.cuh
        src/gather_scatter.cu
        src/gather_scatter.cuh
        src/io.hh
        src/pack.hh
        src/query.hh
        src/query_dev.hh
        src/SDRB.cc
        src/SDRB.hh
        src/timer.hh
        src/types.cc
        src/types.hh

        )

set_property(TARGET cusz_cmake PROPERTY CUDA_ARCHITECTURES 70 75)
set_target_properties(cusz_cmake
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

message(STATUS "cusparse: ${CUDA_cusparse_LIBRARY}")

target_link_libraries(cusz_cmake
        -lcusparse
        filter
        utils
        parhuffbook
        gpuhuffman
        )

