cmake_minimum_required(VERSION 3.16)

project(cusz_asap CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 75)

enable_language(CUDA)

include_directories(src)
include_directories(src/pSZ)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

set(LIB_TYPE STATIC)

## tests
################################################################################
add_executable(test_rle test/test_rle.cu)
add_executable(test_huff test/test_huff_class.cu)
add_executable(test_dropout test/test_dropout.cu)
add_executable(test_lorenzodq test/test_lorenzodq.cu)
################################################################################

## standalone driver programs
################################################################################
add_executable(rle src/standalone/alt_comp_routes.cu)
################################################################################


add_library(parhuffman ${LIB_TYPE}
        src/par_huffman_sortbyfreq.cu
        src/par_merge.cuh
        src/par_merge.cu
        src/par_huffman.cuh
        src/par_huffman.cu)
set_target_properties(parhuffman PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(psz
        src/utils/format.hh
        src/utils/io.hh
        src/utils/verify.hh
        src/types.hh
        src/datapack.hh
        src/pSZ/psz_14.hh
        src/pSZ/psz_14blocked.hh
        src/pSZ/psz_dualquant.hh
        src/pSZ/psz_exe.cc
        src/pSZ/psz_workflow.hh
        )


## new for 0.3
add_library(
        lossless ${LIB_TYPE}
        src/wrapper/lossless_huffman.cu
        src/wrapper/handle_sparsity.cu
)

add_library(
        predictor ${LIB_TYPE}
        src/wrapper/extrap_lorenzo.cu
        # src/wrapper/interp_spline.cu
)


## to link with parhuffman
add_library(gpuhuffman ${LIB_TYPE}
        src/huff_interface.cu
        src/huff_interface.h
        src/kernel/hist.h
        )

add_executable(cusz_cmake
        src/lorenzo_trait.cuh
        src/kernel/rle.cuh
        src/analysis/analyzer.hh
        src/snippets.cc
        src/cusz.cu
        src/filter.cu
        src/filter.cuh
        src/argparse.cc
        src/argparse.hh
        src/dryrun.cu
        src/dryrun.cuh
        src/kernel/lorenzo.h
        src/cusz_interface.cu
        src/cusz_interface.cuh
        src/gather_scatter.cu
        src/gather_scatter.cuh
        src/pack.hh
        src/query.hh
        src/query_dev.hh
        src/utils/io.hh
        src/utils/timer.hh
        src/types.hh
        )

set_target_properties(cusz_cmake
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cusz_cmake
        -lcusparse
        utils
        parhuffman
        gpuhuffman
        )

