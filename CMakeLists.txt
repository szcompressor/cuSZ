cmake_minimum_required(VERSION 3.16)

project(cusz_asap CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

enable_language(CUDA)

include_directories(src)
include_directories(src/pSZ)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

set(LIB_TYPE STATIC)

add_library(parhuffman ${LIB_TYPE}
        src/par_huffman_sortbyfreq.cu
        src/par_merge.cuh
        src/par_merge.cu
        src/par_huffman.cuh
        src/par_huffman.cu)
set_property(TARGET parhuffman PROPERTY CUDA_ARCHITECTURES 75)
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

## to link with parhuffman
add_library(gpuhuffman ${LIB_TYPE}
        src/hist.cu
        src/hist.cuh
        src/huff_codec.cu
        src/huff_codec.cuh
        src/huff_interface.cu
        src/huff_interface.cuh
        src/hist.cu
        src/hist.cuh
        )

#add_library(utils ${LIB_TYPE}
#        src/analysis.hh
#        src/datapack.hh
#        src/utils/verify.hh
#        src/utils/format.hh
#        )


add_executable(cusz_ood
        src/ood/compressor.hh
        src/ood/sparse_op.hh
        src/lorenzo_trait.cuh
        src/snippets.cc
        src/cusz.cu
        src/filter.cu
        src/filter.cuh
        src/argparse.cc
        src/argparse.hh
        src/dryrun.cu
        src/dryrun.cuh
        src/dualquant.cu
        src/dualquant.cuh
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
        test/test_sparse_op.cu)

add_executable(cusz_cmake
        src/lorenzo_trait.cuh
        src/analysis/analyzer.hh
        src/snippets.cc
        src/cusz.cu
        src/filter.cu
        src/filter.cuh
        src/argparse.cc
        src/argparse.hh
        src/dryrun.cu
        src/dryrun.cuh
        src/dualquant.cu
        src/dualquant.cuh
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

set_property(TARGET cusz_cmake PROPERTY CUDA_ARCHITECTURES 70 75)
set_target_properties(cusz_cmake
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cusz_cmake
        -lcusparse
        utils
        parhuffman
        gpuhuffman
        )

