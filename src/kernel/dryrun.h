/**
 * @file cusz_dryrun.h
 * @author Jiannan Tian
 * @brief cuSZ dryrun mode, checking data quality from lossy compression.
 * @version 0.3
 * @date 2020-09-20
 * (create) 2020-05-14, (release) 2020-09-20, (rev1) 2021-01-25, (rev2) 2021-06-21
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef KERNEL_DRYRUN_H
#define KERNEL_DRYRUN_H

#define TIX threadIdx.x
#define TIY threadIdx.y
#define TIZ threadIdx.z
#define BIX blockIdx.x
#define BIY blockIdx.y
#define BIZ blockIdx.z
#define BDX blockDim.x
#define BDY blockDim.y
#define BDZ blockDim.z

#if CUDART_VERSION >= 11000
// #pragma message __FILE__ ": (CUDA 11 onward), cub from system path"
#include <cub/cub.cuh>
#else
// #pragma message __FILE__ ": (CUDA 10 or earlier), cub from git submodule"
#include "../../external/cub/cub/cub.cuh"
#endif

namespace cusz {

template <typename Data = float, typename FP = float, int BLOCK = 256, int SEQ = 4>
__global__ void dual_quant_dryrun(Data* data, size_t len, FP ebx2_r, FP ebx2)
{
    constexpr auto  NTHREAD = BLOCK / SEQ;
    __shared__ Data shmem[BLOCK];
    auto            id_base = BIX * BLOCK;
    // Data            thread_scope[SEQ];

#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
        auto id = id_base + TIX + i * NTHREAD;
        if (id < len) {
            shmem[TIX + i * NTHREAD] = round(data[id] * ebx2_r) * ebx2;
            data[id]                 = shmem[TIX + i * NTHREAD];
        }
    }

    /* EOF */
}

}  // namespace cusz

#endif