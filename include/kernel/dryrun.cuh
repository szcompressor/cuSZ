/**
 * @file dryrun.cuh
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

#ifndef CUSZ_KERNEL_DRYRUN_CUH
#define CUSZ_KERNEL_DRYRUN_CUH

namespace cusz {

template <typename Data = float, typename FP = float, int BLOCK = 256, int SEQ = 4>
// template <typename Data = float, typename FP = float>
__global__ void dualquant_dryrun_kernel(Data* in_data, Data* out_xdata, size_t len, FP ebx2_r, FP ebx2)
{
    {
        constexpr auto  NTHREAD = BLOCK / SEQ;
        __shared__ Data shmem[BLOCK];
        auto            id_base = blockIdx.x * BLOCK;

#pragma unroll
        for (auto i = 0; i < SEQ; i++) {
            auto id = id_base + threadIdx.x + i * NTHREAD;
            if (id < len) {
                shmem[threadIdx.x + i * NTHREAD] = round(in_data[id] * ebx2_r) * ebx2;
                out_xdata[id]                    = shmem[threadIdx.x + i * NTHREAD];
            }
        }
    }

    // simplistic
    // {
    //     auto id = blockIdx.x * blockDim.x + threadIdx.x;
    //     if (id < len) out_xdata[id] = round(in_data[id] * ebx2_r) * ebx2;
    // }
}

}  // namespace cusz

#endif