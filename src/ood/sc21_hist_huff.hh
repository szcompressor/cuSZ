/**
 * @file draft_hist_wrapper.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-04-05
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef DRAFT_HIST_WRAPPER
#define DRAFT_HIST_WRAPPER

#define nworker blockDim.x

#include <cstdlib>
#include "../kernel/hist.cuh"

namespace sc21 {

template <typename Input, bool CutOff = true>
void GetFrequency(Input* d_in, size_t len, unsigned int* d_freq, int dict_size)
{
    // Parameters for thread and block count optimization
    // Initialize to device-specific values
    int deviceId, max_bytes, max_bytes_opt_in, num_SMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&max_bytes, cudaDevAttrMaxSharedMemoryPerBlock, deviceId);
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, deviceId);

    // Account for opt-in extra shared memory on certain architectures
    cudaDeviceGetAttribute(&max_bytes_opt_in, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId);
    max_bytes = std::max(max_bytes, max_bytes_opt_in);

    // Optimize launch
    int num_buckets      = dict_size;
    int num_values       = len;
    int items_per_thread = 1;
    int r_per_block      = (max_bytes / (int)sizeof(int)) / (num_buckets + 1);
    int num_blocks       = num_SMs;
    // fits to size
    int threads_per_block = ((((num_values / (num_blocks * items_per_thread)) + 1) / 64) + 1) * 64;
    while (threads_per_block > 1024) {
        if (r_per_block <= 1) { threads_per_block = 1024; }
        else {
            r_per_block /= 2;
            num_blocks *= 2;
            threads_per_block = ((((num_values / (num_blocks * items_per_thread)) + 1) / 64) + 1) * 64;
        }
    }

    cudaFuncSetAttribute(
        Histogram<Input, unsigned int, CutOff>,  //
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_bytes);

    Histogram<Input, unsigned int, CutOff>                                                    //
        <<<num_blocks, threads_per_block, ((num_buckets + 1) * r_per_block) * sizeof(int)>>>  //
        (d_in, d_freq, num_values, num_buckets, r_per_block);
    cudaDeviceSynchronize();
}

// temporary draft::CopyHuffmanUintsDenseToSparse
template <typename Huff>
__global__ void
GatherHuffmanChunks(Huff* input_dn, Huff* output_sp, size_t* sp_entries, size_t* sp_uints, size_t dn_chunk_size)
{
    auto len      = sp_uints[blockIdx.x];
    auto sp_entry = sp_entries[blockIdx.x];
    auto dn_entry = dn_chunk_size * blockIdx.x;

    for (auto i = 0; i < (len + nworker - 1) / nworker; i++) {
        auto _tid = threadIdx.x + i * nworker;
        if (_tid < len) *(output_sp + sp_entry + _tid) = *(input_dn + dn_entry + _tid);
        __syncthreads();
    }
}

}  // namespace sc21

#endif