/**
 * @file compaction_g.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef F712F74C_7488_4445_83EE_EE7F88A64BBA
#define F712F74C_7488_4445_83EE_EE7F88A64BBA

#include <cuda_runtime.h>
#include <cstring>
#include "compaction.hh"

#include <stdint.h>
#include <stdlib.h>

// TODO filename -> `compaction`
template <typename T>
struct CompactionDRAM {
    using type = T;
    T*        val;
    uint32_t* idx;
    uint32_t* count;
    uint32_t* h_count;

    void allocate(size_t len, bool device = true)
    {
        if (device) {
            cudaMalloc(&idx, sizeof(uint32_t) * len);
            cudaMalloc(&val, sizeof(T) * len);
            cudaMalloc(&count, sizeof(T) * 1);
            cudaMallocHost(&h_count, sizeof(T) * 1);
        }
        else {
            cudaMallocHost(&idx, sizeof(uint32_t) * len);
            cudaMallocHost(&val, sizeof(T) * len);
            cudaMallocHost(&count, sizeof(T) * 1);

            memset(count, 0x0, sizeof(T) * 1);
        }
    }

    void make_count_host_accessible(cudaStream_t stream)
    {
        cudaMemcpyAsync(h_count, count, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    }

    uint32_t access_count_on_host() { return *h_count; }

    void allocate_managed(size_t len)
    {
        cudaMallocManaged(&idx, sizeof(uint32_t) * len);
        cudaMallocManaged(&val, sizeof(T) * len);
        cudaMallocManaged(&count, sizeof(T) * 1);

        cudaMemset(count, 0x0, sizeof(T) * 1);
    }

    void destroy()
    {
        if (h_count) cudaFreeHost(h_count);
        cudaFree(idx);
        cudaFree(val);
        cudaFree(count);
    }
};

#endif /* F712F74C_7488_4445_83EE_EE7F88A64BBA */
