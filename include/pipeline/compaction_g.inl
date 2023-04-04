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
struct CompactCudaDram {
   private:
    static const cudaMemcpyKind h2d = cudaMemcpyHostToDevice;
    static const cudaMemcpyKind d2h = cudaMemcpyDeviceToHost;

   public:
    using type = T;

    // `h_` for host-accessible
    T *       val, *h_val;
    uint32_t *idx, *h_idx;
    uint32_t *count, *h_count;
    size_t    reserved_len;

    // CompactCudaDram() {}
    // ~CompactCudaDram() {}

    CompactCudaDram& set_reserved_len(size_t _reserved_len)
    {
        reserved_len = _reserved_len;
        return *this;
    }

    CompactCudaDram& malloc()
    {
        cudaMalloc(&val, sizeof(T) * reserved_len);
        cudaMalloc(&idx, sizeof(uint32_t) * reserved_len);
        cudaMalloc(&count, sizeof(uint32_t) * 1);
        // init value
        cudaMemset(count, 0x0, sizeof(T) * 1);

        return *this;
    }

    CompactCudaDram& mallochost()
    {
        cudaMallocHost(&h_val, sizeof(T) * reserved_len);
        cudaMallocHost(&h_idx, sizeof(uint32_t) * reserved_len);
        cudaMallocHost(&h_count, sizeof(uint32_t) * 1);
        // init value
        *h_count = 0;

        return *this;
    }

    CompactCudaDram& free()
    {
        cudaFree(idx), cudaFree(val), cudaFree(count);
        return *this;
    }

    CompactCudaDram& freehost()
    {
        cudaFreeHost(h_idx), cudaFreeHost(h_val), cudaFreeHost(h_count);
        return *this;
    }

    // memcpy
    CompactCudaDram& make_host_accessible(cudaStream_t stream = 0)
    {
        cudaMemcpyAsync(h_count, count, sizeof(uint32_t), d2h, stream);
        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(h_val, val, sizeof(T) * (*h_count), d2h, stream);
        cudaMemcpyAsync(h_idx, idx, sizeof(uint32_t) * (*h_count), d2h, stream);

        return *this;
    }

    // accessor
    uint32_t num_outliers() const { return *h_count; }
};

#endif /* F712F74C_7488_4445_83EE_EE7F88A64BBA */
