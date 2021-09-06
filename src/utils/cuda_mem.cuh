#ifndef UTILS_CUDA_MEM_CUH
#define UTILS_CUDA_MEM_CUH

/**
 * @file cuda_mem.cuh
 * @author Jiannan Tian
 * @brief CUDA memory operation wrappers.
 * @version 0.2
 * @date 2020-09-20
 * Created on 2020-04-30
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace mem {

enum MemcpyDirection { h2d, d2h };

template <typename T>
inline T* create_CUDA_space(size_t len, uint8_t filling_val = 0x00)
{
    T* d_var;
    cudaMalloc(&d_var, len * sizeof(T));
    cudaMemset(d_var, filling_val, len * sizeof(T));
    return d_var;
}

template <typename T>
inline T* create_devspace_memcpy_h2d(T* var, size_t l)
{
    T* d_var;
    cudaMalloc(&d_var, l * sizeof(T));
    cudaMemcpy(d_var, var, l * sizeof(T), cudaMemcpyHostToDevice);
    return d_var;
}
template <typename T>
inline T* create_devspace_memcpy_d2h(T* d_var, size_t l)
{
    // auto var = new T[l];
    T* var;
    cudaMallocHost(&var, l * sizeof(T));
    cudaMemcpy(var, d_var, l * sizeof(T), cudaMemcpyDeviceToHost);
    return var;
}

}  // namespace mem

#endif
