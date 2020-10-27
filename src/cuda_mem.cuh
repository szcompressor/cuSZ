#ifndef CUDA_MEM_CUH
#define CUDA_MEM_CUH

/**
 * @file cuda_mem.cu
 * @author Jiannan Tian
 * @brief CUDA memory operation wrappers.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-04-30
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace mem {

enum MemcpyDirection { h2d, d2h };

template <typename T>
T* CreateCUDASpace(size_t l, uint8_t i = 0);

template <typename T>
void CopyBetweenSpaces(T* src, T* dst, MemcpyDirection direct);

template <typename T>
T* CreateDeviceSpaceAndMemcpyFromHost(T* var, size_t l);

template <typename T>
T* CreateHostSpaceAndMemcpyFromDevice(T* d_var, size_t l);
}  // namespace mem

#endif
