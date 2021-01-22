/**
 * @file cuda_wrap.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-01-20
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUDA_WRAP_HH

/**
 * @brief CUDA kernel config; pass by non-pointer value
 *
 */
typedef struct CUDAKernelConfig {
    dim3         Dg;  // dimension of grid
    dim3         Db;  // dimension of block
    size_t       Ns;  // per-block shmem bytes
    cudaStream_t S;   // stream

} kernel_cfg_t;

#endif