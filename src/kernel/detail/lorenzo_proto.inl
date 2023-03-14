/**
 * @file lorenzo_proto.inl
 * @author Jiannan Tian
 * @brief (prototype) Dual-EQ Lorenzo method.
 * @version 0.2
 * @date 2021-01-16
 * (create) 2019-09-23; (release) 2020-09-20; (rev1) 2021-01-16; (rev2) 2021-02-20; (rev3) 2021-04-11
 * (rev4) 2021-04-30
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_KERNEL_LORENZO_PROTOTYPE_CUH
#define CUSZ_KERNEL_LORENZO_PROTOTYPE_CUH

#include <cstddef>
#include <stdexcept>

#include "utils/cuda_err.cuh"
#include "utils/timer.h"

namespace psz {

namespace cuda {
namespace __kernel {

namespace prototype {  // easy algorithmic description

template <typename T, typename EQ = int32_t, typename FP = T, int BLK = 256>
__global__ void c_lorenzo_1d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* eq, T* outlier)
{
    __shared__ T buf[BLK];

    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < len3.x) {
        buf[threadIdx.x] = round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness

    T delta = buf[threadIdx.x] - (threadIdx.x == 0 ? 0 : buf[threadIdx.x - 1]);

    bool quantizable = fabs(delta) < radius;
    T    candidate   = delta + radius;
    if (id < len3.x) {                             // postquant
        data[id] = (1 - quantizable) * candidate;  // output; reuse data for outlier
        eq[id]   = quantizable * static_cast<EQ>(candidate);
    }
}

template <typename T, typename EQ = int32_t, typename FP = T, int BLK = 16>
__global__ void c_lorenzo_2d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* eq, T* outlier)
{
    __shared__ T buf[BLK][BLK + 1];

    auto y = threadIdx.y, x = threadIdx.x;
    auto giy = blockIdx.y * blockDim.y + y, gix = blockIdx.x * blockDim.x + x;

    auto id = gix + giy * stride3.y;  // low to high dim, inner to outer
    if (gix < len3.x and giy < len3.y) {
        buf[y][x] = round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness

    T delta = buf[y][x] - ((x > 0 ? buf[y][x - 1] : 0) +                // dist=1
                           (y > 0 ? buf[y - 1][x] : 0) -                // dist=1
                           (x > 0 and y > 0 ? buf[y - 1][x - 1] : 0));  // dist=2

    bool quantizable = fabs(delta) < radius;
    T    candidate   = delta + radius;
    if (gix < len3.x and giy < len3.y) {
        data[id] = (1 - quantizable) * candidate;  // output; reuse data for outlier
        eq[id]   = quantizable * static_cast<EQ>(candidate);
    }
}

template <typename T, typename EQ, typename FP, int BLK = 8>
__global__ void c_lorenzo_3d1l(T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, EQ* eq, T* outlier)
{
    __shared__ T buf[BLK][BLK][BLK + 1];

    auto z = threadIdx.z, y = threadIdx.y, x = threadIdx.x;
    auto giz = blockIdx.z * blockDim.z + z, giy = blockIdx.y * blockDim.y + y, gix = blockIdx.x * blockDim.x + x;

    auto id = gix + giy * stride3.y + giz * stride3.z;  // low to high in dim, inner to outer
    if (gix < len3.x and giy < len3.y and giz < len3.z) {
        buf[z][y][x] = round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness

    T delta = buf[z][y][x] - ((z > 0 and y > 0 and x > 0 ? buf[z - 1][y - 1][x - 1] : 0)  // dist=3
                              - (y > 0 and x > 0 ? buf[z][y - 1][x - 1] : 0)              // dist=2
                              - (z > 0 and x > 0 ? buf[z - 1][y][x - 1] : 0)              //
                              - (z > 0 and y > 0 ? buf[z - 1][y - 1][x] : 0)              //
                              + (x > 0 ? buf[z][y][x - 1] : 0)                            // dist=1
                              + (y > 0 ? buf[z][y - 1][x] : 0)                            //
                              + (z > 0 ? buf[z - 1][y][x] : 0));                          //

    bool quantizable = fabs(delta) < radius;
    T    candidate   = delta + radius;
    if (gix < len3.x and giy < len3.y and giz < len3.z) {
        data[id] = (1 - quantizable) * candidate;  // output; reuse data for outlier
        eq[id]   = quantizable * static_cast<EQ>(candidate);
    }
}

template <typename T, typename EQ = int32_t, typename FP = T, int BLK = 256>
__global__ void x_lorenzo_1d1l(EQ* eq, T* scattered_outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata)
{
    __shared__ T buf[BLK];

    auto id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < len3.x)
        buf[threadIdx.x] = scattered_outlier[id] + static_cast<T>(eq[id]) - radius;  // fuse
    else
        buf[threadIdx.x] = 0;
    __syncthreads();

    for (auto d = 1; d < BLK; d *= 2) {
        T n = 0;
        if (threadIdx.x >= d) n = buf[threadIdx.x - d];  // like __shfl_up_sync(0x1f, var, d); warp_sync
        __syncthreads();
        if (threadIdx.x >= d) buf[threadIdx.x] += n;
        __syncthreads();
    }

    if (id < len3.x) { xdata[id] = buf[threadIdx.x] * ebx2; }
}

template <typename T, typename EQ = int32_t, typename FP = T, int BLK = 16>
__global__ void x_lorenzo_2d1l(EQ* eq, T* scattered_outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata)
{
    __shared__ T buf[BLK][BLK + 1];

    auto   giy = blockIdx.y * blockDim.y + threadIdx.y, gix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id = gix + giy * stride3.y;

    if (gix < len3.x and giy < len3.y)
        buf[threadIdx.y][threadIdx.x] = scattered_outlier[id] + static_cast<T>(eq[id]) - radius;  // fuse
    else
        buf[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    for (auto d = 1; d < BLK; d *= 2) {
        T n = 0;
        if (threadIdx.x >= d) n = buf[threadIdx.y][threadIdx.x - d];
        __syncthreads();
        if (threadIdx.x >= d) buf[threadIdx.y][threadIdx.x] += n;
        __syncthreads();
    }

    for (auto d = 1; d < BLK; d *= 2) {
        T n = 0;
        if (threadIdx.y >= d) n = buf[threadIdx.y - d][threadIdx.x];
        __syncthreads();
        if (threadIdx.y >= d) buf[threadIdx.y][threadIdx.x] += n;
        __syncthreads();
    }

    if (gix < len3.x and giy < len3.y) { xdata[id] = buf[threadIdx.y][threadIdx.x] * ebx2; }
}

template <typename T, typename EQ = int32_t, typename FP = T, int BLK = 8>
__global__ void x_lorenzo_3d1l(EQ* eq, T* scattered_outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata)
{
    __shared__ T buf[BLK][BLK][BLK + 1];

    auto giz = blockIdx.z * BLK + threadIdx.z, giy = blockIdx.y * BLK + threadIdx.y,
         gix  = blockIdx.x * BLK + threadIdx.x;
    size_t id = gix + giy * stride3.y + giz * stride3.z;  // low to high in dim, inner to outer

    if (gix < len3.x and giy < len3.y and giz < len3.z)
        buf[threadIdx.z][threadIdx.y][threadIdx.x] = scattered_outlier[id] + static_cast<T>(eq[id]) - radius;  // id
    else
        buf[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    for (auto dist = 1; dist < BLK; dist *= 2) {
        T addend = 0;
        if (threadIdx.x >= dist) addend = buf[threadIdx.z][threadIdx.y][threadIdx.x - dist];
        __syncthreads();
        if (threadIdx.x >= dist) buf[threadIdx.z][threadIdx.y][threadIdx.x] += addend;
        __syncthreads();
    }

    for (auto dist = 1; dist < BLK; dist *= 2) {
        T addend = 0;
        if (threadIdx.y >= dist) addend = buf[threadIdx.z][threadIdx.y - dist][threadIdx.x];
        __syncthreads();
        if (threadIdx.y >= dist) buf[threadIdx.z][threadIdx.y][threadIdx.x] += addend;
        __syncthreads();
    }

    for (auto dist = 1; dist < BLK; dist *= 2) {
        T addend = 0;
        if (threadIdx.z >= dist) addend = buf[threadIdx.z - dist][threadIdx.y][threadIdx.x];
        __syncthreads();
        if (threadIdx.z >= dist) buf[threadIdx.z][threadIdx.y][threadIdx.x] += addend;
        __syncthreads();
    }

    if (gix < len3.x and giy < len3.y and giz < len3.z) {
        xdata[id] = buf[threadIdx.z][threadIdx.y][threadIdx.x] * ebx2;
    }
}

}  // namespace prototype
}  // namespace __kernel
}  // namespace cuda
}  // namespace psz

#endif
