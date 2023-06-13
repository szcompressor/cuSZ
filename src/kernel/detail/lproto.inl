/**
 * @file lorenzo_proto.inl
 * @author Jiannan Tian
 * @brief (prototype) Dual-Eq Lorenzo method.
 * @version 0.2
 * @date 2019-09-23
 * (create) 2019-09-23 (rev) 2023-04-03
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_KERNEL_LORENZO_PROTOTYPE_CUH
#define CUSZ_KERNEL_LORENZO_PROTOTYPE_CUH

#include <cstddef>
#include <stdexcept>

#include "../../utils/it_cuda.hh"
#include "pipeline/compact_cuda.inl"
#include "utils/cuda_err.cuh"
#include "utils/timer.h"

namespace psz {

namespace cuda {
namespace __kernel {

namespace proto {  // easy algorithmic description

template <typename T, typename Eq = int32_t, typename Fp = T, typename Compact = CompactCudaDram<T>, int BLK = 256>
__global__ void c_lorenzo_1d1l(T* in_data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq, Compact compact)
{
    SETUP_ND_GPU_CUDA;
    __shared__ T buf[BLK];

    auto id   = gid1();
    auto data = [&](auto dx) -> T& { return buf[t().x + dx]; };

    // prequant (fp presence)
    if (id < len3.x) { data(0) = round(in_data[id] * ebx2_r); }
    __syncthreads();

    T    delta       = data(0) - (t().x == 0 ? 0 : data(-1));
    bool quantizable = fabs(delta) < radius;
    T    candidate   = delta + radius;
    if (check_boundary1()) {  // postquant
        eq[id] = quantizable * static_cast<Eq>(candidate);
        if (not quantizable) {
            auto dram_idx         = atomicAdd(compact.num, 1);
            compact.val[dram_idx] = candidate;
            compact.idx[dram_idx] = id;
        }
    }
}

template <typename T, typename Eq = int32_t, typename Fp = T, typename Compact = CompactCudaDram<T>, int BLK = 16>
__global__ void c_lorenzo_2d1l(T* in_data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq, Compact compact)
{
    SETUP_ND_GPU_CUDA;

    __shared__ T buf[BLK][BLK + 1];

    auto y = threadIdx.y, x = threadIdx.x;
    auto data = [&](auto dx, auto dy) -> T& { return buf[t().y + dy][t().x + dx]; };

    auto id = gid2();

    if (check_boundary2()) { data(0, 0) = round(in_data[id] * ebx2_r); }
    __syncthreads();

    T delta = data(0, 0) - ((x > 0 ? data(-1, 0) : 0) +             // dist=1
                            (y > 0 ? data(0, -1) : 0) -             // dist=1
                            (x > 0 and y > 0 ? data(-1, -1) : 0));  // dist=2

    bool quantizable = fabs(delta) < radius;
    T    candidate   = delta + radius;
    if (check_boundary2()) {
        eq[id] = quantizable * static_cast<Eq>(candidate);
        if (not quantizable) {
            auto dram_idx         = atomicAdd(compact.num, 1);
            compact.val[dram_idx] = candidate;
            compact.idx[dram_idx] = id;
        }
    }
}

template <typename T, typename Eq = int32_t, typename Fp = T, typename Compact = CompactCudaDram<T>, int BLK = 8>
__global__ void c_lorenzo_3d1l(T* in_data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq, Compact compact)
{
    SETUP_ND_GPU_CUDA;
    __shared__ T buf[BLK][BLK][BLK + 1];

    auto z = t().z, y = t().y, x = t().x;
    auto data = [&](auto dx, auto dy, auto dz) -> T& { return buf[t().z + dz][t().y + dy][t().x + dx]; };

    auto id = gid3();
    if (check_boundary3()) { data(0, 0, 0) = round(in_data[id] * ebx2_r); }
    __syncthreads();

    T delta = data(0, 0, 0) - ((z > 0 and y > 0 and x > 0 ? data(-1, -1, -1) : 0)  // dist=3
                               - (y > 0 and x > 0 ? data(-1, -1, 0) : 0)           // dist=2
                               - (z > 0 and x > 0 ? data(-1, 0, -1) : 0)           //
                               - (z > 0 and y > 0 ? data(0, -1, -1) : 0)           //
                               + (x > 0 ? data(-1, 0, 0) : 0)                      // dist=1
                               + (y > 0 ? data(0, -1, 0) : 0)                      //
                               + (z > 0 ? data(0, 0, -1) : 0));                    //

    bool quantizable = fabs(delta) < radius;
    T    candidate   = delta + radius;
    if (check_boundary3()) {
        eq[id] = quantizable * static_cast<Eq>(candidate);
        if (not quantizable) {
            auto dram_idx         = atomicAdd(compact.num, 1);
            compact.val[dram_idx] = candidate;
            compact.idx[dram_idx] = id;
        }
    }
}

template <typename T, typename Eq = int32_t, typename Fp = T, int BLK = 256>
__global__ void x_lorenzo_1d1l(Eq* eq, T* scattered_outlier, dim3 len3, dim3 stride3, int radius, Fp ebx2, T* xdata)
{
    SETUP_ND_GPU_CUDA;
    __shared__ T buf[BLK];

    auto id   = gid1();
    auto data = [&](auto dx) -> T& { return buf[t().x + dx]; };

    if (id < len3.x)
        data(0) = scattered_outlier[id] + static_cast<T>(eq[id]) - radius;  // fuse
    else
        data(0) = 0;
    __syncthreads();

    for (auto d = 1; d < BLK; d *= 2) {
        T n = 0;
        if (t().x >= d) n = data(-d);  // like __shfl_up_sync(0x1f, var, d); warp_sync
        __syncthreads();
        if (t().x >= d) data(0) += n;
        __syncthreads();
    }

    if (id < len3.x) { xdata[id] = data(0) * ebx2; }
}

template <typename T, typename Eq = int32_t, typename Fp = T, int BLK = 16>
__global__ void x_lorenzo_2d1l(Eq* eq, T* scattered_outlier, dim3 len3, dim3 stride3, int radius, Fp ebx2, T* xdata)
{
    SETUP_ND_GPU_CUDA;
    __shared__ T buf[BLK][BLK + 1];

    auto id   = gid2();
    auto data = [&](auto dx, auto dy) -> T& { return buf[t().y + dy][t().x + dx]; };

    if (check_boundary2())
        data(0, 0) = scattered_outlier[id] + static_cast<T>(eq[id]) - radius;  // fuse
    else
        data(0, 0) = 0;
    __syncthreads();

    for (auto d = 1; d < BLK; d *= 2) {
        T n = 0;
        if (t().x >= d) n = data(-d, 0);
        __syncthreads();
        if (t().x >= d) data(0, 0) += n;
        __syncthreads();
    }

    for (auto d = 1; d < BLK; d *= 2) {
        T n = 0;
        if (t().y >= d) n = data(0, -d);
        __syncthreads();
        if (t().y >= d) data(0, 0) += n;
        __syncthreads();
    }

    if (check_boundary2()) { xdata[id] = data(0, 0) * ebx2; }
}

template <typename T, typename Eq = int32_t, typename Fp = T, int BLK = 8>
__global__ void x_lorenzo_3d1l(Eq* eq, T* scattered_outlier, dim3 len3, dim3 stride3, int radius, Fp ebx2, T* xdata)
{
    SETUP_ND_GPU_CUDA;
    __shared__ T buf[BLK][BLK][BLK + 1];

    auto id   = gid3();
    auto data = [&](auto dx, auto dy, auto dz) -> T& { return buf[t().z + dz][t().y + dy][t().x + dx]; };

    if (check_boundary3())
        data(0, 0, 0) = scattered_outlier[id] + static_cast<T>(eq[id]) - radius;
    else
        data(0, 0, 0) = 0;
    __syncthreads();

    for (auto dist = 1; dist < BLK; dist *= 2) {
        T addend = 0;
        if (t().x >= dist) addend = data(-dist, 0, 0);
        __syncthreads();
        if (t().x >= dist) data(0, 0, 0) += addend;
        __syncthreads();
    }

    for (auto dist = 1; dist < BLK; dist *= 2) {
        T addend = 0;
        if (t().y >= dist) addend = data(0, -dist, 0);
        __syncthreads();
        if (t().y >= dist) data(0, 0, 0) += addend;
        __syncthreads();
    }

    for (auto dist = 1; dist < BLK; dist *= 2) {
        T addend = 0;
        if (t().z >= dist) addend = data(0, 0, -dist);
        __syncthreads();
        if (t().z >= dist) data(0, 0, 0) += addend;
        __syncthreads();
    }

    if (check_boundary3()) { xdata[id] = data(0, 0, 0) * ebx2; }
}

}  // namespace proto
}  // namespace __kernel
}  // namespace cuda
}  // namespace psz

#endif
