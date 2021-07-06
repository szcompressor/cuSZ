/**
 * @file legacy_lorenzo.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-04-29
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef KERNEL_LEGACY_LORENZU_CUH
#define KERNEL_LEGACY_LORENZU_CUH

#if CUDART_VERSION >= 11000
#pragma message(__FILE__ ": (CUDA 11 onward), cub from system path")
#include <cub/cub.cuh>
#else
#pragma message(__FILE__ ": (CUDA 10 or earlier), cub from git submodule")
#include "../../external/cub/cub/cub.cuh"
#endif

#include "../metadata.hh"
#include "../type_aliasing.hh"

#if __cplusplus >= 201703L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

extern __shared__ char scratch[];

// clang-format off
namespace legacy_kernel { 
template <typename Data, typename Quant> __global__ void x_lorenzo_2d1l_v0_16x16data_mapto_16x1(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_v2_8x8x8data_mapto_8x1x8(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_v3_8x8x8data_mapto_8x1x8(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_v4_8x8x8data_mapto_8x1x8(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_v5_32x8x8data_mapto_32x1x8(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_v6_32x8x8data_mapto_32x1x8(lorenzo_unzip, Data*, Data*, Quant*);
}
// clang-format on

template <typename Data, typename Quant>
__global__ void
legacy_kernel::x_lorenzo_2d1l_v0_16x16data_mapto_16x1(lorenzo_unzip ctx, Data* xdata, Data* outlier, Quant* quant)
{
    static const auto Block = 16;
    static_assert(Block == 16, "In one case, we need Block for 2D == 16");

    Data thread_scope[Block];
    //     ------> gi0 (x)
    //  |   t0    t1    t2    t3
    //  |  ts0_0 ts0_0 ts0_0 ts0_0
    //  |  ts0_1 ts0_1 ts0_1 ts0_1
    // gi1 ts0_2 ts0_2 ts0_2 ts0_2
    // (y)  |     |     |     |
    //     ts0_f ts0_f ts0_f ts0_f

    auto gi0      = bix * Block + tix;
    auto gi1_base = biy * Block;  // bdy: 16 -> 1
    auto radius   = static_cast<Data>(ctx.radius);

    auto get_gid = [&](auto i) { return (gi1_base + i) * ctx.stride1 + gi0; };

#pragma unroll
    for (auto i = 0; i < Block; i++) {
        auto gid = get_gid(i);

        if (gi0 < ctx.d0 and gi1_base + i < ctx.d1)
            thread_scope[i] = outlier[gid] + static_cast<Data>(quant[gid]) - radius;  // fuse
        else
            thread_scope[i] = 0;
        __syncthreads();
    }

    // sequential partial-sum
    for (auto i = 1; i < Block; i++) thread_scope[i] += thread_scope[i - 1];
    __syncthreads();

    // shuffle
#pragma unroll
    for (auto& i : thread_scope) {
        for (auto d = 1; d < Block; d *= 2) {
            Data n = __shfl_up_sync(0xffffffff, i, d, 16);
            if (tix >= d) i += n;
        }
        i *= ctx.ebx2;
    }

#pragma unroll
    for (auto i = 0; i < Block; i++) {
        auto gid = get_gid(i);
        if (gi0 < ctx.d0 and gi1_base + i < ctx.d1) xdata[gid] = thread_scope[i];
    }
    __syncthreads();
}

template <typename Data, typename Quant>
__global__ void
legacy_kernel::x_lorenzo_3d1l_v2_8x8x8data_mapto_8x1x8(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* quant)
{
    static const auto Block          = 8;
    static const auto YSequentiality = Block;
    static_assert(Block == 8, "In one case, we need Block for 3D == 8");

    __shared__ Data intermediate[Block][Block][Block];
    Data            thread_scope[YSequentiality];

    auto gi0      = bix * Block + tix;
    auto gi1_base = biy * Block;
    auto gi2      = biz * Block + tiz;
    auto radius   = static_cast<Data>(ctx.radius);
    auto get_gid  = [&](auto i) { return gi2 * ctx.stride2 + (gi1_base + i) * ctx.stride1 + gi0; };

    // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
#pragma unroll
    for (auto i = 0; i < YSequentiality; i++) {
        auto gid = get_gid(i);
        if (gi0 < ctx.d0 and gi1_base + i < ctx.d1 and gi2 < ctx.d2)
            thread_scope[i] = outlier[gid] + static_cast<Data>(quant[gid]) - radius;  // fuse
        else
            thread_scope[i] = 0;
    }
    // sequential partial-sum
    for (auto i = 1; i < YSequentiality; i++) thread_scope[i] += thread_scope[i - 1];

        // shuffle
#pragma unroll
    for (auto& i : thread_scope) {
        // partial-sum
        for (auto d = 1; d < Block; d *= 2) {
            Data n = __shfl_up_sync(0xffffffff, i, d, 8);
            if (tix >= d) i += n;
        }
        // y-index
        auto y = &i - thread_scope;
        // xz transpose
        intermediate[tiz][y][tix] = i;
        __syncthreads();  // necessary barrier
        i = intermediate[tix][y][tiz];

        // partial-sum
        for (auto d = 1; d < Block; d *= 2) {
            Data n = __shfl_up_sync(0xffffffff, i, d, 8);
            if (tix >= d) i += n;
        }
        i *= ctx.ebx2;  // scale by eb*2
    }
    gi0 = bix * Block + tiz, gi2 = biz * Block + tix;
#pragma unroll
    for (auto i = 0; i < YSequentiality; i++) {
        if (gi0 < ctx.d0 and gi1_base + i < ctx.d1 and gi2 < ctx.d2) { data[get_gid(i)] = thread_scope[i]; }
    }
}

template <typename Data, typename Quant>
__global__ void
legacy_kernel::x_lorenzo_3d1l_v3_8x8x8data_mapto_8x1x8(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* quant)
{
    static const auto Block          = 8;
    static const auto YSequentiality = Block;
    static_assert(Block == 8, "In one case, we need Block for 3D == 8");

    __shared__ Data intermediate[Block][Block];
    Data            thread_scope[YSequentiality];

    auto gi0 = bix * Block + tix, gi1_base = biy * Block, gi2 = biz * Block + tiz;
    auto get_gid = [&](auto y) { return gi2 * ctx.stride2 + (gi1_base + y) * ctx.stride1 + gi0; };

    auto y = 0;

    // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
#pragma unroll
    for (y = 0; y < YSequentiality; y++) {
        auto gid = get_gid(y);
        if (gi0 < ctx.d0 and gi1_base + y < ctx.d1 and gi2 < ctx.d2)
            thread_scope[y] = outlier[gid] + static_cast<Data>(quant[gid]) - static_cast<Data>(ctx.radius);  // fuse
        else
            thread_scope[y] = 0;
    }
    // sequential partial-sum
    for (y = 1; y < YSequentiality; y++) thread_scope[y] += thread_scope[y - 1];

    // shuffle, ND partial-sums
    auto dist = 1;
    Data addend;

#pragma unroll
    for (auto& val : thread_scope) {
        // clang-format off
        for (dist = 1; dist < Block; dist *= 2) { addend = __shfl_up_sync(0xffffffff, val, dist, 8); if (tix >= dist) val += addend; }
        intermediate[tiz][tix] = val; __syncthreads(); val = intermediate[tix][tiz]; // xz transpose
        __syncthreads();  
        for (dist = 1; dist < Block; dist *= 2) { addend = __shfl_up_sync(0xffffffff, val, dist, 8); if (tix >= dist) val += addend; }
        // clang-format on
    }

    gi0 = bix * Block + tiz, gi2 = biz * Block + tix;
#pragma unroll
    for (y = 0; y < YSequentiality; y++) {
        if (gi0 < ctx.d0 and gi1_base + y < ctx.d1 and gi2 < ctx.d2) { data[get_gid(y)] = thread_scope[y] * ctx.ebx2; }
    }
}

template <typename Data, typename Quant>
__global__ void
legacy_kernel::x_lorenzo_3d1l_v4_8x8x8data_mapto_8x1x8(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* quant)
{
    static const auto Block          = 8;
    static const auto YSequentiality = Block;
    static_assert(Block == 8, "In one case, we need Block for 3D == 8");

    __shared__ Data intermediate[Block][Block];
    Data            thread_scope[YSequentiality];

    auto gi0 = bix * Block + tix, gi1_base = biy * Block, gi2 = biz * Block + tiz;
    auto get_gid = [&](auto y) { return gi2 * ctx.stride2 + (gi1_base + y) * ctx.stride1 + gi0; };

    auto y = 0;

    // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
#pragma unroll
    for (y = 0; y < YSequentiality; y++) {
        auto gid = get_gid(y);
        if (gi0 < ctx.d0 and gi1_base + y < ctx.d1 and gi2 < ctx.d2)
            thread_scope[y] = outlier[gid] + static_cast<Data>(quant[gid]) - static_cast<Data>(ctx.radius);  // fuse
        else
            thread_scope[y] = 0;
    }
    // sequential partial-sum
    for (y = 1; y < YSequentiality; y++) thread_scope[y] += thread_scope[y - 1];

    // shuffle, ND partial-sums
    auto dist = 1;
    Data addend;

#pragma unroll
    for (auto i = 0; i < Block; i++) {
        Data val = thread_scope[i];
        for (dist = 1; dist < Block; dist *= 2) {
            addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (tix >= dist) val += addend;
        }
        // x-z transpose
        intermediate[tiz][tix] = val;
        __syncthreads();
        val = intermediate[tix][tiz];
        __syncthreads();

        for (dist = 1; dist < Block; dist *= 2) {
            addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (tix >= dist) val += addend;
        }

        intermediate[tiz][tix] = val;
        __syncthreads();
        val = intermediate[tix][tiz];
        __syncthreads();

        thread_scope[i] = val;
    }

    gi0 = bix * Block + tix, gi2 = biz * Block + tiz;
#pragma unroll
    for (y = 0; y < YSequentiality; y++) {
        if (gi0 < ctx.d0 and gi1_base + y < ctx.d1 and gi2 < ctx.d2) { data[get_gid(y)] = thread_scope[y] * ctx.ebx2; }
    }
}

template <typename Data, typename Quant>
__global__ void
legacy_kernel::x_lorenzo_3d1l_v5_32x8x8data_mapto_32x1x8(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* quant)
{
    static const auto Block          = 8;
    static const auto YSequentiality = Block;
    static_assert(Block == 8, "In one case, we need Block for 3D == 8");

    __shared__ Data intermediate[4][Block][Block];
    Data            thread_scope[YSequentiality];

    auto seg_id  = tix / 8;
    auto seg_tix = tix % 8;

    auto gi0 = bix * (4 * Block) + tix, gi1_base = biy * Block, gi2 = biz * Block + tiz;
    auto get_gid = [&](auto y) { return gi2 * ctx.stride2 + (gi1_base + y) * ctx.stride1 + gi0; };

    auto y = 0;

    // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
#pragma unroll
    for (y = 0; y < YSequentiality; y++) {
        auto gid = get_gid(y);
        if (gi0 < ctx.d0 and gi1_base + y < ctx.d1 and gi2 < ctx.d2)
            thread_scope[y] = outlier[gid] + static_cast<Data>(quant[gid]) - static_cast<Data>(ctx.radius);  // fuse
        else
            thread_scope[y] = 0;
    }
    // sequential partial-sum
    for (y = 1; y < YSequentiality; y++) thread_scope[y] += thread_scope[y - 1];

    // shuffle, ND partial-sums
    auto dist = 1;
    Data addend;

#pragma unroll
    for (auto i = 0; i < Block; i++) {
        Data val = thread_scope[i];

        for (dist = 1; dist < Block; dist *= 2) {
            addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        // x-z transpose
        intermediate[seg_id][tiz][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_id][seg_tix][tiz];
        __syncthreads();

        for (dist = 1; dist < Block; dist *= 2) {
            addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        intermediate[seg_id][tiz][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_id][seg_tix][tiz];
        __syncthreads();

        thread_scope[i] = val;
    }

#pragma unroll
    for (y = 0; y < YSequentiality; y++) {
        if (gi0 < ctx.d0 and gi1_base + y < ctx.d1 and gi2 < ctx.d2) { data[get_gid(y)] = thread_scope[y] * ctx.ebx2; }
    }
}

template <typename Data, typename Quant>
__global__ void
legacy_kernel::x_lorenzo_3d1l_v6_32x8x8data_mapto_32x1x8(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* quant)
{
    static const auto Block          = 8;
    static const auto YSequentiality = Block;
    static_assert(Block == 8, "In one case, we need Block for 3D == 8");

    __shared__ Data intermediate[4][Block][Block];
    Data            thread_scope = 0;

    auto seg_id  = tix / 8;
    auto seg_tix = tix % 8;

    auto gi0 = bix * (4 * Block) + tix, gi1_base = biy * Block, gi2 = biz * Block + tiz;
    auto get_gid = [&](auto y) { return gi2 * ctx.stride2 + (gi1_base + y) * ctx.stride1 + gi0; };

    auto y = 0;

    // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
#pragma unroll
    for (y = 0; y < YSequentiality; y++) {
        auto gid = get_gid(y);
        if (gi0 < ctx.d0 and gi1_base + y < ctx.d1 and gi2 < ctx.d2)
            thread_scope += outlier[gid] + static_cast<Data>(quant[gid]) - static_cast<Data>(ctx.radius);  // fuse

        // shuffle, ND partial-sums

        Data val = thread_scope;

        for (auto dist = 1; dist < Block; dist *= 2) {
            Data addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        // x-z transpose
        intermediate[seg_id][tiz][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_id][seg_tix][tiz];
        __syncthreads();

        for (auto dist = 1; dist < Block; dist *= 2) {
            Data addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        intermediate[seg_id][tiz][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_id][seg_tix][tiz];
        __syncthreads();

        // thread_scope += val;

        if (gi0 < ctx.d0 and gi1_base + y < ctx.d1 and gi2 < ctx.d2) { data[get_gid(y)] = val * ctx.ebx2; }
    }
}

#endif