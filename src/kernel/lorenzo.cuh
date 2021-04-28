/**
 * @file lorenzo.cuh
 * @author Jiannan Tian
 * @brief Dual-Quant Lorenzo method.
 * @version 0.2
 * @date 2021-01-16
 * (create) 19-09-23; (release) 2020-09-20; (rev1) 2021-01-16; (rev2) 2021-02-20; (rev3) 2021-04-11
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef KERNEL_LORENZO_CUH
#define KERNEL_LORENZO_CUH

#include <cstddef>

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
namespace kernel {
template <typename Data, typename Quant, int Sequentiality = 8> __global__ void c_lorenzo_1d1l_v2(lorenzo_zip, Data*, Quant*);
template <typename Data, typename Quant> __global__ void c_lorenzo_2d1l_v1_16x16data_mapto_16x2(lorenzo_zip, Data*, Quant*);
template <typename Data, typename Quant> __global__ void c_lorenzo_3d1l_v1_32x8x8data_mapto_32x1x8(lorenzo_zip, Data*, Quant*);

template <typename Data, typename Quant> __global__ void x_lorenzo_1d1l_cub(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_2d1l_v1_16x16data_mapto_16x2(lorenzo_unzip, Data*, Data*, Quant*);

template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_v4_8x8x8data_mapto_8x1x8(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_v5_32x8x8data_mapto_32x1x8(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_v6_32x8x8data_mapto_32x1x8(lorenzo_unzip, Data*, Data*, Quant*);
}

namespace legacy_kernel { 
template <typename Data, typename Quant> __global__ void x_lorenzo_2d1l_v0_16x16data_mapto_16x1(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_v2_8x8x8data_mapto_8x1x8(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l_v3_8x8x8data_mapto_8x1x8(lorenzo_unzip, Data*, Data*, Quant*);
}
// clang-format on

template <typename Data, typename Quant, int Sequentiality>
__global__ void kernel::c_lorenzo_1d1l_v2(lorenzo_zip ctx, Data* d, Quant* q)
{
    static const auto Block = MetadataTrait<1>::Block;

    static const auto nThreads = Block / Sequentiality;

    __shared__ union ShareMem {
        uint8_t  uninitialized[Block * (sizeof(Data) + sizeof(Quant))];
        Data     data[Block];
        uint32_t outlier_loc[Block];
    } shmem;

    auto id_base = bix * Block;

    Data thread_scope[Sequentiality];
    Data from_last_stripe{0};

#pragma unroll
    for (auto i = 0; i < Sequentiality; i++) {
        auto id = id_base + tix + i * nThreads;
        if (id < ctx.d0) { shmem.data[tix + i * nThreads] = round(d[id] * ctx.ebx2_r); }
    }
    __syncthreads();
    for (auto i = 0; i < Sequentiality; i++) thread_scope[i] = shmem.data[tix * Sequentiality + i];
    if (tix > 0) from_last_stripe = shmem.data[tix * Sequentiality - 1];
    __syncthreads();

    auto shmem_quant = reinterpret_cast<Quant*>(shmem.uninitialized + sizeof(Data) * Block);

    // i == 0
    {
        Data delta                           = thread_scope[0] - from_last_stripe;
        bool quantizable                     = fabs(delta) < ctx.radius;
        Data candidate                       = delta + ctx.radius;
        shmem.data[0 + tix * Sequentiality]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
        shmem_quant[0 + tix * Sequentiality] = quantizable * static_cast<Quant>(candidate);
    }
#pragma unroll
    for (auto i = 1; i < Sequentiality; i++) {
        Data delta                           = thread_scope[i] - thread_scope[i - 1];
        bool quantizable                     = fabs(delta) < ctx.radius;
        Data candidate                       = delta + ctx.radius;
        shmem.data[i + tix * Sequentiality]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
        shmem_quant[i + tix * Sequentiality] = quantizable * static_cast<Quant>(candidate);
    }
    __syncthreads();

#pragma unroll
    for (auto i = 0; i < Sequentiality; i++) {
        auto id = id_base + tix + i * nThreads;
        if (id < ctx.d0) {  //
            q[id] = shmem_quant[tix + i * nThreads];
            d[id] = shmem.data[tix + i * nThreads];
        }
    }
}

template <typename Data, typename Quant>
__global__ void kernel::c_lorenzo_2d1l_v1_16x16data_mapto_16x2(lorenzo_zip ctx, Data* d, Quant* q)
{
    static const auto Block          = 16;
    static const auto YSequentiality = 8;

    Data center[YSequentiality + 1] = {0};  //   nw  north
                                            // west  center

    auto gi0      = bix * bdx + tix;                     // bdx == 16
    auto gi1_base = biy * Block + tiy * YSequentiality;  // bdy * YSequentiality = Block == 16
    auto radius   = static_cast<Data>(ctx.radius);
    auto get_gid  = [&](auto i) { return (gi1_base + i) * ctx.stride1 + gi0; };

    // read from global memory
#pragma unroll
    for (auto i = 0; i < YSequentiality; i++) {
        if (gi0 < ctx.d0 and gi1_base + i < ctx.d1) center[i + 1] = round(d[get_gid(i)] * ctx.ebx2_r);
    }

    auto tmp = __shfl_up_sync(0xffffffff, center[YSequentiality], 16);  // same-warp, next-16
    if (tiy == 1) center[0] = tmp;

#pragma unroll
    for (auto i = YSequentiality; i > 0; i--) {
        center[i] -= center[i - 1];
        auto west = __shfl_up_sync(0xffffffff, center[i], 1, 16);
        if (tix > 0) center[i] -= west;
    }
    __syncthreads();

    // original form
    // Data delta = center[i] - center[i - 1] + west[i] - west[i - 1];
    // short form
    // Data delta = center[i] - west[i];

#pragma unroll
    for (auto i = 1; i < YSequentiality + 1; i++) {
        auto gid         = get_gid(i - 1);
        bool quantizable = fabs(center[i]) < ctx.radius;
        Data candidate   = center[i] + ctx.radius;
        if (gi0 < ctx.d0 and gi1_base + i - 1 < ctx.d1) {
            d[gid] = (1 - quantizable) * candidate;  // output; reuse data for outlier
            q[gid] = quantizable * static_cast<Quant>(candidate);
        }
    }
}

template <typename Data, typename Quant>
__global__ void kernel::c_lorenzo_3d1l_v1_32x8x8data_mapto_32x1x8(lorenzo_zip ctx, Data* d, Quant* q)
{
    static const auto Block = 8;
    __shared__ Data   shmem[Block][Block][4 * Block];

    auto z = tiz;

    auto gi0      = bix * (Block * 4) + tix;
    auto gi1_base = biy * Block;
    auto gi2      = biz * Block + z;

    if (gi0 < ctx.d0 and gi2 < ctx.d2) {
        auto base_id = gi0 + gi1_base * ctx.stride1 + gi2 * ctx.stride2;
        for (auto y = 0; y < Block; y++) {
            if (gi1_base + y < ctx.d1) {
                shmem[z][y][tix] = round(d[base_id + y * ctx.stride1] * ctx.ebx2_r);  // prequant (fp presence)
            }
        }
    }
    __syncthreads();  // necessary to ensure correctness

    auto x = tix % 8;

    for (auto y = 0; y < Block; y++) {
        Data delta;
        delta = shmem[z][y][tix] - ((z > 0 and y > 0 and x > 0 ? shmem[z - 1][y - 1][tix - 1] : 0)  // dist=3
                                    - (y > 0 and x > 0 ? shmem[z][y - 1][tix - 1] : 0)              // dist=2
                                    - (z > 0 and x > 0 ? shmem[z - 1][y][tix - 1] : 0)              //
                                    - (z > 0 and y > 0 ? shmem[z - 1][y - 1][tix] : 0)              //
                                    + (x > 0 ? shmem[z][y][tix - 1] : 0)                            // dist=1
                                    + (y > 0 ? shmem[z][y - 1][tix] : 0)                            //
                                    + (z > 0 ? shmem[z - 1][y][tix] : 0));                          //

        bool quantizable = fabs(delta) < ctx.radius;
        Data candidate   = delta + ctx.radius;

        auto id = gi0 + (gi1_base + y) * ctx.stride1 + gi2 * ctx.stride2;
        if (gi0 < ctx.d0 and (gi1_base + y) < ctx.d1 and gi2 < ctx.d2) {
            d[id] = (1 - quantizable) * candidate;  // output; reuse data for outlier
            q[id] = quantizable * static_cast<Quant>(candidate);
        }
    }
}

template <typename Data, typename Quant>
__global__ void kernel::x_lorenzo_1d1l_cub(lorenzo_unzip ctx, Data* xdata, Data* outlier, Quant* quant)
{
    static const auto Block         = MetadataTrait<1>::Block;
    static const auto Sequentiality = MetadataTrait<1>::Sequentiality;  // items per thread
    static const auto Db            = Block / Sequentiality;            // dividable

    // coalesce-load (warp-striped) and transpose in shmem (similar for store)
    typedef cub::BlockLoad<Data, Db, Sequentiality, cub::BLOCK_LOAD_WARP_TRANSPOSE>   BlockLoadT_outlier;
    typedef cub::BlockLoad<Quant, Db, Sequentiality, cub::BLOCK_LOAD_WARP_TRANSPOSE>  BlockLoadT_quant;
    typedef cub::BlockStore<Data, Db, Sequentiality, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT_xdata;
    typedef cub::BlockScan<Data, Db, cub::BLOCK_SCAN_RAKING_MEMOIZE> BlockScanT_xdata;  // TODO autoselect algorithm

    __shared__ union TempStorage {  // overlap shared memory space
        typename BlockLoadT_outlier::TempStorage load_outlier;
        typename BlockLoadT_quant::TempStorage   load_quant;
        typename BlockStoreT_xdata::TempStorage  store_xdata;
        typename BlockScanT_xdata::TempStorage   scan_xdata;
    } temp_storage;

    // thread-scope tiled data
    union ThreadData {
        Data xdata[Sequentiality];
        Data outlier[Sequentiality];
    } thread_scope;
    Quant thread_scope_quant[Sequentiality];

    // TODO pad for potential out-of-range access
    // (bix * bdx * Sequentiality) denotes the start of the data chunk that belongs to this thread block
    BlockLoadT_quant(temp_storage.load_quant).Load(quant + (bix * bdx) * Sequentiality, thread_scope_quant);
    __syncthreads();  // barrier for shmem reuse
    BlockLoadT_outlier(temp_storage.load_outlier).Load(outlier + (bix * bdx) * Sequentiality, thread_scope.outlier);
    __syncthreads();  // barrier for shmem reuse

    auto radius = static_cast<Data>(ctx.radius);
#pragma unroll
    for (auto i = 0; i < Sequentiality; i++) {
        auto id = (bix * bdx + tix) * Sequentiality + i;
        thread_scope.xdata[i] =
            id < ctx.d0 ? thread_scope.outlier[i] + static_cast<Data>(thread_scope_quant[i]) - radius : 0;
    }
    __syncthreads();

    BlockScanT_xdata(temp_storage.scan_xdata).InclusiveSum(thread_scope.xdata, thread_scope.xdata);
    __syncthreads();  // barrier for shmem reuse

#pragma unroll
    for (auto i = 0; i < Sequentiality; i++) thread_scope.xdata[i] *= ctx.ebx2;
    __syncthreads();  // barrier for shmem reuse

    BlockStoreT_xdata(temp_storage.store_xdata).Store(xdata + (bix * bdx) * Sequentiality, thread_scope.xdata);
}

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
kernel::x_lorenzo_2d1l_v1_16x16data_mapto_16x2(lorenzo_unzip ctx, Data* xdata, Data* outlier, Quant* quant)
{
    static const auto Block          = 16;
    static const auto YSequentiality = Block / 2;  // sequentiality in y direction
    static_assert(Block == 16, "In one case, we need Block for 2D == 16");

    __shared__ Data intermediate[Block];  // TODO use warp shuffle to eliminate this
    Data            thread_scope[YSequentiality];
    //     ------> gi0 (x)
    //  |  t00    t01    t02    t03   ... t0f
    //  |  ts00_0 ts00_0 ts00_0 ts00_0
    // gi1 ts00_1 ts00_1 ts00_1 ts00_1
    // (y)  |     |     |     |
    //     ts00_7 ts00_7 ts00_7 ts00_7
    //
    //  |  t10    t11    t12    t13   ... t1f
    //  |  ts00_0 ts00_0 ts00_0 ts00_0
    // gi1 ts00_1 ts00_1 ts00_1 ts00_1
    // (y)  |     |     |     |
    //     ts00_7 ts00_7 ts00_7 ts00_7

    auto gi0      = bix * Block + tix;
    auto gi1_base = biy * Block + tiy * YSequentiality;  // bdy * YSequentiality = Block == 16
    auto radius   = static_cast<Data>(ctx.radius);
    auto get_gid  = [&](auto i) { return (gi1_base + i) * ctx.stride1 + gi0; };

#pragma unroll
    for (auto i = 0; i < YSequentiality; i++) {
        auto gid = get_gid(i);
        // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
        if (gi0 < ctx.d0 and gi1_base + i < ctx.d1)
            thread_scope[i] = outlier[gid] + static_cast<Data>(quant[gid]) - radius;  // fuse
        else
            thread_scope[i] = 0;  // TODO set as init state?
    }
    // sequential partial-sum
    for (auto i = 1; i < YSequentiality; i++) thread_scope[i] += thread_scope[i - 1];
    // store for cross-thread update
    if (tiy == 0) intermediate[tix] = thread_scope[YSequentiality - 1];
    __syncthreads();  // somehow deletable
    // load and update
    if (tiy == 1) {
        auto tmp = intermediate[tix];
#pragma unroll
        for (auto& i : thread_scope) i += tmp;
    }
    // partial-sum
#pragma unroll
    for (auto& i : thread_scope) {
        for (auto d = 1; d < Block; d *= 2) {
            Data n = __shfl_up_sync(0xffffffff, i, d, 16);
            if (tix >= d) i += n;
        }
        i *= ctx.ebx2;
    }
#pragma unroll
    for (auto i = 0; i < YSequentiality; i++) {
        auto gid = get_gid(i);
        if (gi0 < ctx.d0 and gi1_base + i < ctx.d1) xdata[gid] = thread_scope[i];
    }
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
kernel::x_lorenzo_3d1l_v4_8x8x8data_mapto_8x1x8(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* quant)
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
kernel::x_lorenzo_3d1l_v5_32x8x8data_mapto_32x1x8(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* quant)
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
kernel::x_lorenzo_3d1l_v6_32x8x8data_mapto_32x1x8(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* quant)
{
    static const auto Block          = 8;
    static const auto YSequentiality = Block;
    static_assert(Block == 8, "In one case, we need Block for 3D == 8");

    __shared__ Data intermediate[4][Block][Block];
    // Data            thread_scope[YSequentiality];
    Data thread_scope = 0;

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
