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

using DIM    = unsigned int;
using STRIDE = unsigned int;

namespace kernel {

// clang-format off
template <typename Data, typename Quant, typename FP = float, int Block = 256, int Sequentiality = 4> __global__ void c_lorenzo_1d1l_v2
(Data*, Quant*, DIM, int, FP);
template <typename Data, typename Quant, typename FP = float, int Block = 256, int Sequentiality = 8> __global__ void x_lorenzo_1d1l_cub
(Data*, Data*, Quant*, DIM, int, FP);

template <typename Data, typename Quant, typename FP = float> __global__ void c_lorenzo_2d1l_v1_16x16data_mapto_16x2
(Data*, Quant*, DIM, DIM, STRIDE, int, FP);
template <typename Data, typename Quant, typename FP = float> __global__ void x_lorenzo_2d1l_v1_16x16data_mapto_16x2
(Data*, Data*, Quant*, DIM, DIM, STRIDE, int, FP);

template <typename Data, typename Quant, typename FP = float> __global__ void c_lorenzo_3d1l_v1_32x8x8data_mapto_32x1x8
(Data*, Quant*, DIM, DIM, DIM, STRIDE, STRIDE, int, FP);
template <typename Data, typename Quant, typename FP = float> __global__ void x_lorenzo_3d1l_v5var1_32x8x8data_mapto_32x1x8
(Data*, Data*, Quant*, DIM, DIM, DIM, STRIDE, STRIDE, int, FP);
template <typename Data, typename Quant, typename FP = float> __global__ void x_lorenzo_3d1l_v6var1_32x8x8data_mapto_32x1x8
(Data*, Data*, Quant*, DIM, DIM, DIM, STRIDE, STRIDE, int, FP);
// clang-format on

}  // namespace kernel

template <typename Data, typename Quant, typename FP, int Block, int Sequentiality>
__global__ void kernel::c_lorenzo_1d1l_v2(  //
    Data*  d,
    Quant* q,
    DIM    dimx,
    int    radius,
    FP     ebx2_r)
{
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
        if (id < dimx) { shmem.data[tix + i * nThreads] = round(d[id] * ebx2_r); }
    }
    __syncthreads();
    for (auto i = 0; i < Sequentiality; i++) thread_scope[i] = shmem.data[tix * Sequentiality + i];
    if (tix > 0) from_last_stripe = shmem.data[tix * Sequentiality - 1];
    __syncthreads();

    auto shmem_quant = reinterpret_cast<Quant*>(shmem.uninitialized + sizeof(Data) * Block);

    // i == 0
    {
        Data delta                           = thread_scope[0] - from_last_stripe;
        bool quantizable                     = fabs(delta) < radius;
        Data candidate                       = delta + radius;
        shmem.data[0 + tix * Sequentiality]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
        shmem_quant[0 + tix * Sequentiality] = quantizable * static_cast<Quant>(candidate);
    }
#pragma unroll
    for (auto i = 1; i < Sequentiality; i++) {
        Data delta                           = thread_scope[i] - thread_scope[i - 1];
        bool quantizable                     = fabs(delta) < radius;
        Data candidate                       = delta + radius;
        shmem.data[i + tix * Sequentiality]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
        shmem_quant[i + tix * Sequentiality] = quantizable * static_cast<Quant>(candidate);
    }
    __syncthreads();

#pragma unroll
    for (auto i = 0; i < Sequentiality; i++) {
        auto id = id_base + tix + i * nThreads;
        if (id < dimx) {  //
            q[id] = shmem_quant[tix + i * nThreads];
            d[id] = shmem.data[tix + i * nThreads];
        }
    }
}

template <typename Data, typename Quant, typename FP>
__global__ void kernel::c_lorenzo_2d1l_v1_16x16data_mapto_16x2(
    Data*  d,
    Quant* q,
    DIM    dimx,
    DIM    dimy,
    STRIDE stridey,
    int    radius,
    FP     ebx2_r)
{
    static const auto Block          = 16;
    static const auto YSequentiality = 8;

    Data center[YSequentiality + 1] = {0};  //   nw  north
                                            // west  center

    auto gi0      = bix * bdx + tix;                     // bdx == 16
    auto gi1_base = biy * Block + tiy * YSequentiality;  // bdy * YSequentiality = Block == 16
    auto get_gid  = [&](auto i) { return (gi1_base + i) * stridey + gi0; };

    // read from global memory
#pragma unroll
    for (auto i = 0; i < YSequentiality; i++) {
        if (gi0 < dimx and gi1_base + i < dimy) center[i + 1] = round(d[get_gid(i)] * ebx2_r);
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
        bool quantizable = fabs(center[i]) < radius;
        Data candidate   = center[i] + radius;
        if (gi0 < dimx and gi1_base + i - 1 < dimy) {
            d[gid] = (1 - quantizable) * candidate;  // output; reuse data for outlier
            q[gid] = quantizable * static_cast<Quant>(candidate);
        }
    }
}

template <typename Data, typename Quant, typename FP>
__global__ void kernel::c_lorenzo_3d1l_v1_32x8x8data_mapto_32x1x8(
    Data*  d,
    Quant* q,
    DIM    dimx,
    DIM    dimy,
    DIM    dimz,
    STRIDE stridey,
    STRIDE stridez,
    int    radius,
    FP     ebx2_r)
{
    static const auto Block = 8;
    __shared__ Data   shmem[Block][Block][4 * Block];

    auto z = tiz;

    auto gi0      = bix * (Block * 4) + tix;
    auto gi1_base = biy * Block;
    auto gi2      = biz * Block + z;

    if (gi0 < dimx and gi2 < dimz) {
        auto base_id = gi0 + gi1_base * stridey + gi2 * stridez;
        for (auto y = 0; y < Block; y++) {
            if (gi1_base + y < dimy) {
                shmem[z][y][tix] = round(d[base_id + y * stridey] * ebx2_r);  // prequant (fp presence)
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

        bool quantizable = fabs(delta) < radius;
        Data candidate   = delta + radius;

        auto id = gi0 + (gi1_base + y) * stridey + gi2 * stridez;
        if (gi0 < dimx and (gi1_base + y) < dimy and gi2 < dimz) {
            d[id] = (1 - quantizable) * candidate;  // output; reuse data for outlier
            q[id] = quantizable * static_cast<Quant>(candidate);
        }
    }
}

template <typename Data, typename Quant, typename FP, int Block, int Sequentiality>
__global__ void kernel::x_lorenzo_1d1l_cub(  //
    Data*  xdata,
    Data*  outlier,
    Quant* quant,
    DIM    dimx,
    int    radius,
    FP     ebx2)
{
    static const auto Db = Block / Sequentiality;  // dividable

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

#pragma unroll
    for (auto i = 0; i < Sequentiality; i++) {
        auto id = (bix * bdx + tix) * Sequentiality + i;
        thread_scope.xdata[i] =
            id < dimx ? thread_scope.outlier[i] + static_cast<Data>(thread_scope_quant[i]) - radius : 0;
    }
    __syncthreads();

    BlockScanT_xdata(temp_storage.scan_xdata).InclusiveSum(thread_scope.xdata, thread_scope.xdata);
    __syncthreads();  // barrier for shmem reuse

#pragma unroll
    for (auto i = 0; i < Sequentiality; i++) thread_scope.xdata[i] *= ebx2;
    __syncthreads();  // barrier for shmem reuse

    BlockStoreT_xdata(temp_storage.store_xdata).Store(xdata + (bix * bdx) * Sequentiality, thread_scope.xdata);
}

template <typename Data, typename Quant, typename FP>
__global__ void kernel::x_lorenzo_2d1l_v1_16x16data_mapto_16x2(
    Data*    xdata,
    Data*    outlier,
    Quant*   quant,
    DIM      dimx,
    unsigned dimy,
    unsigned stridey,
    int      radius,
    FP       ebx2)
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
    auto get_gid  = [&](auto i) { return (gi1_base + i) * stridey + gi0; };

#pragma unroll
    for (auto i = 0; i < YSequentiality; i++) {
        auto gid = get_gid(i);
        // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
        if (gi0 < dimx and gi1_base + i < dimy)
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
        i *= ebx2;
    }
#pragma unroll
    for (auto i = 0; i < YSequentiality; i++) {
        auto gid = get_gid(i);
        if (gi0 < dimx and gi1_base + i < dimy) xdata[gid] = thread_scope[i];
    }
}

template <typename Data, typename Quant, typename FP>
__global__ void kernel::x_lorenzo_3d1l_v5var1_32x8x8data_mapto_32x1x8(
    Data*  data,
    Data*  outlier,
    Quant* quant,
    DIM    dimx,
    DIM    dimy,
    DIM    dimz,
    STRIDE stridey,
    STRIDE stridez,
    int    radius,
    FP     ebx2)
{
    static const auto Block          = 8;
    static const auto YSequentiality = Block;
    static_assert(Block == 8, "In one case, we need Block for 3D == 8");

    __shared__ Data intermediate[Block][4][Block];
    Data            thread_scope[YSequentiality];

    auto seg_id  = tix / 8;
    auto seg_tix = tix % 8;

    auto gi0 = bix * (4 * Block) + tix, gi1_base = biy * Block, gi2 = biz * Block + tiz;
    auto get_gid = [&](auto y) { return gi2 * stridez + (gi1_base + y) * stridey + gi0; };

    auto y = 0;

    // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
#pragma unroll
    for (y = 0; y < YSequentiality; y++) {
        auto gid = get_gid(y);
        if (gi0 < dimx and gi1_base + y < dimy and gi2 < dimz)
            thread_scope[y] = outlier[gid] + static_cast<Data>(quant[gid]) - static_cast<Data>(radius);  // fuse
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
        intermediate[tiz][seg_id][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_tix][seg_id][tiz];
        __syncthreads();

        for (dist = 1; dist < Block; dist *= 2) {
            addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        intermediate[tiz][seg_id][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_tix][seg_id][tiz];
        __syncthreads();

        thread_scope[i] = val;
    }

#pragma unroll
    for (y = 0; y < YSequentiality; y++) {
        if (gi0 < dimx and gi1_base + y < dimy and gi2 < dimz) { data[get_gid(y)] = thread_scope[y] * ebx2; }
    }
}

template <typename Data, typename Quant, typename FP>
__global__ void kernel::x_lorenzo_3d1l_v6var1_32x8x8data_mapto_32x1x8(
    Data*  data,
    Data*  outlier,
    Quant* quant,
    DIM    dimx,
    DIM    dimy,
    DIM    dimz,
    STRIDE stridey,
    STRIDE stridez,
    int    radius,
    FP     ebx2)
{
    static const auto Block          = 8;
    static const auto YSequentiality = Block;
    static_assert(Block == 8, "In one case, we need Block for 3D == 8");

    __shared__ Data intermediate[Block][4][Block];
    Data            thread_scope = 0;

    auto seg_id  = tix / 8;
    auto seg_tix = tix % 8;

    auto gi0 = bix * (4 * Block) + tix, gi1_base = biy * Block, gi2 = biz * Block + tiz;
    auto get_gid = [&](auto y) { return gi2 * stridez + (gi1_base + y) * stridey + gi0; };

    auto y = 0;

    // even if we hit the else branch, all threads in a warp hit the y-boundary simultaneously
#pragma unroll
    for (y = 0; y < YSequentiality; y++) {
        auto gid = get_gid(y);
        if (gi0 < dimx and gi1_base + y < dimy and gi2 < dimz)
            thread_scope += outlier[gid] + static_cast<Data>(quant[gid]) - static_cast<Data>(radius);  // fuse

        Data val = thread_scope;

        // shuffle, ND partial-sums
        for (auto dist = 1; dist < Block; dist *= 2) {
            Data addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        // x-z transpose
        intermediate[tiz][seg_id][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_tix][seg_id][tiz];
        __syncthreads();

        for (auto dist = 1; dist < Block; dist *= 2) {
            Data addend = __shfl_up_sync(0xffffffff, val, dist, 8);
            if (seg_tix >= dist) val += addend;
        }

        intermediate[tiz][seg_id][seg_tix] = val;
        __syncthreads();
        val = intermediate[seg_tix][seg_id][tiz];
        __syncthreads();

        // thread_scope += val;

        if (gi0 < dimx and gi1_base + y < dimy and gi2 < dimz) { data[get_gid(y)] = val * ebx2; }
    }
}

#endif
