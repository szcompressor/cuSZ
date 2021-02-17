/**
 * @file cusz_dualquant.cu
 * @author Jiannan Tian
 * @brief Dual-Quantization method of cuSZ.
 * @version 0.2
 * @date 2021-01-16
 * (create) 19-09-23; (release) 2020-09-20; (rev1) 2021-01-16
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#if CUDART_VERSION >= 11000
#pragma message(__FILE__ ": (CUDA 11 onward), cub from system path")
#include <cub/cub.cuh>
#else
#pragma message(__FILE__ ": (CUDA 10 or earlier), cub from git submodule")
#include "../external/cub/cub/cub.cuh"
#endif

#include <cuda_runtime.h>
#include <cstddef>

#include "dualquant.cuh"
#include "metadata.hh"
#include "type_aliasing.hh"

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

namespace kernel = cusz::predictor_quantizer;

// v2 ////////////////////////////////////////////////////////////

template <typename Data, typename Quant>
__global__ void kernel::c_lorenzo_1d1l(lorenzo_zip ctx, Data* d, Quant* q)
{
    static const auto Block = MetadataTrait<1>::Block;
    Data(&s1df)[Block]      = *reinterpret_cast<Data(*)[Block]>(&scratch);

    auto id = bix * bdx + tix;

    if (id < ctx.d0) {
        // prequant (fp presence)
        s1df[tix] = round(d[id] * ctx.ebx2_r);
        __syncthreads();  // necessary to ensure correctness
        // postquant
        Data pred = tix == 0 ? 0 : s1df[tix - 1];
        __syncthreads();

        Data delta       = s1df[tix] - pred;
        bool quantizable = fabs(delta) < ctx.radius;
        Data candidate   = delta + ctx.radius;
        d[id]            = (1 - quantizable) * candidate;  // output; reuse data for outlier
        q[id]            = quantizable * static_cast<Quant>(candidate);
    }
}

template <typename Data, typename Quant>
__global__ void kernel::c_lorenzo_2d1l(lorenzo_zip ctx, Data* d, Quant* q)
{
    static const auto Block   = MetadataTrait<2>::Block;
    Data(&s2df)[Block][Block] = *reinterpret_cast<Data(*)[Block][Block]>(&scratch);

    auto y = tiy, x = tix;
    auto gi1 = biy * bdy + y, gi0 = bix * bdx + x;

    if (gi0 < ctx.d0 and gi1 < ctx.d1) {
        size_t id = gi0 + gi1 * ctx.stride1;  // low to high dim, inner to outer

        // prequant (fp presence)
        s2df[y][x] = round(d[id] * ctx.ebx2_r);
        __syncthreads();  // necessary to ensure correctness

        Data delta       = s2df[y][x] - ((x > 0 ? s2df[y][x - 1] : 0) +                // dist=1
                                   (y > 0 ? s2df[y - 1][x] : 0) -                // dist=1
                                   (x > 0 and y > 0 ? s2df[y - 1][x - 1] : 0));  // dist=2
        bool quantizable = fabs(delta) < ctx.radius;
        Data candidate   = delta + ctx.radius;
        d[id]            = (1 - quantizable) * candidate;  // output; reuse data for outlier
        q[id]            = quantizable * static_cast<Quant>(candidate);
    }
}

template <typename Data, typename Quant>
__global__ void kernel::c_lorenzo_3d1l(lorenzo_zip ctx, Data* d, Quant* q)
{
    static const auto Block          = MetadataTrait<3>::Block;
    Data(&s3df)[Block][Block][Block] = *reinterpret_cast<Data(*)[Block][Block][Block]>(&scratch);

    auto z = tiz, y = tiy, x = tix;
    auto gi2 = biz * bdz + z, gi1 = biy * bdy + y, gi0 = bix * bdx + x;

    if (gi0 < ctx.d0 and gi1 < ctx.d1 and gi2 < ctx.d2) {
        size_t id = gi0 + gi1 * ctx.stride1 + gi2 * ctx.stride2;  // low to high in dim, inner to outer

        // prequant (fp presence)
        s3df[z][y][x] = round(d[id] * ctx.ebx2_r);
        __syncthreads();  // necessary to ensure correctness

        Data delta       = s3df[z][y][x] - ((z > 0 and y > 0 and x > 0 ? s3df[z - 1][y - 1][x - 1] : 0)  // dist=3
                                      - (y > 0 and x > 0 ? s3df[z][y - 1][x - 1] : 0)              // dist=2
                                      - (z > 0 and x > 0 ? s3df[z - 1][y][x - 1] : 0)              //
                                      - (z > 0 and y > 0 ? s3df[z - 1][y - 1][x] : 0)              //
                                      + (x > 0 ? s3df[z][y][x - 1] : 0)                            // dist=1
                                      + (y > 0 ? s3df[z][y - 1][x] : 0)                            //
                                      + (z > 0 ? s3df[z - 1][y][x] : 0));                          //
        bool quantizable = fabs(delta) < ctx.radius;
        Data candidate   = delta + ctx.radius;
        d[id]            = (1 - quantizable) * candidate;  // output; reuse data for outlier
        q[id]            = quantizable * static_cast<Quant>(candidate);
    }
}

template <typename Data, typename Quant>
__global__ void kernel::x_lorenzo_1d1l(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* q)
{
    static const auto Block = MetadataTrait<1>::Block;
    Data(&buffer)[Block]    = *reinterpret_cast<Data(*)[Block]>(&scratch);

    auto id     = bix * bdx + tix;
    auto radius = static_cast<Data>(ctx.radius);

    if (id < ctx.d0)
        buffer[tix] = outlier[id] + static_cast<Data>(q[id]) - radius;  // fuse
    else
        buffer[tix] = 0;
    __syncthreads();

    for (auto d = 1; d < Block; d *= 2) {
        Data n = 0;
        if (tix >= d) n = buffer[tix - d];  // like __shfl_up_sync(0x1f, var, d); warp_sync
        __syncthreads();
        if (tix >= d) buffer[tix] += n;
        __syncthreads();
    }

    if (id < ctx.d0) { data[id] = buffer[tix] * ctx.ebx2; }
    __syncthreads();
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
__global__ void kernel::x_lorenzo_2d1l(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* q)
{
    static const auto Block     = MetadataTrait<2>::Block;
    Data(&buffer)[Block][Block] = *reinterpret_cast<Data(*)[Block][Block]>(&scratch);

    auto   gi1 = biy * bdy + tiy, gi0 = bix * bdx + tix;
    size_t id     = gi0 + gi1 * ctx.stride1;
    auto   radius = static_cast<Data>(ctx.radius);

    if (gi0 < ctx.d0 and gi1 < ctx.d1)
        buffer[tiy][tix] = outlier[id] + static_cast<Data>(q[id]) - radius;  // fuse
    else
        buffer[tiy][tix] = 0;
    __syncthreads();

    for (auto d = 1; d < Block; d *= 2) {
        Data n = 0;
        if (tix >= d) n = buffer[tiy][tix - d];
        __syncthreads();
        if (tix >= d) buffer[tiy][tix] += n;
        __syncthreads();
    }

    for (auto d = 1; d < Block; d *= 2) {
        Data n = 0;
        if (tiy >= d) n = buffer[tiy - d][tix];
        __syncthreads();
        if (tiy >= d) buffer[tiy][tix] += n;
        __syncthreads();
    }

    if (gi0 < ctx.d0 and gi1 < ctx.d1) { data[id] = buffer[tiy][tix] * ctx.ebx2; }
    __syncthreads();
}

template <typename Data, typename Quant>
__global__ void kernel::x_lorenzo_2d1l_16x16_v0(lorenzo_unzip ctx, Data* xdata, Data* outlier, Quant* quant)
{
    static const auto Block = MetadataTrait<2>::Block;
    static_assert(Block == 16, "In one case, we need Block for 2D == 16");

    Data thread_scope[Block];
    //     ------> gi0 (x)
    //  |   t0    t1    t2    t3
    //  |  ts0_0 ts0_0 ts0_0 ts0_0
    //  |  ts0_1 ts0_1 ts0_1 ts0_1
    // gi1 ts0_2 ts0_2 ts0_2 ts0_2
    // (y)  |     |     |     |
    //     ts0_f ts0_f ts0_f ts0_f

    auto gi0      = bix * bdx + tix;
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
            Data n = __shfl_up_sync(0xf, i, d);
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
__global__ void kernel::x_lorenzo_2d1l_16x16_v1(lorenzo_unzip ctx, Data* xdata, Data* outlier, Quant* quant)
{
    static const auto Block          = MetadataTrait<2>::Block;
    static const auto YSequentiality = Block / 2;  // sequentiality in y direction
    static_assert(Block == 16, "In one case, we need Block for 2D == 16");

    __shared__ Data intermediate[Block];
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

    auto gi0      = bix * bdx + tix;                     // bdx == 16
    auto gi1_base = biy * Block + tiy * YSequentiality;  // bdy * YSequentiality = Block == 16
    auto radius   = static_cast<Data>(ctx.radius);

    auto get_gid = [&](auto i) { return (gi1_base + i) * ctx.stride1 + gi0; };

#pragma unroll
    for (auto i = 0; i < YSequentiality; i++) {
        auto gid = get_gid(i);

        if (gi0 < ctx.d0 and gi1_base + i < ctx.d1)
            thread_scope[i] = outlier[gid] + static_cast<Data>(quant[gid]) - radius;  // fuse
        else
            thread_scope[i] = 0;
        __syncthreads();
    }

    // sequential partial-sum
    for (auto i = 1; i < YSequentiality; i++) thread_scope[i] += thread_scope[i - 1];
    __syncthreads();
    if (tiy == 0) intermediate[tix] = thread_scope[YSequentiality - 1];
    __syncthreads();
    if (tiy == 1) {
        auto tmp = intermediate[tix];
#pragma unroll
        for (auto& i : thread_scope) i += tmp;
    }
    __syncthreads();

    // shuffle
#pragma unroll
    for (auto& i : thread_scope) {
        for (auto d = 1; d < Block; d *= 2) {
            Data n = __shfl_up_sync(0xf, i, d);
            if (tix >= d) i += n;
        }
        i *= ctx.ebx2;
    }

#pragma unroll
    for (auto i = 0; i < YSequentiality; i++) {
        auto gid = get_gid(i);
        if (gi0 < ctx.d0 and gi1_base + i < ctx.d1) xdata[gid] = thread_scope[i];
    }
    __syncthreads();
}

template <typename Data, typename Quant>
__global__ void kernel::x_lorenzo_3d1l(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* q)
{
    static const auto Block            = MetadataTrait<3>::Block;
    Data(&buffer)[Block][Block][Block] = *reinterpret_cast<Data(*)[Block][Block][Block]>(&scratch);

    auto   gi2 = biz * bdz + tiz, gi1 = biy * bdy + tiy, gi0 = bix * bdx + tix;
    size_t id     = gi0 + gi1 * ctx.stride1 + gi2 * ctx.stride2;  // low to high in dim, inner to outer
    auto   radius = static_cast<Data>(ctx.radius);

    if (gi0 < ctx.d0 and gi1 < ctx.d1 and gi2 < ctx.d2)
        buffer[tiz][tiy][tix] = outlier[id] + static_cast<Data>(q[id]) - radius;  // id
    else
        buffer[tiz][tiy][tix] = 0;
    __syncthreads();

    for (auto d = 1; d < Block; d *= 2) {
        Data n = 0;
        if (tix >= d) n = buffer[tiz][tiy][tix - d];
        __syncthreads();
        if (tix >= d) buffer[tiz][tiy][tix] += n;
        __syncthreads();
    }

    for (auto d = 1; d < Block; d *= 2) {
        Data n = 0;
        if (tiy >= d) n = buffer[tiz][tiy - d][tix];
        __syncthreads();
        if (tiy >= d) buffer[tiz][tiy][tix] += n;
        __syncthreads();
    }

    for (auto d = 1; d < Block; d *= 2) {
        Data n = 0;
        if (tiz >= d) n = buffer[tiz - d][tiy][tix];
        __syncthreads();
        if (tiz >= d) buffer[tiz][tiy][tix] += n;
        __syncthreads();
    }

    if (gi0 < ctx.d0 and gi1 < ctx.d1 and gi2 < ctx.d2) { data[id] = buffer[tiz][tiy][tix] * ctx.ebx2; }
    __syncthreads();
}

template __global__ void kernel::c_lorenzo_1d1l<FP4, UI1>(lorenzo_zip, FP4*, UI1*);
template __global__ void kernel::c_lorenzo_1d1l<FP4, UI2>(lorenzo_zip, FP4*, UI2*);
template __global__ void kernel::c_lorenzo_2d1l<FP4, UI1>(lorenzo_zip, FP4*, UI1*);
template __global__ void kernel::c_lorenzo_2d1l<FP4, UI2>(lorenzo_zip, FP4*, UI2*);
template __global__ void kernel::c_lorenzo_3d1l<FP4, UI1>(lorenzo_zip, FP4*, UI1*);
template __global__ void kernel::c_lorenzo_3d1l<FP4, UI2>(lorenzo_zip, FP4*, UI2*);

template __global__ void kernel::x_lorenzo_1d1l<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel::x_lorenzo_1d1l<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);
template __global__ void kernel::x_lorenzo_1d1l_cub<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel::x_lorenzo_1d1l_cub<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);
template __global__ void kernel::x_lorenzo_2d1l<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel::x_lorenzo_2d1l<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);
template __global__ void kernel::x_lorenzo_2d1l_16x16_v0<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel::x_lorenzo_2d1l_16x16_v0<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);
template __global__ void kernel::x_lorenzo_2d1l_16x16_v1<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel::x_lorenzo_2d1l_16x16_v1<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);
template __global__ void kernel::x_lorenzo_3d1l<FP4, UI1>(lorenzo_unzip, FP4*, FP4*, UI1*);
template __global__ void kernel::x_lorenzo_3d1l<FP4, UI2>(lorenzo_unzip, FP4*, FP4*, UI2*);