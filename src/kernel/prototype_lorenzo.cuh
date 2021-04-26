/**
 * @file prototype_lorenzo.cuh
 * @author Jiannan Tian
 * @brief (prototype) Dual-Quant Lorenzo method.
 * @version 0.2
 * @date 2021-01-16
 * (create) 19-09-23; (release) 2020-09-20; (rev1) 2021-01-16; (rev2) 2021-02-20; (rev3) 2021-04-11
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_DUALQUANT_CUH
#define CUSZ_DUALQUANT_CUH

#include <cuda_runtime.h>
#include <cstddef>

#if CUDART_VERSION >= 11000
#pragma message(__FILE__ ": (CUDA 11 onward), cub from system path")
#include <cub/cub.cuh>
#else
#pragma message(__FILE__ ": (CUDA 10 or earlier), cub from git submodule")
#include "../../external/cub/cub/cub.cuh"
#endif

#include <cuda_runtime.h>
#include <cstddef>

#include "../metadata.hh"
#include "../type_aliasing.hh"

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
namespace prototype_kernel { // easy algorithmic description
template <typename Data, typename Quant> __global__ void c_lorenzo_1d1l(lorenzo_zip, Data*, Quant*);
template <typename Data, typename Quant> __global__ void c_lorenzo_2d1l(lorenzo_zip, Data*, Quant*);
template <typename Data, typename Quant> __global__ void c_lorenzo_3d1l(lorenzo_zip, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_1d1l(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_2d1l(lorenzo_unzip, Data*, Data*, Quant*);
template <typename Data, typename Quant> __global__ void x_lorenzo_3d1l(lorenzo_unzip, Data*, Data*, Quant*); 
}
// clang-format on

template <typename Data, typename Quant>
__global__ void prototype_kernel::c_lorenzo_1d1l(lorenzo_zip ctx, Data* d, Quant* q)
{
    static const auto Block = MetadataTrait<1>::Block;
    Data(&s1df)[Block]      = *reinterpret_cast<Data(*)[Block]>(&scratch);

    auto id = bix * bdx + tix;
    if (id < ctx.d0) {
        s1df[tix] = round(d[id] * ctx.ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();    // necessary to ensure correctness
    if (id < ctx.d0) {  // postquant
        Data delta       = s1df[tix] - (tix == 0 ? 0 : s1df[tix - 1]);
        bool quantizable = fabs(delta) < ctx.radius;
        Data candidate   = delta + ctx.radius;
        d[id]            = (1 - quantizable) * candidate;  // output; reuse data for outlier
        q[id]            = quantizable * static_cast<Quant>(candidate);
    }
}

template <typename Data, typename Quant>
__global__ void prototype_kernel::c_lorenzo_2d1l(lorenzo_zip ctx, Data* d, Quant* q)
{
    static const auto Block   = MetadataTrait<2>::Block;
    Data(&s2df)[Block][Block] = *reinterpret_cast<Data(*)[Block][Block]>(&scratch);

    auto y = tiy, x = tix;
    auto gi1 = biy * bdy + y, gi0 = bix * bdx + x;

    auto id = gi0 + gi1 * ctx.stride1;  // low to high dim, inner to outer
    if (gi0 < ctx.d0 and gi1 < ctx.d1) {
        s2df[y][x] = round(d[id] * ctx.ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness
    if (gi0 < ctx.d0 and gi1 < ctx.d1) {
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
__global__ void prototype_kernel::c_lorenzo_3d1l(lorenzo_zip ctx, Data* d, Quant* q)
{
    static const auto Block          = MetadataTrait<3>::Block;
    Data(&s3df)[Block][Block][Block] = *reinterpret_cast<Data(*)[Block][Block][Block]>(&scratch);

    auto z = tiz, y = tiy, x = tix;
    auto gi2 = biz * bdz + z, gi1 = biy * bdy + y, gi0 = bix * bdx + x;

    auto id = gi0 + gi1 * ctx.stride1 + gi2 * ctx.stride2;  // low to high in dim, inner to outer
    if (gi0 < ctx.d0 and gi1 < ctx.d1 and gi2 < ctx.d2) {
        s3df[z][y][x] = round(d[id] * ctx.ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness
    if (gi0 < ctx.d0 and gi1 < ctx.d1 and gi2 < ctx.d2) {
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
__global__ void prototype_kernel::x_lorenzo_1d1l(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* q)
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
__global__ void prototype_kernel::x_lorenzo_3d1l(lorenzo_unzip ctx, Data* data, Data* outlier, Quant* q)
{
    static const auto Block            = MetadataTrait<3>::Block;
    Data(&buffer)[Block][Block][Block] = *reinterpret_cast<Data(*)[Block][Block][Block]>(&scratch);

    auto   gi2 = biz * Block + tiz, gi1 = biy * Block + tiy, gi0 = bix * Block + tix;
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

#endif
