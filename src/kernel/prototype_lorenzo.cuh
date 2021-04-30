/**
 * @file prototype_lorenzo.cuh
 * @author Jiannan Tian
 * @brief (prototype) Dual-Quant Lorenzo method.
 * @version 0.2
 * @date 2021-01-16
 * (create) 2019-09-23; (release) 2020-09-20; (rev1) 2021-01-16; (rev2) 2021-02-20; (rev3) 2021-04-11
 * (rev4) 2021-04-30
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef KERNEL_PROTOTYPE_LORENZO_CUH
#define KERNEL_PROTOTYPE_LORENZO_CUH

#if CUDART_VERSION >= 11000
#pragma message(__FILE__ ": (CUDA 11 onward), cub from system path")
#include <cub/cub.cuh>
#else
#pragma message(__FILE__ ": (CUDA 10 or earlier), cub from git submodule")
#include "../../external/cub/cub/cub.cuh"
#endif

#include <cstddef>

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

// TODO disabling dynamic shmem alloction results in wrong number
extern __shared__ char scratch[];

using DIM    = unsigned int;
using STRIDE = unsigned int;

namespace prototype_kernel {  // easy algorithmic description

// clang-format off
template <typename Data, typename Quant, typename FP, int Block = 256, bool ProbePredError = true> __global__ void c_lorenzo_1d1l
(Data*, Quant*, DIM, int, FP, int* = nullptr, Data* = nullptr, FP = 1.0);
template <typename Data, typename Quant, typename FP, int Block = 256> __global__ void x_lorenzo_1d1l
(Data*, Data*, Quant*, DIM, int, FP);

template <typename Data, typename Quant, typename FP, int Block = 16, bool ProbePredError = true> __global__ void c_lorenzo_2d1l
(Data*, Quant*, DIM, DIM, STRIDE, int, FP, int* = nullptr, Data* = nullptr, FP = 1.0);
template <typename Data, typename Quant, typename FP, int Block = 16> __global__ void x_lorenzo_2d1l
(Data*, Data*, Quant*, DIM, DIM, STRIDE, int, FP);

template <typename Data, typename Quant, typename FP, int Block = 8, bool ProbePredError = true> __global__ void c_lorenzo_3d1l
(Data*, Quant*, DIM, DIM, DIM, STRIDE, STRIDE, int, FP, int* = nullptr, Data* = nullptr, FP = 1.0);
template <typename Data, typename Quant, typename FP, int Block = 8> __global__ void x_lorenzo_3d1l
(Data*, Data*, Quant*, DIM, DIM, DIM, STRIDE, STRIDE, int, FP);
// clang-format on

}  // namespace prototype_kernel

template <typename Data, typename Quant, typename FP, int Block = 256, bool ProbePredError = true>
__global__ void prototype_kernel::c_lorenzo_1d1l(  //
    Data*  data,
    Quant* quant,
    DIM    dimx,
    int    radius,
    FP     ebx2_r,
    int*   integer_error,
    Data*  raw_error,
    FP     ebx2)
{
    Data(&shmem)[Block] = *reinterpret_cast<Data(*)[Block]>(&scratch);

    auto id = bix * bdx + tix;
    if (id < dimx) {
        shmem[tix] = round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness

    Data delta = shmem[tix] - (tix == 0 ? 0 : shmem[tix - 1]);

    if CONSTEXPR (ProbePredError) {
        if (id < dimx) {  // postquant
            integer_error[id] = delta;
            raw_error[id]     = delta * ebx2;
        }
        return;
    }

    {
        bool quantizable = fabs(delta) < radius;
        Data candidate   = delta + radius;
        if (id < dimx) {                                // postquant
            data[id]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
            quant[id] = quantizable * static_cast<Quant>(candidate);
        }
    }
    // EOF
}

template <typename Data, typename Quant, typename FP, int Block = 16, bool ProbePredError = true>
__global__ void prototype_kernel::c_lorenzo_2d1l(  //
    Data*  data,
    Quant* quant,
    DIM    dimx,
    DIM    dimy,
    STRIDE stridey,
    int    radius,
    FP     ebx2_r,
    int*   integer_error,
    Data*  raw_error,
    FP     ebx2)
{
    Data(&shmem)[Block][Block] = *reinterpret_cast<Data(*)[Block][Block]>(&scratch);

    auto y = tiy, x = tix;
    auto giy = biy * bdy + y, gix = bix * bdx + x;

    auto id = gix + giy * stridey;  // low to high dim, inner to outer
    if (gix < dimx and giy < dimy) {
        shmem[y][x] = round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness

    Data delta = shmem[y][x] - ((x > 0 ? shmem[y][x - 1] : 0) +                // dist=1
                                (y > 0 ? shmem[y - 1][x] : 0) -                // dist=1
                                (x > 0 and y > 0 ? shmem[y - 1][x - 1] : 0));  // dist=2

    if CONSTEXPR (ProbePredError) {
        if (gix < dimx and giy < dimy) {
            integer_error[id] = static_cast<int>(delta);
            raw_error[id]     = delta * ebx2;
        }
        return;
    }

    {
        bool quantizable = fabs(delta) < radius;
        Data candidate   = delta + radius;
        if (gix < dimx and giy < dimy) {
            data[id]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
            quant[id] = quantizable * static_cast<Quant>(candidate);
        }
    }
    // EOF
}

template <typename Data, typename Quant, typename FP, int Block = 8, bool ProbePredError = true>
__global__ void prototype_kernel::c_lorenzo_3d1l(  //
    Data*  data,
    Quant* quant,
    DIM    dimx,
    DIM    dimy,
    DIM    dimz,
    STRIDE stridey,
    STRIDE stridez,
    int    radius,
    FP     ebx2_r,
    int*   integer_error,
    Data*  raw_error,
    FP     ebx2)
{
    Data(&shmem)[Block][Block][Block] = *reinterpret_cast<Data(*)[Block][Block][Block]>(&scratch);

    auto z = tiz, y = tiy, x = tix;
    auto giz = biz * bdz + z, giy = biy * bdy + y, gix = bix * bdx + x;

    auto id = gix + giy * stridey + giz * stridez;  // low to high in dim, inner to outer
    if (gix < dimx and giy < dimy and giz < dimz) {
        shmem[z][y][x] = round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness

    Data delta = shmem[z][y][x] - ((z > 0 and y > 0 and x > 0 ? shmem[z - 1][y - 1][x - 1] : 0)  // dist=3
                                   - (y > 0 and x > 0 ? shmem[z][y - 1][x - 1] : 0)              // dist=2
                                   - (z > 0 and x > 0 ? shmem[z - 1][y][x - 1] : 0)              //
                                   - (z > 0 and y > 0 ? shmem[z - 1][y - 1][x] : 0)              //
                                   + (x > 0 ? shmem[z][y][x - 1] : 0)                            // dist=1
                                   + (y > 0 ? shmem[z][y - 1][x] : 0)                            //
                                   + (z > 0 ? shmem[z - 1][y][x] : 0));                          //

    if CONSTEXPR (ProbePredError) {
        if (gix < dimx and giy < dimy and giz < dimz) {
            integer_error[id] = static_cast<int>(delta);
            raw_error[id]     = delta * ebx2;
        }
        return;
    }

    {
        bool quantizable = fabs(delta) < radius;
        Data candidate   = delta + radius;
        if (gix < dimx and giy < dimy and giz < dimz) {
            data[id]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
            quant[id] = quantizable * static_cast<Quant>(candidate);
        }
    }
    // EOF
}

template <typename Data, typename Quant, typename FP, int Block>
__global__ void prototype_kernel::x_lorenzo_1d1l(  //
    Data*  data,
    Data*  outlier,
    Quant* quant,
    DIM    dimx,
    int    radius,
    FP     ebx2)
{
    Data(&shmem)[Block] = *reinterpret_cast<Data(*)[Block]>(&scratch);

    auto id = bix * bdx + tix;

    if (id < dimx)
        shmem[tix] = outlier[id] + static_cast<Data>(quant[id]) - radius;  // fuse
    else
        shmem[tix] = 0;
    __syncthreads();

    for (auto d = 1; d < Block; d *= 2) {
        Data n = 0;
        if (tix >= d) n = shmem[tix - d];  // like __shfl_up_sync(0x1f, var, d); warp_sync
        __syncthreads();
        if (tix >= d) shmem[tix] += n;
        __syncthreads();
    }

    if (id < dimx) { data[id] = shmem[tix] * ebx2; }
    __syncthreads();
}

template <typename Data, typename Quant, typename FP, int Block>
__global__ void prototype_kernel::x_lorenzo_2d1l(  //
    Data*  data,
    Data*  outlier,
    Quant* quant,
    DIM    dimx,
    DIM    dimy,
    STRIDE stridey,
    int    radius,
    FP     ebx2)
{
    Data(&shmem)[Block][Block] = *reinterpret_cast<Data(*)[Block][Block]>(&scratch);

    auto   giy = biy * bdy + tiy, gix = bix * bdx + tix;
    size_t id = gix + giy * stridey;

    if (gix < dimx and giy < dimy)
        shmem[tiy][tix] = outlier[id] + static_cast<Data>(quant[id]) - radius;  // fuse
    else
        shmem[tiy][tix] = 0;
    __syncthreads();

    for (auto d = 1; d < Block; d *= 2) {
        Data n = 0;
        if (tix >= d) n = shmem[tiy][tix - d];
        __syncthreads();
        if (tix >= d) shmem[tiy][tix] += n;
        __syncthreads();
    }

    for (auto d = 1; d < Block; d *= 2) {
        Data n = 0;
        if (tiy >= d) n = shmem[tiy - d][tix];
        __syncthreads();
        if (tiy >= d) shmem[tiy][tix] += n;
        __syncthreads();
    }

    if (gix < dimx and giy < dimy) { data[id] = shmem[tiy][tix] * ebx2; }
    __syncthreads();
}

template <typename Data, typename Quant, typename FP, int Block>
__global__ void prototype_kernel::x_lorenzo_3d1l(  //
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
    Data(&shmem)[Block][Block][Block] = *reinterpret_cast<Data(*)[Block][Block][Block]>(&scratch);

    auto   giz = biz * Block + tiz, giy = biy * Block + tiy, gix = bix * Block + tix;
    size_t id = gix + giy * stridey + giz * stridez;  // low to high in dim, inner to outer

    if (gix < dimx and giy < dimy and giz < dimz)
        shmem[tiz][tiy][tix] = outlier[id] + static_cast<Data>(quant[id]) - radius;  // id
    else
        shmem[tiz][tiy][tix] = 0;
    __syncthreads();

    for (auto dist = 1; dist < Block; dist *= 2) {
        Data addend = 0;
        if (tix >= dist) addend = shmem[tiz][tiy][tix - dist];
        __syncthreads();
        if (tix >= dist) shmem[tiz][tiy][tix] += addend;
        __syncthreads();
    }

    for (auto dist = 1; dist < Block; dist *= 2) {
        Data addend = 0;
        if (tiy >= dist) addend = shmem[tiz][tiy - dist][tix];
        __syncthreads();
        if (tiy >= dist) shmem[tiz][tiy][tix] += addend;
        __syncthreads();
    }

    for (auto dist = 1; dist < Block; dist *= 2) {
        Data addend = 0;
        if (tiz >= dist) addend = shmem[tiz - dist][tiy][tix];
        __syncthreads();
        if (tiz >= dist) shmem[tiz][tiy][tix] += addend;
        __syncthreads();
    }

    if (gix < dimx and giy < dimy and giz < dimz) { data[id] = shmem[tiz][tiy][tix] * ebx2; }
    __syncthreads();
}

#endif
