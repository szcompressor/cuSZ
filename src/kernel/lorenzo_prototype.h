/**
 * @file lorenzo_prototype.h
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

#ifndef CUSZ_LORENZO_PROTOTYPE_H
#define CUSZ_LORENZO_PROTOTYPE_H

#if CUDART_VERSION >= 11000
#pragma message(__FILE__ ": (CUDA 11 onward), cub from system path")
#include <cub/cub.cuh>
#else
#pragma message(__FILE__ ": (CUDA 10 or earlier), cub from git submodule")
#include "../../external/cub/cub/cub.cuh"
#endif

#include <cstddef>

#define TIX threadIdx.x
#define TIY threadIdx.y
#define TIZ threadIdx.z
#define BIX blockIdx.x
#define BIY blockIdx.y
#define BIZ blockIdx.z
#define BDX blockDim.x
#define BDY blockDim.y
#define BDZ blockDim.z

// TODO disabling dynamic shmem alloction results in wrong number
extern __shared__ char scratch[];

using DIM    = unsigned int;
using STRIDE = unsigned int;

namespace cusz {
namespace prototype {  // easy algorithmic description

// clang-format off
template <typename Data, typename Quant, typename FP, int BLOCK = 256, bool PROBE_PRED_ERROR = true> __global__ void c_lorenzo_1d1l
(Data*, Quant*, DIM, int, FP, int* = nullptr, Data* = nullptr, FP = 1.0);
template <typename Data, typename Quant, typename FP, int BLOCK = 256> __global__ void x_lorenzo_1d1l
(Data*, Data*, Quant*, DIM, int, FP);

template <typename Data, typename Quant, typename FP, int BLOCK = 16, bool PROBE_PRED_ERROR = true> __global__ void c_lorenzo_2d1l
(Data*, Quant*, DIM, DIM, STRIDE, int, FP, int* = nullptr, Data* = nullptr, FP = 1.0);
template <typename Data, typename Quant, typename FP, int BLOCK = 16> __global__ void x_lorenzo_2d1l
(Data*, Data*, Quant*, DIM, DIM, STRIDE, int, FP);

template <typename Data, typename Quant, typename FP, int BLOCK = 8, bool PROBE_PRED_ERROR = true> __global__ void c_lorenzo_3d1l
(Data*, Quant*, DIM, DIM, DIM, STRIDE, STRIDE, int, FP, int* = nullptr, Data* = nullptr, FP = 1.0);
template <typename Data, typename Quant, typename FP, int BLOCK = 8> __global__ void x_lorenzo_3d1l
(Data*, Data*, Quant*, DIM, DIM, DIM, STRIDE, STRIDE, int, FP);
// clang-format on

}  // namespace prototype
}  // namespace cusz

template <typename Data, typename Quant, typename FP, int BLOCK = 256, bool PROBE_PRED_ERROR = true>
__global__ void cusz::prototype::c_lorenzo_1d1l(  //
    Data*  data,
    Quant* quant,
    DIM    dimx,
    int    radius,
    FP     ebx2_r,
    int*   integer_error,
    Data*  raw_error,
    FP     ebx2)
{
    Data(&shmem)[BLOCK] = *reinterpret_cast<Data(*)[BLOCK]>(&scratch);

    auto id = BIX * BDX + TIX;
    if (id < dimx) {
        shmem[TIX] = round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness

    Data delta = shmem[TIX] - (TIX == 0 ? 0 : shmem[TIX - 1]);

    if CONSTEXPR (PROBE_PRED_ERROR) {
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

template <typename Data, typename Quant, typename FP, int BLOCK = 16, bool PROBE_PRED_ERROR = true>
__global__ void cusz::prototype::c_lorenzo_2d1l(  //
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
    Data(&shmem)[BLOCK][BLOCK] = *reinterpret_cast<Data(*)[BLOCK][BLOCK]>(&scratch);

    auto y = TIY, x = TIX;
    auto giy = BIY * BDY + y, gix = BIX * BDX + x;

    auto id = gix + giy * stridey;  // low to high dim, inner to outer
    if (gix < dimx and giy < dimy) {
        shmem[y][x] = round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness

    Data delta = shmem[y][x] - ((x > 0 ? shmem[y][x - 1] : 0) +                // dist=1
                                (y > 0 ? shmem[y - 1][x] : 0) -                // dist=1
                                (x > 0 and y > 0 ? shmem[y - 1][x - 1] : 0));  // dist=2

    if CONSTEXPR (PROBE_PRED_ERROR) {
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

template <typename Data, typename Quant, typename FP, int BLOCK = 8, bool PROBE_PRED_ERROR = true>
__global__ void cusz::prototype::c_lorenzo_3d1l(  //
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
    Data(&shmem)[BLOCK][BLOCK][BLOCK] = *reinterpret_cast<Data(*)[BLOCK][BLOCK][BLOCK]>(&scratch);

    auto z = TIZ, y = TIY, x = TIX;
    auto giz = BIZ * BDZ + z, giy = BIY * BDY + y, gix = BIX * BDX + x;

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

    if CONSTEXPR (PROBE_PRED_ERROR) {
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

template <typename Data, typename Quant, typename FP, int BLOCK>
__global__ void cusz::prototype::x_lorenzo_1d1l(  //
    Data*  data,
    Data*  outlier,
    Quant* quant,
    DIM    dimx,
    int    radius,
    FP     ebx2)
{
    Data(&shmem)[BLOCK] = *reinterpret_cast<Data(*)[BLOCK]>(&scratch);

    auto id = BIX * BDX + TIX;

    if (id < dimx)
        shmem[TIX] = outlier[id] + static_cast<Data>(quant[id]) - radius;  // fuse
    else
        shmem[TIX] = 0;
    __syncthreads();

    for (auto d = 1; d < BLOCK; d *= 2) {
        Data n = 0;
        if (TIX >= d) n = shmem[TIX - d];  // like __shfl_up_sync(0x1f, var, d); warp_sync
        __syncthreads();
        if (TIX >= d) shmem[TIX] += n;
        __syncthreads();
    }

    if (id < dimx) { data[id] = shmem[TIX] * ebx2; }
    __syncthreads();
}

template <typename Data, typename Quant, typename FP, int BLOCK>
__global__ void cusz::prototype::x_lorenzo_2d1l(  //
    Data*  data,
    Data*  outlier,
    Quant* quant,
    DIM    dimx,
    DIM    dimy,
    STRIDE stridey,
    int    radius,
    FP     ebx2)
{
    Data(&shmem)[BLOCK][BLOCK] = *reinterpret_cast<Data(*)[BLOCK][BLOCK]>(&scratch);

    auto   giy = BIY * BDY + TIY, gix = BIX * BDX + TIX;
    size_t id = gix + giy * stridey;

    if (gix < dimx and giy < dimy)
        shmem[TIY][TIX] = outlier[id] + static_cast<Data>(quant[id]) - radius;  // fuse
    else
        shmem[TIY][TIX] = 0;
    __syncthreads();

    for (auto d = 1; d < BLOCK; d *= 2) {
        Data n = 0;
        if (TIX >= d) n = shmem[TIY][TIX - d];
        __syncthreads();
        if (TIX >= d) shmem[TIY][TIX] += n;
        __syncthreads();
    }

    for (auto d = 1; d < BLOCK; d *= 2) {
        Data n = 0;
        if (TIY >= d) n = shmem[TIY - d][TIX];
        __syncthreads();
        if (TIY >= d) shmem[TIY][TIX] += n;
        __syncthreads();
    }

    if (gix < dimx and giy < dimy) { data[id] = shmem[TIY][TIX] * ebx2; }
    __syncthreads();
}

template <typename Data, typename Quant, typename FP, int BLOCK>
__global__ void cusz::prototype::x_lorenzo_3d1l(  //
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
    Data(&shmem)[BLOCK][BLOCK][BLOCK] = *reinterpret_cast<Data(*)[BLOCK][BLOCK][BLOCK]>(&scratch);

    auto   giz = BIZ * BLOCK + TIZ, giy = BIY * BLOCK + TIY, gix = BIX * BLOCK + TIX;
    size_t id = gix + giy * stridey + giz * stridez;  // low to high in dim, inner to outer

    if (gix < dimx and giy < dimy and giz < dimz)
        shmem[TIZ][TIY][TIX] = outlier[id] + static_cast<Data>(quant[id]) - radius;  // id
    else
        shmem[TIZ][TIY][TIX] = 0;
    __syncthreads();

    for (auto dist = 1; dist < BLOCK; dist *= 2) {
        Data addend = 0;
        if (TIX >= dist) addend = shmem[TIZ][TIY][TIX - dist];
        __syncthreads();
        if (TIX >= dist) shmem[TIZ][TIY][TIX] += addend;
        __syncthreads();
    }

    for (auto dist = 1; dist < BLOCK; dist *= 2) {
        Data addend = 0;
        if (TIY >= dist) addend = shmem[TIZ][TIY - dist][TIX];
        __syncthreads();
        if (TIY >= dist) shmem[TIZ][TIY][TIX] += addend;
        __syncthreads();
    }

    for (auto dist = 1; dist < BLOCK; dist *= 2) {
        Data addend = 0;
        if (TIZ >= dist) addend = shmem[TIZ - dist][TIY][TIX];
        __syncthreads();
        if (TIZ >= dist) shmem[TIZ][TIY][TIX] += addend;
        __syncthreads();
    }

    if (gix < dimx and giy < dimy and giz < dimz) { data[id] = shmem[TIZ][TIY][TIX] * ebx2; }
    __syncthreads();
}

#endif
