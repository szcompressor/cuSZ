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

#include <cstddef>

// TODO disabling dynamic shmem alloction results in wrong number
// extern __shared__ char scratch[];

using DIM    = unsigned int;
using STRIDE = unsigned int;

namespace cusz {
namespace prototype {  // easy algorithmic description

// clang-format off
template <typename Data, typename Quant, typename FP, int BLOCK = 256, bool PROBE_PRED_ERROR = false> __global__ void c_lorenzo_1d1l
(Data*, Quant*, DIM, int, FP, int* = nullptr, Data* = nullptr, FP = 1.0);
template <typename Data, typename Quant, typename FP, int BLOCK = 256> __global__ void x_lorenzo_1d1l
(Data*, Data*, Quant*, DIM, int, FP);

template <typename Data, typename Quant, typename FP, int BLOCK = 16, bool PROBE_PRED_ERROR = false> __global__ void c_lorenzo_2d1l
(Data*, Quant*, DIM, DIM, STRIDE, int, FP, int* = nullptr, Data* = nullptr, FP = 1.0);
template <typename Data, typename Quant, typename FP, int BLOCK = 16> __global__ void x_lorenzo_2d1l
(Data*, Data*, Quant*, DIM, DIM, STRIDE, int, FP);

template <typename Data, typename Quant, typename FP, int BLOCK = 8, bool PROBE_PRED_ERROR = false> __global__ void c_lorenzo_3d1l
(Data*, Quant*, DIM, DIM, DIM, STRIDE, STRIDE, int, FP, int* = nullptr, Data* = nullptr, FP = 1.0);
template <typename Data, typename Quant, typename FP, int BLOCK = 8> __global__ void x_lorenzo_3d1l
(Data*, Data*, Quant*, DIM, DIM, DIM, STRIDE, STRIDE, int, FP);
// clang-format on

}  // namespace prototype
}  // namespace cusz

template <typename Data, typename Quant, typename FP, int BLOCK, bool PROBE_PRED_ERROR>
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
    // Data(&shmem)[BLOCK] = *reinterpret_cast<Data(*)[BLOCK]>(&scratch);
    __shared__ Data shmem[BLOCK];

    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < dimx) {
        shmem[threadIdx.x] = round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness

    Data delta = shmem[threadIdx.x] - (threadIdx.x == 0 ? 0 : shmem[threadIdx.x - 1]);

#ifndef DPCPP_SHOWCASE
    if CONSTEXPR (PROBE_PRED_ERROR) {
        if (id < dimx) {  // postquant
            integer_error[id] = delta;
            raw_error[id]     = delta * ebx2;
        }
        return;
    }
#endif

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

template <typename Data, typename Quant, typename FP, int BLOCK, bool PROBE_PRED_ERROR>
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
    // Data(&shmem)[BLOCK][BLOCK] = *reinterpret_cast<Data(*)[BLOCK][BLOCK]>(&scratch);
    __shared__ Data shmem[BLOCK][BLOCK];

    auto y = threadIdx.y, x = threadIdx.x;
    auto giy = blockIdx.y * blockDim.y + y, gix = blockIdx.x * blockDim.x + x;

    auto id = gix + giy * stridey;  // low to high dim, inner to outer
    if (gix < dimx and giy < dimy) {
        shmem[y][x] = round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness

    Data delta = shmem[y][x] - ((x > 0 ? shmem[y][x - 1] : 0) +                // dist=1
                                (y > 0 ? shmem[y - 1][x] : 0) -                // dist=1
                                (x > 0 and y > 0 ? shmem[y - 1][x - 1] : 0));  // dist=2

#ifndef DPCPP_SHOWCASE
    if CONSTEXPR (PROBE_PRED_ERROR) {
        if (gix < dimx and giy < dimy) {
            integer_error[id] = static_cast<int>(delta);
            raw_error[id]     = delta * ebx2;
        }
        return;
    }
#endif

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

template <typename Data, typename Quant, typename FP, int BLOCK, bool PROBE_PRED_ERROR>
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
    // Data(&shmem)[BLOCK][BLOCK][BLOCK] = *reinterpret_cast<Data(*)[BLOCK][BLOCK][BLOCK]>(&scratch);
    __shared__ Data shmem[BLOCK][BLOCK][BLOCK];

    auto z = threadIdx.z, y = threadIdx.y, x = threadIdx.x;
    auto giz = blockIdx.z * blockDim.z + z, giy = blockIdx.y * blockDim.y + y, gix = blockIdx.x * blockDim.x + x;

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

#ifndef DPCPP_SHOWCASE
    if CONSTEXPR (PROBE_PRED_ERROR) {
        if (gix < dimx and giy < dimy and giz < dimz) {
            integer_error[id] = static_cast<int>(delta);
            raw_error[id]     = delta * ebx2;
        }
        return;
    }
#endif

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
    // Data(&shmem)[BLOCK] = *reinterpret_cast<Data(*)[BLOCK]>(&scratch);
    __shared__ Data shmem[BLOCK];

    auto id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < dimx)
        shmem[threadIdx.x] = outlier[id] + static_cast<Data>(quant[id]) - radius;  // fuse
    else
        shmem[threadIdx.x] = 0;
    __syncthreads();

    for (auto d = 1; d < BLOCK; d *= 2) {
        Data n = 0;
        if (threadIdx.x >= d) n = shmem[threadIdx.x - d];  // like __shfl_up_sync(0x1f, var, d); warp_sync
        __syncthreads();
        if (threadIdx.x >= d) shmem[threadIdx.x] += n;
        __syncthreads();
    }

    if (id < dimx) { data[id] = shmem[threadIdx.x] * ebx2; }
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
    // Data(&shmem)[BLOCK][BLOCK] = *reinterpret_cast<Data(*)[BLOCK][BLOCK]>(&scratch);
    __shared__ Data shmem[BLOCK][BLOCK];

    auto   giy = blockIdx.y * blockDim.y + threadIdx.y, gix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id = gix + giy * stridey;

    if (gix < dimx and giy < dimy)
        shmem[threadIdx.y][threadIdx.x] = outlier[id] + static_cast<Data>(quant[id]) - radius;  // fuse
    else
        shmem[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    for (auto d = 1; d < BLOCK; d *= 2) {
        Data n = 0;
        if (threadIdx.x >= d) n = shmem[threadIdx.y][threadIdx.x - d];
        __syncthreads();
        if (threadIdx.x >= d) shmem[threadIdx.y][threadIdx.x] += n;
        __syncthreads();
    }

    for (auto d = 1; d < BLOCK; d *= 2) {
        Data n = 0;
        if (threadIdx.y >= d) n = shmem[threadIdx.y - d][threadIdx.x];
        __syncthreads();
        if (threadIdx.y >= d) shmem[threadIdx.y][threadIdx.x] += n;
        __syncthreads();
    }

    if (gix < dimx and giy < dimy) { data[id] = shmem[threadIdx.y][threadIdx.x] * ebx2; }
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
    // Data(&shmem)[BLOCK][BLOCK][BLOCK] = *reinterpret_cast<Data(*)[BLOCK][BLOCK][BLOCK]>(&scratch);
    __shared__ Data shmem[BLOCK][BLOCK][BLOCK];

    auto giz = blockIdx.z * BLOCK + threadIdx.z, giy = blockIdx.y * BLOCK + threadIdx.y,
         gix  = blockIdx.x * BLOCK + threadIdx.x;
    size_t id = gix + giy * stridey + giz * stridez;  // low to high in dim, inner to outer

    if (gix < dimx and giy < dimy and giz < dimz)
        shmem[threadIdx.z][threadIdx.y][threadIdx.x] = outlier[id] + static_cast<Data>(quant[id]) - radius;  // id
    else
        shmem[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    for (auto dist = 1; dist < BLOCK; dist *= 2) {
        Data addend = 0;
        if (threadIdx.x >= dist) addend = shmem[threadIdx.z][threadIdx.y][threadIdx.x - dist];
        __syncthreads();
        if (threadIdx.x >= dist) shmem[threadIdx.z][threadIdx.y][threadIdx.x] += addend;
        __syncthreads();
    }

    for (auto dist = 1; dist < BLOCK; dist *= 2) {
        Data addend = 0;
        if (threadIdx.y >= dist) addend = shmem[threadIdx.z][threadIdx.y - dist][threadIdx.x];
        __syncthreads();
        if (threadIdx.y >= dist) shmem[threadIdx.z][threadIdx.y][threadIdx.x] += addend;
        __syncthreads();
    }

    for (auto dist = 1; dist < BLOCK; dist *= 2) {
        Data addend = 0;
        if (threadIdx.z >= dist) addend = shmem[threadIdx.z - dist][threadIdx.y][threadIdx.x];
        __syncthreads();
        if (threadIdx.z >= dist) shmem[threadIdx.z][threadIdx.y][threadIdx.x] += addend;
        __syncthreads();
    }

    if (gix < dimx and giy < dimy and giz < dimz) { data[id] = shmem[threadIdx.z][threadIdx.y][threadIdx.x] * ebx2; }
    __syncthreads();
}

#endif
