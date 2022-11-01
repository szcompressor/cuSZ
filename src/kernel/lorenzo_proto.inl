/**
 * @file lorenzo_prototype.cuh
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

#ifndef CUSZ_KERNEL_LORENZO_PROTOTYPE_CUH
#define CUSZ_KERNEL_LORENZO_PROTOTYPE_CUH

#include <cstddef>
#include <stdexcept>

// TODO disabling dynamic shmem alloction results in wrong number
// extern __shared__ char scratch[];

using DIM    = unsigned int;
using STRIDE = unsigned int;

#include "utils/cuda_err.cuh"
#include "utils/timer.h"

namespace cusz {
namespace prototype {  // easy algorithmic description

template <typename Data, typename Quant, typename FP, int BLOCK = 256, bool PROBE_PRED_ERROR = false>
__global__ void c_lorenzo_1d1l(  //
    Data*  data,
    Quant* quant,
    dim3   len3,
    dim3   stride3,
    int    radius,
    FP     ebx2_r,
    int*   integer_error = nullptr,
    Data*  raw_error     = nullptr,
    FP     ebx2          = 1.0)
{
    __shared__ Data shmem[BLOCK];

    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < len3.x) {
        shmem[threadIdx.x] = round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness

    Data delta = shmem[threadIdx.x] - (threadIdx.x == 0 ? 0 : shmem[threadIdx.x - 1]);

    // #ifndef DPCPP_SHOWCASE
    //     if CONSTEXPR (PROBE_PRED_ERROR) {
    //         if (id < len3.x) {  // postquant
    //             integer_error[id] = delta;
    //             raw_error[id]     = delta * ebx2;
    //         }
    //         return;
    //     }
    // #endif

    {
        bool quantizable = fabs(delta) < radius;
        Data candidate   = delta + radius;
        if (id < len3.x) {                              // postquant
            data[id]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
            quant[id] = quantizable * static_cast<Quant>(candidate);
        }
    }
    // EOF
}

template <typename Data, typename Quant, typename FP, int BLOCK = 16, bool PROBE_PRED_ERROR = false>
__global__ void c_lorenzo_2d1l(  //
    Data*  data,
    Quant* quant,
    dim3   len3,
    dim3   stride3,
    int    radius,
    FP     ebx2_r,
    int*   integer_error = nullptr,
    Data*  raw_error     = nullptr,
    FP     ebx2          = 1.0)
{
    __shared__ Data shmem[BLOCK][BLOCK + 1];

    auto y = threadIdx.y, x = threadIdx.x;
    auto giy = blockIdx.y * blockDim.y + y, gix = blockIdx.x * blockDim.x + x;

    auto id = gix + giy * stride3.y;  // low to high dim, inner to outer
    if (gix < len3.x and giy < len3.y) {
        shmem[y][x] = round(data[id] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();  // necessary to ensure correctness

    Data delta = shmem[y][x] - ((x > 0 ? shmem[y][x - 1] : 0) +                // dist=1
                                (y > 0 ? shmem[y - 1][x] : 0) -                // dist=1
                                (x > 0 and y > 0 ? shmem[y - 1][x - 1] : 0));  // dist=2

    // #ifndef DPCPP_SHOWCASE
    //     if CONSTEXPR (PROBE_PRED_ERROR) {
    //         if (gix < len3.x and giy < len3.y) {
    //             integer_error[id] = static_cast<int>(delta);
    //             raw_error[id]     = delta * ebx2;
    //         }
    //         return;
    //     }
    // #endif

    {
        bool quantizable = fabs(delta) < radius;
        Data candidate   = delta + radius;
        if (gix < len3.x and giy < len3.y) {
            data[id]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
            quant[id] = quantizable * static_cast<Quant>(candidate);
        }
    }
    // EOF
}

template <typename Data, typename Quant, typename FP, int BLOCK = 8, bool PROBE_PRED_ERROR = false>
__global__ void c_lorenzo_3d1l(  //
    Data*  data,
    Quant* quant,
    dim3   len3,
    dim3   stride3,
    int    radius,
    FP     ebx2_r,
    int*   integer_error = nullptr,
    Data*  raw_error     = nullptr,
    FP     ebx2          = 1.0)
{
    __shared__ Data shmem[BLOCK][BLOCK][BLOCK + 1];

    auto z = threadIdx.z, y = threadIdx.y, x = threadIdx.x;
    auto giz = blockIdx.z * blockDim.z + z, giy = blockIdx.y * blockDim.y + y, gix = blockIdx.x * blockDim.x + x;

    auto id = gix + giy * stride3.y + giz * stride3.z;  // low to high in dim, inner to outer
    if (gix < len3.x and giy < len3.y and giz < len3.z) {
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

    // #ifndef DPCPP_SHOWCASE
    //     if CONSTEXPR (PROBE_PRED_ERROR) {
    //         if (gix < len3.x and giy < len3.y and giz < len3.z) {
    //             integer_error[id] = static_cast<int>(delta);
    //             raw_error[id]     = delta * ebx2;
    //         }
    //         return;
    //     }
    // #endif

    {
        bool quantizable = fabs(delta) < radius;
        Data candidate   = delta + radius;
        if (gix < len3.x and giy < len3.y and giz < len3.z) {
            data[id]  = (1 - quantizable) * candidate;  // output; reuse data for outlier
            quant[id] = quantizable * static_cast<Quant>(candidate);
        }
    }
    // EOF
}

template <typename Data, typename Quant, typename FP, int BLOCK = 256>
__global__ void x_lorenzo_1d1l(  //
    Data*  xdata_outlier,
    Quant* quant,
    dim3   len3,
    dim3   stride3,
    int    radius,
    FP     ebx2)
{
    __shared__ Data shmem[BLOCK];

    auto id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < len3.x)
        shmem[threadIdx.x] = xdata_outlier[id] + static_cast<Data>(quant[id]) - radius;  // fuse
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

    if (id < len3.x) { xdata_outlier[id] = shmem[threadIdx.x] * ebx2; }
}

template <typename Data, typename Quant, typename FP, int BLOCK = 16>
__global__ void x_lorenzo_2d1l(  //
    Data*  xdata_outlier,
    Quant* quant,
    dim3   len3,
    dim3   stride3,
    int    radius,
    FP     ebx2)
{
    __shared__ Data shmem[BLOCK][BLOCK + 1];

    auto   giy = blockIdx.y * blockDim.y + threadIdx.y, gix = blockIdx.x * blockDim.x + threadIdx.x;
    size_t id = gix + giy * stride3.y;

    if (gix < len3.x and giy < len3.y)
        shmem[threadIdx.y][threadIdx.x] = xdata_outlier[id] + static_cast<Data>(quant[id]) - radius;  // fuse
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

    if (gix < len3.x and giy < len3.y) { xdata_outlier[id] = shmem[threadIdx.y][threadIdx.x] * ebx2; }
}

template <typename Data, typename Quant, typename FP, int BLOCK = 8>
__global__ void x_lorenzo_3d1l(  //
    Data*  xdata_outlier,
    Quant* quant,
    dim3   len3,
    dim3   stride3,
    int    radius,
    FP     ebx2)
{
    __shared__ Data shmem[BLOCK][BLOCK][BLOCK + 1];

    auto giz = blockIdx.z * BLOCK + threadIdx.z, giy = blockIdx.y * BLOCK + threadIdx.y,
         gix  = blockIdx.x * BLOCK + threadIdx.x;
    size_t id = gix + giy * stride3.y + giz * stride3.z;  // low to high in dim, inner to outer

    if (gix < len3.x and giy < len3.y and giz < len3.z)
        shmem[threadIdx.z][threadIdx.y][threadIdx.x] = xdata_outlier[id] + static_cast<Data>(quant[id]) - radius;  // id
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

    if (gix < len3.x and giy < len3.y and giz < len3.z) {
        xdata_outlier[id] = shmem[threadIdx.z][threadIdx.y][threadIdx.x] * ebx2;
    }
}

}  // namespace prototype
}  // namespace cusz

template <typename T, typename E, typename FP>
void launch_construct_LorenzoI_proto(
    T* const     data,
    dim3 const   len3,
    T* const     anchor,
    dim3 const   placeholder_1,
    E* const     errctrl,
    dim3 const   placeholder_2,
    T*           outlier,
    double const eb,
    int const    radius,
    float&       time_elapsed,
    cudaStream_t stream)
{
    auto divide3 = [](dim3 len, dim3 sublen) {
        return dim3(
            (len.x - 1) / sublen.x + 1,  //
            (len.y - 1) / sublen.y + 1,  //
            (len.z - 1) / sublen.z + 1);
    };

    auto ndim = [&]() {
        if (len3.z == 1 and len3.y == 1)
            return 1;
        else if (len3.z == 1 and len3.y != 1)
            return 2;
        else
            return 3;
    };

    constexpr auto SUBLEN_1D = 256;
    constexpr auto BLOCK_1D  = dim3(256, 1, 1);
    auto           GRID_1D   = divide3(len3, SUBLEN_1D);

    constexpr auto SUBLEN_2D = dim3(16, 16, 1);
    constexpr auto BLOCK_2D  = dim3(16, 16, 1);
    auto           GRID_2D   = divide3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(8, 8, 8);
    constexpr auto BLOCK_3D  = dim3(8, 8, 8);
    auto           GRID_3D   = divide3(len3, SUBLEN_3D);

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream);

    if (ndim() == 1) {
        cusz::prototype::c_lorenzo_1d1l<T, E, FP>
            <<<GRID_1D, BLOCK_1D, 0, stream>>>(data, errctrl, len3, leap3, radius, ebx2_r);
    }
    else if (ndim() == 2) {
        cusz::prototype::c_lorenzo_2d1l<T, E, FP>
            <<<GRID_2D, BLOCK_2D, 0, stream>>>(data, errctrl, len3, leap3, radius, ebx2_r);
    }
    else if (ndim() == 3) {
        cusz::prototype::c_lorenzo_3d1l<T, E, FP>
            <<<GRID_3D, BLOCK_3D, 0, stream>>>(data, errctrl, len3, leap3, radius, ebx2_r);
    }
    else {
        throw std::runtime_error("Lorenzo only works for 123-D.");
    }

    STOP_CUDAEVENT_RECORDING(stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    TIME_ELAPSED_CUDAEVENT(&time_elapsed);
    DESTROY_CUDAEVENT_PAIR;
}

template <typename T, typename E, typename FP>
void launch_reconstruct_LorenzoI_proto(
    T*           xdata,
    dim3 const   len3,
    T*           anchor,
    dim3 const   placeholder_1,
    E*           errctrl,
    dim3 const   placeholder_2,
    T*           outlier,
    double const eb,
    int const    radius,
    float&       time_elapsed,
    cudaStream_t stream)
{
    auto divide3 = [](dim3 len, dim3 sublen) {
        return dim3(
            (len.x - 1) / sublen.x + 1,  //
            (len.y - 1) / sublen.y + 1,  //
            (len.z - 1) / sublen.z + 1);
    };

    auto ndim = [&]() {
        if (len3.z == 1 and len3.y == 1)
            return 1;
        else if (len3.z == 1 and len3.y != 1)
            return 2;
        else
            return 3;
    };

    constexpr auto SUBLEN_1D = 256;
    constexpr auto BLOCK_1D  = dim3(256, 1, 1);
    auto           GRID_1D   = divide3(len3, SUBLEN_1D);

    constexpr auto SUBLEN_2D = dim3(16, 16, 1);
    constexpr auto BLOCK_2D  = dim3(16, 16, 1);
    auto           GRID_2D   = divide3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(8, 8, 8);
    constexpr auto BLOCK_3D  = dim3(8, 8, 8);
    auto           GRID_3D   = divide3(len3, SUBLEN_3D);

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream);

    if (ndim() == 1) {
        cusz::prototype::x_lorenzo_1d1l<T, E, FP>
            <<<GRID_1D, BLOCK_1D, 0, stream>>>(outlier, errctrl, len3, leap3, radius, ebx2);
    }
    else if (ndim() == 2) {
        cusz::prototype::x_lorenzo_2d1l<T, E, FP>
            <<<GRID_2D, BLOCK_2D, 0, stream>>>(outlier, errctrl, len3, leap3, radius, ebx2);
    }
    else if (ndim() == 3) {
        cusz::prototype::x_lorenzo_3d1l<T, E, FP>
            <<<GRID_3D, BLOCK_3D, 0, stream>>>(outlier, errctrl, len3, leap3, radius, ebx2);
    }

    STOP_CUDAEVENT_RECORDING(stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    TIME_ELAPSED_CUDAEVENT(&time_elapsed);
    DESTROY_CUDAEVENT_PAIR;
}

#endif
