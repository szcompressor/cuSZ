/**
 * @file launch_prototype.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-09-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef BAA68178_B2F5_4C58_AB0A_A29731EBB9B6
#define BAA68178_B2F5_4C58_AB0A_A29731EBB9B6

#include <stdexcept>
#include "../utils/cuda_err.cuh"
#include "../utils/cuda_mem.cuh"
#include "../utils/timer.hh"
#include "lorenzo_prototype.cuh"

template <typename T, typename E, typename FP>
void launch_construct_LorenzoI_proto(
    T* const     data,
    dim3 const   len3,
    T* const     anchor,
    dim3 const   placeholder_1,
    E* const     errctrl,
    dim3 const   placeholder_2,
    double const eb,
    int const    radius,
    float&       time_elapsed,
    cudaStream_t stream)
{
    auto pardeg3 = [](dim3 len, dim3 sublen) {
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
    auto           GRID_1D   = pardeg3(len3, SUBLEN_1D);

    constexpr auto SUBLEN_2D = dim3(16, 16, 1);
    constexpr auto BLOCK_2D  = dim3(16, 16, 1);
    auto           GRID_2D   = pardeg3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(8, 8, 8);
    constexpr auto BLOCK_3D  = dim3(8, 8, 8);
    auto           GRID_3D   = pardeg3(len3, SUBLEN_3D);

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    // auto outlier = data;

    cuda_timer_t timer;
    timer.timer_start(stream);

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

    timer.timer_end(stream);
    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    time_elapsed = timer.get_time_elapsed();
}

template <typename T, typename E, typename FP>
void launch_reconstruct_LorenzoI_proto(
    T*           xdata,
    dim3 const   len3,
    T*           anchor,
    dim3 const   placeholder_1,
    E*           errctrl,
    dim3 const   placeholder_2,
    double const eb,
    int const    radius,
    float&       time_elapsed,
    cudaStream_t stream)
{
    auto pardeg3 = [](dim3 len, dim3 sublen) {
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
    auto           GRID_1D   = pardeg3(len3, SUBLEN_1D);

    constexpr auto SUBLEN_2D = dim3(16, 16, 1);
    constexpr auto BLOCK_2D  = dim3(16, 16, 1);
    auto           GRID_2D   = pardeg3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(8, 8, 8);
    constexpr auto BLOCK_3D  = dim3(8, 8, 8);
    auto           GRID_3D   = pardeg3(len3, SUBLEN_3D);

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    auto outlier = xdata;

    cuda_timer_t timer;
    timer.timer_start(stream);

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

    timer.timer_end(stream);
    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    time_elapsed = timer.get_time_elapsed();
}

#endif /* BAA68178_B2F5_4C58_AB0A_A29731EBB9B6 */
