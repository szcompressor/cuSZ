/**
 * @file prediction_config.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-06-10
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef PREDICTION_CONFIG_CUH
#define PREDICTION_CONFIG_CUH

#include <cuda_runtime.h>

#include "../common.hh"
#include "../utils.hh"
#include "lorenzo.cuh"
#include "spline3.cuh"

template <typename T, typename E, typename FP, bool NO_R_SEPARATE>
void launch_construct_LorenzoI(
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

    constexpr auto SUBLEN_1D = dim3(256, 1, 1);
    constexpr auto SEQ_1D    = dim3(4, 1, 1);  // x-sequentiality == 4
    constexpr auto BLOCK_1D  = dim3(256 / 4, 1, 1);
    auto           GRID_1D   = pardeg3(len3, SUBLEN_1D);

    constexpr auto SUBLEN_2D = dim3(16, 16, 1);
    constexpr auto SEQ_2D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_2D  = dim3(16, 2, 1);
    auto           GRID_2D   = pardeg3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    constexpr auto SEQ_3D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_3D  = dim3(32, 1, 8);
    auto           GRID_3D   = pardeg3(len3, SUBLEN_3D);

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    auto outlier = data;

    cuda_timer_t timer;
    timer.timer_start(stream);

    if (ndim() == 1) {
        ::cusz::c_lorenzo_1d1l<T, E, FP, SUBLEN_1D.x, SEQ_1D.x, NO_R_SEPARATE>
            <<<GRID_1D, BLOCK_1D, 0, stream>>>(data, errctrl, outlier, len3, leap3, radius, ebx2_r);
    }
    else if (ndim() == 2) {
        ::cusz::c_lorenzo_2d1l_16x16data_mapto16x2<T, E, FP>
            <<<GRID_2D, BLOCK_2D, 0, stream>>>(data, errctrl, outlier, len3, leap3, radius, ebx2_r);
    }
    else if (ndim() == 3) {
        ::cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, E, FP>
            <<<GRID_3D, BLOCK_3D, 0, stream>>>(data, errctrl, outlier, len3, leap3, radius, ebx2_r);
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
void launch_reconstruct_LorenzoI(
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

    constexpr auto SUBLEN_1D = dim3(256, 1, 1);
    constexpr auto SEQ_1D    = dim3(8, 1, 1);  // x-sequentiality == 8
    constexpr auto BLOCK_1D  = dim3(256 / 8, 1, 1);
    auto           GRID_1D   = pardeg3(len3, SUBLEN_1D);

    constexpr auto SUBLEN_2D = dim3(16, 16, 1);
    constexpr auto SEQ_2D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_2D  = dim3(16, 2, 1);
    auto           GRID_2D   = pardeg3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    constexpr auto SEQ_3D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_3D  = dim3(32, 1, 8);
    auto           GRID_3D   = pardeg3(len3, SUBLEN_3D);

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    auto outlier = xdata;

    cuda_timer_t timer;
    timer.timer_start(stream);

    if (ndim() == 1) {
        ::cusz::x_lorenzo_1d1l<T, E, FP, SUBLEN_1D.x, SEQ_1D.x>
            <<<GRID_1D, BLOCK_1D, 0, stream>>>(outlier, errctrl, xdata, len3, leap3, radius, ebx2);
    }
    else if (ndim() == 2) {
        ::cusz::x_lorenzo_2d1l_16x16data_mapto16x2<T, E, FP>
            <<<GRID_2D, BLOCK_2D, 0, stream>>>(outlier, errctrl, xdata, len3, leap3, radius, ebx2);
    }
    else if (ndim() == 3) {
        ::cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, E, FP>
            <<<GRID_3D, BLOCK_3D, 0, stream>>>(outlier, errctrl, xdata, len3, leap3, radius, ebx2);
    }

    timer.timer_end(stream);
    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    time_elapsed = timer.get_time_elapsed();
}

template <typename T, typename E, typename FP, bool NO_R_SEPARATE>
void launch_construct_Spline3(
    T*           data,
    dim3 const   len3,
    T*           anchor,
    dim3 const   an_len3,
    E*           errctrl,
    dim3 const   ec_len3,
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

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    constexpr auto SEQ_3D    = dim3(1, 8, 1);
    constexpr auto BLOCK_3D  = dim3(256, 1, 1);
    auto           GRID_3D   = pardeg3(len3, SUBLEN_3D);

    {
        constexpr auto SUBLEN_TOTAL = SUBLEN_3D.x * SUBLEN_3D.y * SUBLEN_3D.z;
        constexpr auto SEQ_TOTAL    = SEQ_3D.x * SEQ_3D.y * SEQ_3D.z;
        constexpr auto BLOCK_TOTAL  = BLOCK_3D.x * BLOCK_3D.y * BLOCK_3D.z;

        // static_assert(SUBLEN_TOTAL / SEQ_TOTAL == BLOCK_TOTAL, "parallelism does not match!");
        if (SUBLEN_TOTAL / SEQ_TOTAL != BLOCK_TOTAL) throw std::runtime_error("parallelism does not match!");
    }

    ////////////////////////////////////////

    auto ebx2     = eb * 2;
    auto eb_r     = 1 / eb;
    auto leap3    = dim3(1, len3.x, len3.x * len3.y);
    auto ec_leap3 = dim3(1, ec_len3.x, ec_len3.x * ec_len3.y);
    auto an_leap3 = dim3(1, an_len3.x, an_len3.x * an_len3.y);

    cuda_timer_t timer;
    timer.timer_start();

    if (ndim() == 1) {  //
        throw std::runtime_error("Spline1 not implemented");
    }
    else if (ndim() == 2) {
        throw std::runtime_error("Spline2 not implemented");
    }
    else if (ndim() == 3) {
        cusz::c_spline3d_infprecis_32x8x8data<T*, E*, float, 256, false>  //
            <<<GRID_3D, BLOCK_3D, 0, stream>>>                            //
            (data, len3, leap3,                                           //
             errctrl, ec_len3, ec_leap3,                                  //
             anchor, an_leap3,                                            //
             eb_r, ebx2, radius);
    }

    timer.timer_end();

    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    time_elapsed = timer.get_time_elapsed();
}

template <typename T, typename E, typename FP>
void launch_reconstruct_Spline3(
    T*           xdata,
    dim3 const   len3,
    T*           anchor,
    dim3 const   an_len3,
    E*           errctrl,
    dim3 const   ec_len3,
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

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    constexpr auto SEQ_3D    = dim3(1, 8, 1);
    constexpr auto BLOCK_3D  = dim3(256, 1, 1);
    auto           GRID_3D   = pardeg3(len3, SUBLEN_3D);

    {
        constexpr auto SUBLEN_TOTAL = SUBLEN_3D.x * SUBLEN_3D.y * SUBLEN_3D.z;
        constexpr auto SEQ_TOTAL    = SEQ_3D.x * SEQ_3D.y * SEQ_3D.z;
        constexpr auto BLOCK_TOTAL  = BLOCK_3D.x * BLOCK_3D.y * BLOCK_3D.z;

        // static_assert(SUBLEN_TOTAL / SEQ_TOTAL == BLOCK_TOTAL, "parallelism does not match!");
        if (SUBLEN_TOTAL / SEQ_TOTAL != BLOCK_TOTAL) throw std::runtime_error("parallelism does not match!");
    }

    ////////////////////////////////////////

    auto ebx2     = eb * 2;
    auto eb_r     = 1 / eb;
    auto leap3    = dim3(1, len3.x, len3.x * len3.y);
    auto ec_leap3 = dim3(1, ec_len3.x, ec_len3.x * ec_len3.y);
    auto an_leap3 = dim3(1, an_len3.x, an_len3.x * an_len3.y);

    cuda_timer_t timer;
    timer.timer_start();

    cusz::x_spline3d_infprecis_32x8x8data<E*, T*, float, 256>  //
        <<<GRID_3D, BLOCK_3D, 0, stream>>>                     //
        (errctrl, ec_len3, ec_leap3,                           //
         anchor, an_len3, an_leap3,                            //
         xdata, len3, leap3,                                   //
         eb_r, ebx2, radius);

    timer.timer_end();

    if (stream)
        CHECK_CUDA(cudaStreamSynchronize(stream));
    else
        CHECK_CUDA(cudaDeviceSynchronize());

    time_elapsed = timer.get_time_elapsed();
}

#endif
