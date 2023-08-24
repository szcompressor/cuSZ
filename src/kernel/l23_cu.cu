/**
 * @file lorenzo.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-01
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "cusz/type.h"
#include "utils/err.hh"
#include "utils/timer.h"

#include "kernel/l23.hh"
// #include "detail/l21.inl"
#include "detail/l23.inl"

template <typename T, typename EQ, typename FP>
cusz_error_status psz_comp_l23(
    T* const     data,
    dim3 const   len3,
    double const eb,
    int const    radius,
    EQ* const    eq,
    T* const     outlier,
    float*       time_elapsed,
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
    constexpr auto SEQ_1D    = 4;  // x-sequentiality == 4
    constexpr auto BLOCK_1D  = dim3(256 / 4, 1, 1);
    auto           GRID_1D   = divide3(len3, SUBLEN_1D);

    constexpr auto SUBLEN_2D = dim3(16, 16, 1);
    // constexpr auto SEQ_2D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_2D = dim3(16, 2, 1);
    auto           GRID_2D  = divide3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    // constexpr auto SEQ_3D    = dim3(1, 8, 1);  // y-sequentiality == 8
    // constexpr auto BLOCK_3D = dim3(32, 1, 8);  // for v0
    constexpr auto BLOCK_3D = dim3(32, 8, 1);  // for v0::r1_shfl
    auto           GRID_3D  = divide3(len3, SUBLEN_3D);

    auto d = ndim();

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream);

    if (d == 1) {
        //::cusz::c_lorenzo_1d1l<T, EQ, FP, SUBLEN_1D, SEQ_1D>
        //<<<GRID_1D, BLOCK_1D, 0, stream>>>(data, eq, outlier, len3, leap3, radius, ebx2_r);

        psz::cuda::__kernel::v0::c_lorenzo_1d1l<T, EQ, FP, SUBLEN_1D, SEQ_1D>
            <<<GRID_1D, BLOCK_1D, 0, stream>>>(data, len3, leap3, radius, ebx2_r, eq, outlier);
    }
    else if (d == 2) {
        //::cusz::c_lorenzo_2d1l_16x16data_mapto16x2<T, EQ, FP>
        //<<<GRID_2D, BLOCK_2D, 0, stream>>>(data, eq, outlier, len3, leap3, radius, ebx2_r);
        psz::cuda::__kernel::v0::c_lorenzo_2d1l<T, EQ, FP>
            <<<GRID_2D, BLOCK_2D, 0, stream>>>(data, len3, leap3, radius, ebx2_r, eq, outlier);
    }
    else if (d == 3) {
        //::cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, EQ, FP>
        //<<<GRID_3D, BLOCK_3D, 0, stream>>>(data, eq, outlier, len3, leap3, radius, ebx2_r);
        psz::cuda::__kernel::v0::c_lorenzo_3d1l<T, EQ, FP>
            <<<GRID_3D, BLOCK_3D, 0, stream>>>(data, len3, leap3, radius, ebx2_r, eq, outlier);
    }

    STOP_CUDAEVENT_RECORDING(stream);
    CHECK_GPU(cudaStreamSynchronize(stream));
    TIME_ELAPSED_CUDAEVENT(time_elapsed);
    DESTROY_CUDAEVENT_PAIR;

    return CUSZ_SUCCESS;
}

template <typename T, typename EQ, typename FP>
cusz_error_status psz_decomp_l23(
    EQ*          eq,
    dim3 const   len3,
    T*           outlier,
    double const eb,
    int const    radius,
    T*           xdata,
    float*       time_elapsed,
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
    constexpr auto SEQ_1D    = 8;  // x-sequentiality == 8
    constexpr auto BLOCK_1D  = dim3(256 / 8, 1, 1);
    auto           GRID_1D   = divide3(len3, SUBLEN_1D);

    constexpr auto SUBLEN_2D = dim3(16, 16, 1);
    // constexpr auto SEQ_2D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_2D = dim3(16, 2, 1);
    auto           GRID_2D  = divide3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    // constexpr auto SEQ_3D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_3D = dim3(32, 1, 8);
    auto           GRID_3D  = divide3(len3, SUBLEN_3D);

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    auto d = ndim();

    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream);

    if (d == 1) {
        //::cusz::x_lorenzo_1d1l<T, EQ, FP, SUBLEN_1D, SEQ_1D>
        //<<<GRID_1D, BLOCK_1D, 0, stream>>>(outlier, eq, xdata, len3, leap3, radius, ebx2);
        psz::cuda::__kernel::v0::x_lorenzo_1d1l<T, EQ, FP, SUBLEN_1D, SEQ_1D>
            <<<GRID_1D, BLOCK_1D, 0, stream>>>(eq, outlier, len3, leap3, radius, ebx2, xdata);
    }
    else if (d == 2) {
        //::cusz::x_lorenzo_2d1l_16x16data_mapto16x2<T, EQ, FP>
        //<<<GRID_2D, BLOCK_2D, 0, stream>>>(outlier, eq, xdata, len3, leap3, radius, ebx2);
        psz::cuda::__kernel::v0::x_lorenzo_2d1l<T, EQ, FP>
            <<<GRID_2D, BLOCK_2D, 0, stream>>>(eq, outlier, len3, leap3, radius, ebx2, xdata);
    }
    else if (d == 3) {
        //::cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, EQ, FP>
        //<<<GRID_3D, BLOCK_3D, 0, stream>>>(outlier, eq, xdata, len3, leap3, radius, ebx2);
        psz::cuda::__kernel::v0::x_lorenzo_3d1l<T, EQ, FP>
            <<<GRID_3D, BLOCK_3D, 0, stream>>>(eq, outlier, len3, leap3, radius, ebx2, xdata);
    }

    STOP_CUDAEVENT_RECORDING(stream);
    CHECK_GPU(cudaStreamSynchronize(stream));
    TIME_ELAPSED_CUDAEVENT(time_elapsed);
    DESTROY_CUDAEVENT_PAIR;

    return CUSZ_SUCCESS;
}

#define CPP_INS(T, EQ)                                                                                          \
    template cusz_error_status psz_comp_l23<T, EQ>(                                                             \
        T* const data, dim3 const len3, double const eb, int const radius, EQ* const eq, T* const outlier,      \
        float* time_elapsed, cudaStream_t stream);                                                              \
                                                                                                                \
    template cusz_error_status psz_decomp_l23<T, EQ>(                                                           \
        EQ * eq, dim3 const len3, T* outlier, double const eb, int const radius, T* xdata, float* time_elapsed, \
        cudaStream_t stream);

CPP_INS(float, uint8_t);
CPP_INS(float, uint16_t);
CPP_INS(float, uint32_t);
CPP_INS(float, float);

CPP_INS(double, uint8_t);
CPP_INS(double, uint16_t);
CPP_INS(double, uint32_t);
CPP_INS(double, float);

CPP_INS(float, int32_t);
CPP_INS(double, int32_t);

#undef CPP_INS
