/**
 * @file lorenzo_var.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-27
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "cusz/type.h"
#include "kernel/l23.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

#include "detail/lorenzo_var.inl"

template <typename T, typename DeltaT, typename FP>
psz_error_status asz::experimental::psz_comp_l21var(
    T*           data,
    dim3 const   len3,
    double const eb,
    DeltaT*      delta,
    bool*        signum,
    float*       time_elapsed,
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
    constexpr auto SEQ_1D    = 4;  // x-sequentiality == 4
    constexpr auto BLOCK_1D  = dim3(256 / 4, 1, 1);
    auto           GRID_1D   = pardeg3(len3, SUBLEN_1D);

    constexpr auto SUBLEN_2D = dim3(16, 16, 1);
    // constexpr auto SEQ_2D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_2D = dim3(16, 2, 1);
    auto           GRID_2D  = pardeg3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    // constexpr auto SEQ_3D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_3D = dim3(32, 1, 8);
    auto           GRID_3D  = pardeg3(len3, SUBLEN_3D);

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING(stream);

    if (ndim() == 1) {
        cusz::experimental::c_lorenzo_1d1l<T, DeltaT, FP, SEQ_1D, SEQ_1D>  //
            <<<GRID_1D, BLOCK_1D, 0, stream>>>                             //
            (data, delta, signum, len3, leap3, ebx2_r);
    }
    else if (ndim() == 2) {
        cusz::experimental::c_lorenzo_2d1l_16x16data_mapto16x2<T, DeltaT, FP>  //
            <<<GRID_2D, BLOCK_2D, 0, stream>>>                                 //
            (data, delta, signum, len3, leap3, ebx2_r);
    }
    else if (ndim() == 3) {
        cusz::experimental::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, DeltaT, FP>  //
            <<<GRID_3D, BLOCK_3D, 0, stream>>>                                    //
            (data, delta, signum, len3, leap3, ebx2_r);
    }
    else {
        throw std::runtime_error("Lorenzo only works for 123-D.");
    }

    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(cudaStreamSynchronize(stream));

    TIME_ELAPSED_GPUEVENT(time_elapsed);
    DESTROY_GPUEVENT_PAIR;

    return CUSZ_SUCCESS;
}

template <typename T, typename DeltaT, typename FP>
psz_error_status asz::experimental::psz_decomp_l21var(
    DeltaT*      delta,
    bool*        signum,
    dim3 const   len3,
    double const eb,
    T*           xdata,
    float*       time_elapsed,
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
    // constexpr auto SEQ_1D    = 8;  // x-sequentiality == 8
    constexpr auto BLOCK_1D = dim3(256 / 8, 1, 1);
    auto           GRID_1D  = pardeg3(len3, SUBLEN_1D);

    constexpr auto SUBLEN_2D = dim3(16, 16, 1);
    // constexpr auto SEQ_2D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_2D = dim3(16, 2, 1);
    auto           GRID_2D  = pardeg3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    // constexpr auto SEQ_3D    = dim3(1, 8, 1);  // y-sequentiality == 8
    constexpr auto BLOCK_3D = dim3(32, 1, 8);
    auto           GRID_3D  = pardeg3(len3, SUBLEN_3D);

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING(stream);

    if (ndim() == 1) {
        cusz::experimental::x_lorenzo_1d1l<T, DeltaT, FP, 256, 8>  //
            <<<GRID_1D, BLOCK_1D, 0, stream>>>                     //
            (signum, delta, xdata, len3, leap3, ebx2);
    }
    else if (ndim() == 2) {
        cusz::experimental::x_lorenzo_2d1l_16x16data_mapto16x2<T, DeltaT, FP>  //
            <<<GRID_2D, BLOCK_2D, 0, stream>>>                                 //
            (signum, delta, xdata, len3, leap3, ebx2);
    }
    else {
        cusz::experimental::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<T, DeltaT, FP>  //
            <<<GRID_3D, BLOCK_3D, 0, stream>>>                                    //
            (signum, delta, xdata, len3, leap3, ebx2);
    }

    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(cudaStreamSynchronize(stream));

    TIME_ELAPSED_GPUEVENT(time_elapsed);
    DESTROY_GPUEVENT_PAIR;

    return CUSZ_SUCCESS;
}

#define CPP_INS(Tliteral, Eliteral, FPliteral, T, E, FP)                                      \
    template psz_error_status asz::experimental::psz_comp_l21var<T, E, FP>(                            \
        T*, dim3 const, double const, E*, bool*, float*, cudaStream_t);                                               \
                                                                                                                      \
    template psz_error_status asz::experimental::psz_decomp_l21var<T, E, FP>(                          \
        E*, bool*, dim3 const, double const, T*, float*, cudaStream_t);                                               \
                                                                                                                      \
    psz_error_status compress_predict_lorenzo_ivar_T##Tliteral##_E##Eliteral##_FP##FPliteral(                        \
        T* const data, dim3 const len3, double const eb, E* delta, bool* signum, float* time_elapsed,                 \
        cudaStream_t stream)                                                                                          \
    {                                                                                                                 \
        asz::experimental::psz_comp_l21var<T, E, FP>(                                                   \
            data, len3, eb, delta, signum, time_elapsed, stream);                                                     \
        return CUSZ_SUCCESS;                                                                                          \
    }                                                                                                                 \
                                                                                                                      \
    psz_error_status decompress_predict_lorenzo_ivar_T##Tliteral##_E##Eliteral##_FP##FPliteral(                      \
        E* delta, bool* signum, dim3 const len3, double const eb, T* xdata, float* time_elapsed, cudaStream_t stream) \
    {                                                                                                                 \
        asz::experimental::psz_decomp_l21var<T, E, FP>(                                                 \
            delta, signum, len3, eb, xdata, time_elapsed, stream);                                                    \
        return CUSZ_SUCCESS;                                                                                          \
    }

CPP_INS(fp32, ui8, fp32, float, uint8_t, float);
CPP_INS(fp32, ui16, fp32, float, uint16_t, float);
CPP_INS(fp32, ui32, fp32, float, uint32_t, float);
CPP_INS(fp32, fp32, fp32, float, float, float);

CPP_INS(fp64, ui8, fp64, double, uint8_t, double);
CPP_INS(fp64, ui16, fp64, double, uint16_t, double);
CPP_INS(fp64, ui32, fp64, double, uint32_t, double);
CPP_INS(fp64, fp32, fp64, double, float, double);

#undef CPP_INS
