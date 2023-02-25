/**
 * @file v2_lorenzo.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-01-23
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "cusz/type.h"
#include "utils/cuda_err.cuh"
#include "utils/timer.h"

#include "kernel/lorenzo_all.hh"
#include "kernel/v2_lorenzo.hh"

template <typename T, typename E, typename FP>
cusz_error_status v2_compress_predict_lorenzo_i(
    T* const          data,
    dim3 const        len3,
    double const      eb,
    int const         radius,
    E* const          errctrl,
    dim3 const        placeholder_2,
    T* const          anchor,
    dim3 const        placeholder_1,
    CompactionDRAM<T> outlier,
    float*            time_elapsed,
    cudaStream_t      stream)
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
    constexpr auto BLOCK_2D  = dim3(16, 2, 1);
    auto           GRID_2D   = divide3(len3, SUBLEN_2D);

    constexpr auto SUBLEN_3D = dim3(32, 8, 8);
    constexpr auto BLOCK_3D  = dim3(32, 8, 1);  // for v0::r1_shfl
    auto           GRID_3D   = divide3(len3, SUBLEN_3D);

    auto d = ndim();

    // error bound
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / ebx2;
    auto leap3  = dim3(1, len3.x, len3.x * len3.y);

    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream);

    if (d == 1) {
        parsz::cuda::__kernel::v0::compaction::c_lorenzo_1d1l<T, E, FP, SUBLEN_1D, SEQ_1D>
            <<<GRID_1D, BLOCK_1D, 0, stream>>>(data, len3, leap3, radius, ebx2_r, errctrl, outlier);
    }
    else if (d == 2) {
        parsz::cuda::__kernel::v0::compaction::c_lorenzo_2d1l<T, E, FP>
            <<<GRID_2D, BLOCK_2D, 0, stream>>>(data, len3, leap3, radius, ebx2_r, errctrl, outlier);
    }
    else if (d == 3) {
        parsz::cuda::__kernel::v0::compaction::c_lorenzo_3d1l<T, E, FP>
            <<<GRID_3D, BLOCK_3D, 0, stream>>>(data, len3, leap3, radius, ebx2_r, errctrl, outlier);
    }

    STOP_CUDAEVENT_RECORDING(stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    TIME_ELAPSED_CUDAEVENT(time_elapsed);
    DESTROY_CUDAEVENT_PAIR;

    return CUSZ_SUCCESS;
}

#define CPP_TEMPLATE_INIT_AND_C_WRAPPER(Tliteral, Eliteral, FPliteral, T, E, FP)                   \
    template cusz_error_status v2_compress_predict_lorenzo_i<T, E, FP>(                            \
        T* const, dim3 const, double const, int const, E* const, dim3 const, T* const, dim3 const, \
        struct CompactionDRAM<T>, float*, cudaStream_t);                                           \
                                                                                                   \
    // cusz_error_status v2_compress_predict_lorenzo_i_T##Tliteral##_E##Eliteral##_FP##FPliteral(                \
    //     T* const data, dim3 const len3, T* const anchor, dim3 const placeholder_1, E* const errctrl,          \
    //     dim3 const placeholder_2, T* outlier, double const eb, int const radius, float* time_elapsed,         \
    //     cudaStream_t stream)                                                                                  \
    // {                                                                                                         \
    //     return v2_compress_predict_lorenzo_i<T, E, FP>(                                                       \
    //         data, len3, eb, radius, errctrl, placeholder_2, anchor, placeholder_1, outlier, nullptr, nullptr, \
    //         time_elapsed, stream);                                                                            \
    // }

CPP_TEMPLATE_INIT_AND_C_WRAPPER(fp32, ui8, fp32, float, uint8_t, float);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(fp32, ui16, fp32, float, uint16_t, float);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(fp32, ui32, fp32, float, uint32_t, float);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(fp32, fp32, fp32, float, float, float);

CPP_TEMPLATE_INIT_AND_C_WRAPPER(fp64, ui8, fp64, double, uint8_t, double);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(fp64, ui16, fp64, double, uint16_t, double);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(fp64, ui32, fp64, double, uint32_t, double);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(fp64, fp32, fp64, double, float, double);

#undef CPP_TEMPLATE_INIT_AND_C_WRAPPER
