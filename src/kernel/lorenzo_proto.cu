/**
 * @file claunch_cuda_proto.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-09-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "cusz/type.h"
#include "kernel/lorenzo_all.h"
#include "kernel/lorenzo_all.hh"
#include "pipeline/compaction_g.inl"
#include "utils/cuda_err.cuh"
#include "utils/timer.h"

#include "detail/lorenzo_proto.inl"

template <typename T, typename EQ, typename Outlier>
cusz_error_status compress_predict_lorenzo_iproto(
    T* const     data,
    dim3 const   len3,
    double const eb,
    int const    radius,
    EQ* const    eq,
    Outlier*     outlier,
    float*       time_elapsed,
    cudaStream_t stream)
{
    auto divide3 = [](dim3 len, dim3 sublen) {
        return dim3((len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1, (len.z - 1) / sublen.z + 1);
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

    using namespace psz::cuda::__kernel::proto;

    if (ndim() == 1) {
        c_lorenzo_1d1l<T, EQ><<<GRID_1D, BLOCK_1D, 0, stream>>>(data, len3, leap3, radius, ebx2_r, eq, outlier);
    }
    else if (ndim() == 2) {
        c_lorenzo_2d1l<T, EQ><<<GRID_2D, BLOCK_2D, 0, stream>>>(data, len3, leap3, radius, ebx2_r, eq, outlier);
    }
    else if (ndim() == 3) {
        c_lorenzo_3d1l<T, EQ><<<GRID_3D, BLOCK_3D, 0, stream>>>(data, len3, leap3, radius, ebx2_r, eq, outlier);
    }
    else {
        throw std::runtime_error("Lorenzo only works for 123-D.");
    }

    STOP_CUDAEVENT_RECORDING(stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    TIME_ELAPSED_CUDAEVENT(time_elapsed);
    DESTROY_CUDAEVENT_PAIR;

    return CUSZ_SUCCESS;
}

template <typename T, typename EQ>
cusz_error_status decompress_predict_lorenzo_iproto(
    EQ*          eq,
    dim3 const   len3,
    T*           scattered_outlier,
    double const eb,
    int const    radius,
    T*           xdata,
    float*       time_elapsed,
    cudaStream_t stream)
{
    auto divide3 = [](dim3 len, dim3 sublen) {
        return dim3((len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1, (len.z - 1) / sublen.z + 1);
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

    using namespace psz::cuda::__kernel::proto;

    if (ndim() == 1) {
        x_lorenzo_1d1l<T, EQ>
            <<<GRID_1D, BLOCK_1D, 0, stream>>>(eq, scattered_outlier, len3, leap3, radius, ebx2, xdata);
    }
    else if (ndim() == 2) {
        x_lorenzo_2d1l<T, EQ>
            <<<GRID_2D, BLOCK_2D, 0, stream>>>(eq, scattered_outlier, len3, leap3, radius, ebx2, xdata);
    }
    else if (ndim() == 3) {
        x_lorenzo_3d1l<T, EQ>
            <<<GRID_3D, BLOCK_3D, 0, stream>>>(eq, scattered_outlier, len3, leap3, radius, ebx2, xdata);
    }

    STOP_CUDAEVENT_RECORDING(stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    TIME_ELAPSED_CUDAEVENT(time_elapsed);
    DESTROY_CUDAEVENT_PAIR;

    return CUSZ_SUCCESS;
}

#define CPP_TEMPLATE_INIT_AND_C_WRAPPER(T, EQ)                                                                \
    template cusz_error_status compress_predict_lorenzo_iproto<T, EQ>(                                        \
        T* const, dim3 const, double const, int const, EQ* const, CompactCudaDram<T>*, float*, cudaStream_t); \
                                                                                                              \
    template cusz_error_status decompress_predict_lorenzo_iproto<T, EQ>(                                      \
        EQ*, dim3 const, T*, double const, int const, T*, float*, cudaStream_t);

CPP_TEMPLATE_INIT_AND_C_WRAPPER(float, uint8_t);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(float, uint16_t);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(float, uint32_t);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(float, float);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(float, int32_t);

CPP_TEMPLATE_INIT_AND_C_WRAPPER(double, uint8_t);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(double, uint16_t);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(double, uint32_t);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(double, float);
CPP_TEMPLATE_INIT_AND_C_WRAPPER(double, int32_t);

#undef CPP_TEMPLATE_INIT_AND_C_WRAPPER
