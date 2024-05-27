/**
 * @file kernel_cuda.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-07-24
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "detail/hist.inl"
#include "detail/spline3.inl"
// #include "hf/hfcodec.cu.hh"
// #include "hf/hf_struct.h"
#include "kernel/claunch_cuda.h"
#include "kernel/cpplaunch_cuda.hh"
#include "utils/err.hh"

#define C_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                                           \
    psz_error_status claunch_construct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                           \
        bool NO_R_SEPARATE, T* data, dim3 const len3, T* anchor, dim3 const an_len3, E* errctrl, dim3 const ec_len3, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream)                                 \
    {                                                                                                                \
        if (NO_R_SEPARATE)                                                                                           \
            launch_construct_Spline3<T, E, FP, true>(                                                                \
                data, len3, anchor, an_len3, errctrl, ec_len3, eb, radius, *time_elapsed, stream);                   \
        else                                                                                                         \
            launch_construct_Spline3<T, E, FP, false>(                                                               \
                data, len3, anchor, an_len3, errctrl, ec_len3, eb, radius, *time_elapsed, stream);                   \
        return CUSZ_SUCCESS;                                                                                         \
    }                                                                                                                \
    psz_error_status claunch_reconstruct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                         \
        T* xdata, dim3 const len3, T* anchor, dim3 const an_len3, E* errctrl, dim3 const ec_len3, double const eb,   \
        int const radius, float* time_elapsed, cudaStream_t stream)                                                  \
    {                                                                                                                \
        launch_reconstruct_Spline3<T, E, FP>(                                                                        \
            xdata, len3, anchor, an_len3, errctrl, ec_len3, eb, radius, *time_elapsed, stream);                      \
        return CUSZ_SUCCESS;                                                                                         \
    }

C_SPLINE3(fp32, ui8, fp32, float, uint8_t, float);
C_SPLINE3(fp32, ui16, fp32, float, uint16_t, float);
C_SPLINE3(fp32, ui32, fp32, float, uint32_t, float);
C_SPLINE3(fp32, fp32, fp32, float, float, float);

#undef C_SPLINE3

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CPP_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                                    \
    template <>                                                                                                 \
    psz_error_status cusz::cpplaunch_construct_Spline3<T, E, FP>(                                              \
        bool NO_R_SEPARATE, T* data, dim3 const len3, T* anchor, dim3 const an_len3, E* eq, dim3 const ec_len3, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream)                            \
    {                                                                                                           \
        return claunch_construct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                             \
            NO_R_SEPARATE, data, len3, anchor, an_len3, eq, ec_len3, eb, radius, time_elapsed, stream);         \
    }                                                                                                           \
                                                                                                                \
    template <>                                                                                                 \
    psz_error_status cusz::cpplaunch_reconstruct_Spline3<T, E, FP>(                                            \
        T * xdata, dim3 const len3, T* anchor, dim3 const an_len3, E* eq, dim3 const ec_len3, double const eb,  \
        int const radius, float* time_elapsed, cudaStream_t stream)                                             \
    {                                                                                                           \
        return claunch_reconstruct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                           \
            xdata, len3, anchor, an_len3, eq, ec_len3, eb, radius, time_elapsed, stream);                       \
    }

CPP_SPLINE3(fp32, ui8, fp32, float, uint8_t, float);
CPP_SPLINE3(fp32, ui16, fp32, float, uint16_t, float);
CPP_SPLINE3(fp32, ui32, fp32, float, uint32_t, float);
CPP_SPLINE3(fp32, fp32, fp32, float, float, float);

#undef CPP_SPLINE3
