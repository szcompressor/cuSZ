/**
 * @file claunch_cuda_var.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-27
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "kernel/claunch_cuda.h"
#include "kernel/lorenzo_var.cuh"

#define C_LORENZOI_VAR(Tliteral, Eliteral, FPliteral, T, E, FP)                                                       \
    cusz_error_status claunch_construct_LorenzoI_var_T##Tliteral##_E##Eliteral##_FP##FPliteral(                       \
        T* const data, E* delta, bool* signum, dim3 const len3, double const eb, float* time_elapsed,                 \
        cudaStream_t stream)                                                                                          \
    {                                                                                                                 \
        cusz::experimental::launch_construct_LorenzoI_var<T, E, FP>(                                                  \
            data, delta, signum, len3, eb, *time_elapsed, stream);                                                    \
        return CUSZ_SUCCESS;                                                                                          \
    }                                                                                                                 \
                                                                                                                      \
    cusz_error_status claunch_reconstruct_LorenzoI_var_T##Tliteral##_E##Eliteral##_FP##FPliteral(                     \
        bool* signum, E* delta, T* xdata, dim3 const len3, double const eb, float* time_elapsed, cudaStream_t stream) \
    {                                                                                                                 \
        cusz::experimental::launch_reconstruct_LorenzoI_var<T, E, FP>(                                                \
            signum, delta, xdata, len3, eb, *time_elapsed, stream);                                                   \
        return CUSZ_SUCCESS;                                                                                          \
    }

C_LORENZOI_VAR(fp32, ui8, fp32, float, uint8_t, float);
C_LORENZOI_VAR(fp32, ui16, fp32, float, uint16_t, float);
C_LORENZOI_VAR(fp32, ui32, fp32, float, uint32_t, float);
C_LORENZOI_VAR(fp32, fp32, fp32, float, float, float);

C_LORENZOI_VAR(fp64, ui8, fp64, double, uint8_t, double);
C_LORENZOI_VAR(fp64, ui16, fp64, double, uint16_t, double);
C_LORENZOI_VAR(fp64, ui32, fp64, double, uint32_t, double);
C_LORENZOI_VAR(fp64, fp32, fp64, double, float, double);

#undef C_LORENZOI_VAR
