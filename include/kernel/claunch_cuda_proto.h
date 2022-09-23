/**
 * @file claunch_cuda_proto.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-09-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef F8664941_35AD_4E6E_975D_5E495B81E08D
#define F8664941_35AD_4E6E_975D_5E495B81E08D


#ifdef __cplusplus

// hip requires cpp linkage
#include <hip/hip_runtime.h>

extern "C" {
#endif

#include <stdint.h>
#include "../cusz/type.h"

#define C_CONSTRUCT_LORENZOI_PROTO(Tliteral, Eliteral, FPliteral, T, E, FP)                                 \
    cusz_error_status claunch_construct_LorenzoI_proto_T##Tliteral##_E##Eliteral##_FP##FPliteral(           \
        bool NO_R_SEPARATE, T* const data, dim3 const len3, T* const anchor, dim3 const placeholder_1,      \
        E* const errctrl, dim3 const placeholder_2, double const eb, int const radius, float* time_elapsed, \
        hipStream_t stream);

C_CONSTRUCT_LORENZOI_PROTO(fp32, ui8, fp32, float, uint8_t, float);
C_CONSTRUCT_LORENZOI_PROTO(fp32, ui16, fp32, float, uint16_t, float);
C_CONSTRUCT_LORENZOI_PROTO(fp32, ui32, fp32, float, uint32_t, float);
C_CONSTRUCT_LORENZOI_PROTO(fp32, fp32, fp32, float, float, float);

#undef C_CONSTRUCT_LORENZOI_PROTO

#define C_RECONSTRUCT_LORENZOI_PROTO(Tliteral, Eliteral, FPliteral, T, E, FP)                                 \
    cusz_error_status claunch_reconstruct_LorenzoI_proto_T##Tliteral##_E##Eliteral##_FP##FPliteral(           \
        T* xdata, dim3 const len3, T* anchor, dim3 const placeholder_1, E* errctrl, dim3 const placeholder_2, \
        double const eb, int const radius, float* time_elapsed, hipStream_t stream);

C_RECONSTRUCT_LORENZOI_PROTO(fp32, ui8, fp32, float, uint8_t, float);
C_RECONSTRUCT_LORENZOI_PROTO(fp32, ui16, fp32, float, uint16_t, float);
C_RECONSTRUCT_LORENZOI_PROTO(fp32, ui32, fp32, float, uint32_t, float);
C_RECONSTRUCT_LORENZOI_PROTO(fp32, fp32, fp32, float, float, float);

#undef C_RECONSTRUCT_LORENZOI_PROTO

#ifdef __cplusplus
}
#endif

#endif /* F8664941_35AD_4E6E_975D_5E495B81E08D */
