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

#include "kernel/claunch_cuda_proto.h"
#include "kernel/launch_prototype.cuh"

#define C_CONSTRUCT_LORENZOI_PROTO(Tliteral, Eliteral, FPliteral, T, E, FP)                                 \
    cusz_error_status claunch_construct_LorenzoI_proto_T##Tliteral##_E##Eliteral##_FP##FPliteral(           \
        bool NO_R_SEPARATE, T* const data, dim3 const len3, T* const anchor, dim3 const placeholder_1,      \
        E* const errctrl, dim3 const placeholder_2, double const eb, int const radius, float* time_elapsed, \
        cudaStream_t stream)                                                                                \
    {                                                                                                       \
        launch_construct_LorenzoI_proto<T, E, FP>(                                                          \
            data, len3, anchor, placeholder_1, errctrl, placeholder_2, eb, radius, *time_elapsed, stream);  \
        return CUSZ_SUCCESS;                                                                                \
    }

C_CONSTRUCT_LORENZOI_PROTO(fp32, ui8, fp32, float, uint8_t, float);
C_CONSTRUCT_LORENZOI_PROTO(fp32, ui16, fp32, float, uint16_t, float);
C_CONSTRUCT_LORENZOI_PROTO(fp32, ui32, fp32, float, uint32_t, float);
C_CONSTRUCT_LORENZOI_PROTO(fp32, fp32, fp32, float, float, float);

#undef C_CONSTRUCT_LORENZOI_PROTO

#define C_RECONSTRUCT_LORENZOI_PROTO(Tliteral, Eliteral, FPliteral, T, E, FP)                                 \
    cusz_error_status claunch_reconstruct_LorenzoI_proto_T##Tliteral##_E##Eliteral##_FP##FPliteral(           \
        T* xdata, dim3 const len3, T* anchor, dim3 const placeholder_1, E* errctrl, dim3 const placeholder_2, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream)                          \
    {                                                                                                         \
        launch_reconstruct_LorenzoI_proto<T, E, FP>(                                                          \
            xdata, len3, anchor, placeholder_1, errctrl, placeholder_2, eb, radius, *time_elapsed, stream);   \
        return CUSZ_SUCCESS;                                                                                  \
    }

C_RECONSTRUCT_LORENZOI_PROTO(fp32, ui8, fp32, float, uint8_t, float);
C_RECONSTRUCT_LORENZOI_PROTO(fp32, ui16, fp32, float, uint16_t, float);
C_RECONSTRUCT_LORENZOI_PROTO(fp32, ui32, fp32, float, uint32_t, float);
C_RECONSTRUCT_LORENZOI_PROTO(fp32, fp32, fp32, float, float, float);

#undef C_RECONSTRUCT_LORENZOI_PROTO
