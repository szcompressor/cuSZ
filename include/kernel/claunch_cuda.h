/**
 * @file claunch_cuda.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-07-24
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef KERNEL_CUDA_H
#define KERNEL_CUDA_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>

#include "../cusz/type.h"
// #include "../hf/hf_struct.h"

#define C_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                                           \
    cusz_error_status claunch_construct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                           \
        bool NO_R_SEPARATE, T* data, dim3 const len3, T* anchor, dim3 const an_len3, E* errctrl, dim3 const ec_len3, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream);                                \
                                                                                                                     \
    cusz_error_status claunch_reconstruct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                         \
        T* xdata, dim3 const len3, T* anchor, dim3 const an_len3, E* errctrl, dim3 const ec_len3, double const eb,   \
        int const radius, float* time_elapsed, cudaStream_t stream);

C_SPLINE3(fp32, ui8, fp32, float, uint8_t, float);
C_SPLINE3(fp32, ui16, fp32, float, uint16_t, float);
C_SPLINE3(fp32, ui32, fp32, float, uint32_t, float);
C_SPLINE3(fp32, fp32, fp32, float, float, float);

#undef C_SPLINE3

#undef C_COARSE_HUFFMAN_DECODE

#ifdef __cplusplus
}
#endif

#endif
