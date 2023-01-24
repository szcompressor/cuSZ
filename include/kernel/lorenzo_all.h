/**
 * @file kernel_cuda.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-11-02
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef BD8A19DE_E881_4A26_9464_C51DAC6B14E1
#define BD8A19DE_E881_4A26_9464_C51DAC6B14E1

#ifdef __cplusplus
extern "C" {
#endif

#include "cusz/type.h"

#define C_LORENZOI(Tliteral, Eliteral, FPliteral, T, E, FP)                                                            \
    cusz_error_status compress_predict_lorenzo_i_T##Tliteral##_E##Eliteral##_FP##FPliteral(                            \
        T* const data, dim3 const len3, T* const anchor, dim3 const placeholder_1, E* const errctrl,                   \
        dim3 const placeholder_2, T* outlier, double const eb, int const radius, float* time_elapsed,                  \
        cudaStream_t stream);                                                                                          \
    cusz_error_status decompress_predict_lorenzo_i_T##Tliteral##_E##Eliteral##_FP##FPliteral(                          \
        T* xdata, dim3 const len3, T* anchor, dim3 const placeholder_1, E* errctrl, dim3 const placeholder_2,          \
        T* outlier, double const eb, int const radius, float* time_elapsed, cudaStream_t stream);                      \
    cusz_error_status compress_predict_lorenzo_ivar_T##Tliteral##_E##Eliteral##_FP##FPliteral(                         \
        T* const data, dim3 const len3, double const eb, E* delta, bool* signum, float* time_elapsed,                  \
        cudaStream_t stream);                                                                                          \
    cusz_error_status decompress_predict_lorenzo_ivar_T##Tliteral##_E##Eliteral##_FP##FPliteral(                       \
        E* delta, bool* signum, dim3 const len3, double const eb, T* xdata, float* time_elapsed, cudaStream_t stream); \
    cusz_error_status compress_predict_lorenzo_iproto_T##Tliteral##_E##Eliteral##_FP##FPliteral(                       \
        T* const data, dim3 const len3, T* const anchor, dim3 const placeholder_1, E* const errctrl,                   \
        dim3 const placeholder_2, T* outlier, double const eb, int const radius, float* time_elapsed,                  \
        cudaStream_t stream);                                                                                          \
    cusz_error_status decompress_predict_lorenzo_iproto_T##Tliteral##_E##Eliteral##_FP##FPliteral(                     \
        T* xdata, dim3 const len3, T* anchor, dim3 const placeholder_1, E* errctrl, dim3 const placeholder_2,          \
        T* outlier, double const eb, int const radius, float* time_elapsed, cudaStream_t stream);

C_LORENZOI(fp32, ui8, fp32, float, uint8_t, float);
C_LORENZOI(fp32, ui16, fp32, float, uint16_t, float);
C_LORENZOI(fp32, ui32, fp32, float, uint32_t, float);
C_LORENZOI(fp32, fp32, fp32, float, float, float);

C_LORENZOI(fp64, ui8, fp64, double, uint8_t, double);
C_LORENZOI(fp64, ui16, fp64, double, uint16_t, double);
C_LORENZOI(fp64, ui32, fp64, double, uint32_t, double);
C_LORENZOI(fp64, fp32, fp64, double, float, double);

#undef C_LORENZOI

#ifdef __cplusplus
}
#endif

#endif /* BD8A19DE_E881_4A26_9464_C51DAC6B14E1 */
