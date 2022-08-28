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

#include "../cusz/type.h"

#define C_CONSTRUCT_LORENZOI(Tliteral, Eliteral, FPliteral, T, E, FP)                                       \
    cusz_error_status claunch_construct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                 \
        bool NO_R_SEPARATE, T* const data, dim3 const len3, T* const anchor, dim3 const placeholder_1,      \
        E* const errctrl, dim3 const placeholder_2, double const eb, int const radius, float* time_elapsed, \
        cudaStream_t stream);

C_CONSTRUCT_LORENZOI(fp32, ui8, fp32, float, uint8_t, float);
C_CONSTRUCT_LORENZOI(fp32, ui16, fp32, float, uint16_t, float);
C_CONSTRUCT_LORENZOI(fp32, ui32, fp32, float, uint32_t, float);
C_CONSTRUCT_LORENZOI(fp32, fp32, fp32, float, float, float);

#undef C_CONSTRUCT_LORENZOI

#define C_RECONSTRUCT_LORENZOI(Tliteral, Eliteral, FPliteral, T, E, FP)                                       \
    cusz_error_status claunch_reconstruct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                 \
        T* xdata, dim3 const len3, T* anchor, dim3 const placeholder_1, E* errctrl, dim3 const placeholder_2, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream);

C_RECONSTRUCT_LORENZOI(fp32, ui8, fp32, float, uint8_t, float);
C_RECONSTRUCT_LORENZOI(fp32, ui16, fp32, float, uint16_t, float);
C_RECONSTRUCT_LORENZOI(fp32, ui32, fp32, float, uint32_t, float);
C_RECONSTRUCT_LORENZOI(fp32, fp32, fp32, float, float, float);

#undef C_RECONSTRUCT_LORENZOI

#define C_CONSTRUCT_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                                 \
    cusz_error_status claunch_construct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                           \
        bool NO_R_SEPARATE, T* data, dim3 const len3, T* anchor, dim3 const an_len3, E* errctrl, dim3 const ec_len3, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream);

C_CONSTRUCT_SPLINE3(fp32, ui8, fp32, float, uint8_t, float);
C_CONSTRUCT_SPLINE3(fp32, ui16, fp32, float, uint16_t, float);
C_CONSTRUCT_SPLINE3(fp32, ui32, fp32, float, uint32_t, float);
C_CONSTRUCT_SPLINE3(fp32, fp32, fp32, float, float, float);

#undef C_CONSTRUCT_SPLINE3

#define C_RECONSTRUCT_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                             \
    cusz_error_status claunch_reconstruct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                       \
        T* xdata, dim3 const len3, T* anchor, dim3 const an_len3, E* errctrl, dim3 const ec_len3, double const eb, \
        int const radius, float* time_elapsed, cudaStream_t stream);

C_RECONSTRUCT_SPLINE3(fp32, ui8, fp32, float, uint8_t, float);
C_RECONSTRUCT_SPLINE3(fp32, ui16, fp32, float, uint16_t, float);
C_RECONSTRUCT_SPLINE3(fp32, ui32, fp32, float, uint32_t, float);
C_RECONSTRUCT_SPLINE3(fp32, fp32, fp32, float, float, float);

#undef C_RECONSTRUCT_SPLINE3

// #define C_GPUPAR_CODEBOOK(Tliteral, Hliteral, Mliteral, T, H, M)                                                 \
//     cusz_error_status claunch_gpu_parallel_build_codebook_T##Tliteral##_H##Hliteral##_M##Mliteral(               \
//         uint32_t* freq, H* book, int const booklen, uint8_t* revbook, int const revbook_nbyte, float* time_book, \
//         cudaStream_t stream);

// C_GPUPAR_CODEBOOK(ui8, ui32, ui32, uint8_t, uint32_t, uint32_t);
// C_GPUPAR_CODEBOOK(ui16, ui32, ui32, uint8_t, uint32_t, uint32_t);
// C_GPUPAR_CODEBOOK(ui32, ui32, ui32, uint8_t, uint32_t, uint32_t);
// C_GPUPAR_CODEBOOK(ui8, ui64, ui32, uint8_t, uint64_t, uint32_t);
// C_GPUPAR_CODEBOOK(ui16, ui64, ui32, uint8_t, uint64_t, uint32_t);
// C_GPUPAR_CODEBOOK(ui32, ui64, ui32, uint8_t, uint64_t, uint32_t);
// C_GPUPAR_CODEBOOK(ui8, ul, ui32, uint8_t, unsigned long, uint32_t);
// C_GPUPAR_CODEBOOK(ui16, ul, ui32, uint8_t, unsigned long, uint32_t);
// C_GPUPAR_CODEBOOK(ui32, ul, ui32, uint8_t, unsigned long, uint32_t);
//
// #undef C_GPUPAR_CODEBOOK

#define C_COARSE_HUFFMAN_ENCODE(Tliteral, Hliteral, Mliteral, T, H, M)                                          \
    cusz_error_status claunch_coarse_grained_Huffman_encoding_T##Tliteral##_H##Hliteral##_M##Mliteral(          \
        T* uncompressed, H* d_internal_coded, size_t const len, uint32_t* d_freq, H* d_book, int const booklen, \
        H* d_bitstream, M* d_par_metadata, M* h_par_metadata, int const sublen, int const pardeg, int numSMs,   \
        uint8_t** out_compressed, size_t* out_compressed_len, float* time_lossless, cudaStream_t stream);

C_COARSE_HUFFMAN_ENCODE(ui8, ui32, ui32, uint8_t, uint32_t, uint32_t);
C_COARSE_HUFFMAN_ENCODE(ui16, ui32, ui32, uint16_t, uint32_t, uint32_t);
C_COARSE_HUFFMAN_ENCODE(ui32, ui32, ui32, uint32_t, uint32_t, uint32_t);
C_COARSE_HUFFMAN_ENCODE(fp32, ui32, ui32, float, uint32_t, uint32_t);
C_COARSE_HUFFMAN_ENCODE(ui8, ui64, ui32, uint8_t, uint64_t, uint32_t);
C_COARSE_HUFFMAN_ENCODE(ui16, ui64, ui32, uint16_t, uint64_t, uint32_t);
C_COARSE_HUFFMAN_ENCODE(ui32, ui64, ui32, uint32_t, uint64_t, uint32_t);
C_COARSE_HUFFMAN_ENCODE(fp32, ui64, ui32, float, uint64_t, uint32_t);
C_COARSE_HUFFMAN_ENCODE(ui8, ull, ui32, uint8_t, unsigned long long, uint32_t);
C_COARSE_HUFFMAN_ENCODE(ui16, ull, ui32, uint16_t, unsigned long long, uint32_t);
C_COARSE_HUFFMAN_ENCODE(ui32, ull, ui32, uint32_t, unsigned long long, uint32_t);
C_COARSE_HUFFMAN_ENCODE(fp32, ull, ui32, float, unsigned long long, uint32_t);

#undef C_COARSE_HUFFMAN_ENCODE

#define C_COARSE_HUFFMAN_DECODE(Tliteral, Hliteral, Mliteral, T, H, M)                                                \
    cusz_error_status claunch_coarse_grained_Huffman_decoding_T##Tliteral##_H##Hliteral##_M##Mliteral(                \
        H* d_bitstream, uint8_t* d_revbook, int const revbook_nbyte, M* d_par_nbit, M* d_par_entry, int const sublen, \
        int const pardeg, T* out_decompressed, float* time_lossless, cudaStream_t stream);

C_COARSE_HUFFMAN_DECODE(ui8, ui32, ui32, uint8_t, uint32_t, uint32_t);
C_COARSE_HUFFMAN_DECODE(ui16, ui32, ui32, uint16_t, uint32_t, uint32_t);
C_COARSE_HUFFMAN_DECODE(ui32, ui32, ui32, uint32_t, uint32_t, uint32_t);
C_COARSE_HUFFMAN_DECODE(fp32, ui32, ui32, float, uint32_t, uint32_t);
C_COARSE_HUFFMAN_DECODE(ui8, ui64, ui32, uint8_t, uint64_t, uint32_t);
C_COARSE_HUFFMAN_DECODE(ui16, ui64, ui32, uint16_t, uint64_t, uint32_t);
C_COARSE_HUFFMAN_DECODE(ui32, ui64, ui32, uint32_t, uint64_t, uint32_t);
C_COARSE_HUFFMAN_DECODE(fp32, ui64, ui32, float, uint64_t, uint32_t);
C_COARSE_HUFFMAN_DECODE(ui8, ull, ui32, uint8_t, unsigned long long, uint32_t);
C_COARSE_HUFFMAN_DECODE(ui16, ull, ui32, uint16_t, unsigned long long, uint32_t);
C_COARSE_HUFFMAN_DECODE(ui32, ull, ui32, uint32_t, unsigned long long, uint32_t);
C_COARSE_HUFFMAN_DECODE(fp32, ull, ui32, float, unsigned long long, uint32_t);

#undef C_COARSE_HUFFMAN_DECODE

#ifdef __cplusplus
}
#endif

#endif
