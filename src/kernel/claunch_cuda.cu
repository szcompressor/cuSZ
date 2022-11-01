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

#include "hf/hf_struct.h"
#include "hist.inl"
#include "kernel/claunch_cuda.h"
#include "kernel/cpplaunch_cuda.hh"
#include "kernel/launch_lossless.cuh"
#include "lorenzo.inl"
#include "spline3.inl"
#include "utils/cuda_err.cuh"

#define C_LORENZOI(Tliteral, Eliteral, FPliteral, T, E, FP)                                                          \
    cusz_error_status claunch_construct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                          \
        T* const data, dim3 const len3, T* const anchor, dim3 const placeholder_1, E* const errctrl,                 \
        dim3 const placeholder_2, T* outlier, double const eb, int const radius, float* time_elapsed,                \
        cudaStream_t stream)                                                                                         \
    {                                                                                                                \
        launch_construct_LorenzoI<T, E, FP>(                                                                         \
            data, len3, anchor, placeholder_1, errctrl, placeholder_2, outlier, eb, radius, *time_elapsed, stream);  \
        return CUSZ_SUCCESS;                                                                                         \
    }                                                                                                                \
                                                                                                                     \
    cusz_error_status claunch_reconstruct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                        \
        T* xdata, dim3 const len3, T* anchor, dim3 const placeholder_1, E* errctrl, dim3 const placeholder_2,        \
        T* outlier, double const eb, int const radius, float* time_elapsed, cudaStream_t stream)                     \
    {                                                                                                                \
        launch_reconstruct_LorenzoI<T, E, FP>(                                                                       \
            xdata, len3, anchor, placeholder_1, errctrl, placeholder_2, outlier, eb, radius, *time_elapsed, stream); \
        return CUSZ_SUCCESS;                                                                                         \
    }

C_LORENZOI(fp32, ui8, fp32, float, uint8_t, float);
C_LORENZOI(fp32, ui16, fp32, float, uint16_t, float);
C_LORENZOI(fp32, ui32, fp32, float, uint32_t, float);
C_LORENZOI(fp32, fp32, fp32, float, float, float);

C_LORENZOI(fp64, ui8, fp64, double, uint8_t, double);
C_LORENZOI(fp64, ui16, fp64, double, uint16_t, double);
C_LORENZOI(fp64, ui32, fp64, double, uint32_t, double);
C_LORENZOI(fp64, fp32, fp64, double, float, double);

#undef C_LORENZOI

#define C_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                                           \
    cusz_error_status claunch_construct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                           \
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
    cusz_error_status claunch_reconstruct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                         \
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

#define C_HIST(Tliteral, T)                                                                                       \
    cusz_error_status claunch_histogram_T##Tliteral(                                                              \
        T* in_data, size_t in_len, uint32_t* out_freq, int num_buckets, float* milliseconds, cudaStream_t stream) \
    {                                                                                                             \
        launch_histogram<T>(in_data, in_len, out_freq, num_buckets, *milliseconds, stream);                       \
        return cusz_error_status::CUSZ_SUCCESS;                                                                   \
    }

C_HIST(ui8, uint8_t)
C_HIST(ui16, uint16_t)
C_HIST(ui32, uint32_t)
C_HIST(ui64, uint64_t)

#undef C_HIST

// #define C_GPUPAR_CODEBOOK(Tliteral, Hliteral, Mliteral, T, H, M)                                                      \
//     cusz_error_status claunch_gpu_parallel_build_codebook_T##Tliteral##_H##Hliteral##_M##Mliteral(                    \
//         uint32_t* freq, H* book, int const booklen, uint8_t* revbook, int const revbook_nbyte, float* time_book,      \
//         cudaStream_t stream)                                                                                          \
//     {                                                                                                                 \
//         launch_gpu_parallel_build_codebook<T, H, M>(freq, book, booklen, revbook, revbook_nbyte, *time_book, stream); \
//         return CUSZ_SUCCESS;                                                                                          \
//     }

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

#define C_COARSE_HUFFMAN(Tliteral, Hliteral, Mliteral, T, H, M)                                                        \
    cusz_error_status claunch_coarse_grained_Huffman_encoding_T##Tliteral##_H##Hliteral##_M##Mliteral(                 \
        T* uncompressed, H* d_internal_coded, size_t const len, uint32_t* d_freq, H* d_book, int const booklen,        \
        H* d_bitstream, M* d_par_metadata, M* h_par_metadata, int const sublen, int const pardeg, int numSMs,          \
        uint8_t** out_compressed, size_t* out_compressed_len, float* time_lossless, cudaStream_t stream)               \
    {                                                                                                                  \
        launch_coarse_grained_Huffman_encoding<T, H, M>(                                                               \
            uncompressed, d_internal_coded, len, d_freq, d_book, booklen, d_bitstream, d_par_metadata, h_par_metadata, \
            sublen, pardeg, numSMs, *out_compressed, *out_compressed_len, *time_lossless, stream);                     \
        return CUSZ_SUCCESS;                                                                                           \
    }                                                                                                                  \
                                                                                                                       \
    cusz_error_status claunch_coarse_grained_Huffman_encoding_rev1_T##Tliteral##_H##Hliteral##_M##Mliteral(            \
        T* uncompressed, size_t const len, hf_book* book_desc, hf_bitstream* bitstream_desc, uint8_t** out_compressed, \
        size_t* out_compressed_len, float* time_lossless, cudaStream_t stream)                                         \
    {                                                                                                                  \
        launch_coarse_grained_Huffman_encoding_rev1<T, H, M>(                                                          \
            uncompressed, len, book_desc, bitstream_desc, *out_compressed, *out_compressed_len, *time_lossless,        \
            stream);                                                                                                   \
        return CUSZ_SUCCESS;                                                                                           \
    }                                                                                                                  \
                                                                                                                       \
    cusz_error_status claunch_coarse_grained_Huffman_decoding_T##Tliteral##_H##Hliteral##_M##Mliteral(                 \
        H* d_bitstream, uint8_t* d_revbook, int const revbook_nbyte, M* d_par_nbit, M* d_par_entry, int const sublen,  \
        int const pardeg, T* out_decompressed, float* time_lossless, cudaStream_t stream)                              \
    {                                                                                                                  \
        launch_coarse_grained_Huffman_decoding(                                                                        \
            d_bitstream, d_revbook, revbook_nbyte, d_par_nbit, d_par_entry, sublen, pardeg, out_decompressed,          \
            *time_lossless, stream);                                                                                   \
        return CUSZ_SUCCESS;                                                                                           \
    }

C_COARSE_HUFFMAN(ui8, ui32, ui32, uint8_t, uint32_t, uint32_t);
C_COARSE_HUFFMAN(ui16, ui32, ui32, uint16_t, uint32_t, uint32_t);
C_COARSE_HUFFMAN(ui32, ui32, ui32, uint32_t, uint32_t, uint32_t);
C_COARSE_HUFFMAN(fp32, ui32, ui32, float, uint32_t, uint32_t);
C_COARSE_HUFFMAN(ui8, ui64, ui32, uint8_t, uint64_t, uint32_t);
C_COARSE_HUFFMAN(ui16, ui64, ui32, uint16_t, uint64_t, uint32_t);
C_COARSE_HUFFMAN(ui32, ui64, ui32, uint32_t, uint64_t, uint32_t);
C_COARSE_HUFFMAN(fp32, ui64, ui32, float, uint64_t, uint32_t);
C_COARSE_HUFFMAN(ui8, ull, ui32, uint8_t, unsigned long long, uint32_t);
C_COARSE_HUFFMAN(ui16, ull, ui32, uint16_t, unsigned long long, uint32_t);
C_COARSE_HUFFMAN(ui32, ull, ui32, uint32_t, unsigned long long, uint32_t);
C_COARSE_HUFFMAN(fp32, ull, ui32, float, unsigned long long, uint32_t);

#undef C_COARSE_HUFFMAN

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CPP_LORENZOI(Tliteral, Eliteral, FPliteral, T, E, FP)                                                          \
    template <>                                                                                                        \
    cusz_error_status cusz::cpplaunch_construct_LorenzoI_proto<T, E, FP>(                                              \
        T* const data, dim3 const len3, double const eb, int const radius, E* const eq, dim3 const eq_len3,            \
        T* const anchor, dim3 const anchor_len3, T* outlier, uint32_t* outlier_idx, float* time_elapsed,               \
        cudaStream_t stream)                                                                                           \
    {                                                                                                                  \
        return claunch_construct_LorenzoI_proto_T##Tliteral##_E##Eliteral##_FP##FPliteral(                             \
            data, len3, anchor, anchor_len3, eq, eq_len3, outlier, eb, radius, time_elapsed, stream);                  \
    }                                                                                                                  \
                                                                                                                       \
    template <>                                                                                                        \
    cusz_error_status cusz::cpplaunch_reconstruct_LorenzoI_proto<T, E, FP>(                                            \
        E * eq, dim3 const eq_len3, T* anchor, dim3 const anchor_len3, T* outlier, uint32_t* outlier_idx,              \
        double const eb, int const radius, T* xdata, dim3 const xdata_len3, float* time_elapsed, cudaStream_t stream)  \
    {                                                                                                                  \
        return claunch_reconstruct_LorenzoI_proto_T##Tliteral##_E##Eliteral##_FP##FPliteral(                           \
            xdata, xdata_len3, anchor, anchor_len3, eq, eq_len3, outlier, eb, radius, time_elapsed, stream);           \
    }                                                                                                                  \
                                                                                                                       \
    template <>                                                                                                        \
    cusz_error_status cusz::cpplaunch_construct_LorenzoI<T, E, FP>(                                                    \
        T* const data, dim3 const len3, double const eb, int const radius, E* const eq, dim3 const eq_len3,            \
        T* const anchor, dim3 const anchor_len3, T* outlier, uint32_t* outlier_idx, float* time_elapsed,               \
        cudaStream_t stream)                                                                                           \
    {                                                                                                                  \
        return claunch_construct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                                   \
            data, len3, anchor, anchor_len3, eq, eq_len3, outlier, eb, radius, time_elapsed, stream);                  \
    }                                                                                                                  \
                                                                                                                       \
    template <>                                                                                                        \
    cusz_error_status cusz::cpplaunch_reconstruct_LorenzoI<T, E, FP>(                                                  \
        E * eq, dim3 const eq_len3, T* anchor, dim3 const anchor_len3, T* outlier, uint32_t* outlier_idx,              \
        double const eb, int const radius, T* xdata, dim3 const xdata_len3, float* time_elapsed, cudaStream_t stream)  \
    {                                                                                                                  \
        return claunch_reconstruct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                                 \
            xdata, xdata_len3, anchor, anchor_len3, eq, eq_len3, outlier, eb, radius, time_elapsed, stream);           \
    }                                                                                                                  \
                                                                                                                       \
    template <>                                                                                                        \
    cusz_error_status cusz::experimental::cpplaunch_construct_LorenzoI_var<T, E, FP>(                                  \
        T* const data, dim3 const len3, double const eb, E* delta, bool* signum, float* time_elapsed,                  \
        cudaStream_t stream)                                                                                           \
    {                                                                                                                  \
        return claunch_construct_LorenzoI_var_T##Tliteral##_E##Eliteral##_FP##FPliteral(                               \
            data, delta, signum, len3, eb, time_elapsed, stream);                                                      \
    }                                                                                                                  \
                                                                                                                       \
    template <>                                                                                                        \
    cusz_error_status cusz::experimental::cpplaunch_reconstruct_LorenzoI_var<T, E, FP>(                                \
        E * delta, bool* signum, dim3 const len3, double const eb, T* xdata, float* time_elapsed, cudaStream_t stream) \
    {                                                                                                                  \
        return claunch_reconstruct_LorenzoI_var_T##Tliteral##_E##Eliteral##_FP##FPliteral(                             \
            signum, delta, xdata, len3, eb, time_elapsed, stream);                                                     \
    }

CPP_LORENZOI(fp32, ui8, fp32, float, uint8_t, float);
CPP_LORENZOI(fp32, ui16, fp32, float, uint16_t, float);
CPP_LORENZOI(fp32, ui32, fp32, float, uint32_t, float);
CPP_LORENZOI(fp32, fp32, fp32, float, float, float);

CPP_LORENZOI(fp64, ui8, fp64, double, uint8_t, double);
CPP_LORENZOI(fp64, ui16, fp64, double, uint16_t, double);
CPP_LORENZOI(fp64, ui32, fp64, double, uint32_t, double);
CPP_LORENZOI(fp64, fp32, fp64, double, float, double);

#undef CPP_LORENZOI

#define CPP_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                                    \
    template <>                                                                                                 \
    cusz_error_status cusz::cpplaunch_construct_Spline3<T, E, FP>(                                              \
        bool NO_R_SEPARATE, T* data, dim3 const len3, T* anchor, dim3 const an_len3, E* eq, dim3 const ec_len3, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream)                            \
    {                                                                                                           \
        return claunch_construct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                             \
            NO_R_SEPARATE, data, len3, anchor, an_len3, eq, ec_len3, eb, radius, time_elapsed, stream);         \
    }                                                                                                           \
                                                                                                                \
    template <>                                                                                                 \
    cusz_error_status cusz::cpplaunch_reconstruct_Spline3<T, E, FP>(                                            \
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

#define CPP_HIST(Tliteral, T)                                                                                       \
    template <>                                                                                                     \
    cusz_error_status cusz::cpplaunch_histogram<T>(                                                                 \
        T * in_data, size_t in_len, uint32_t * out_freq, int num_buckets, float* milliseconds, cudaStream_t stream) \
    {                                                                                                               \
        return claunch_histogram_T##Tliteral(in_data, in_len, out_freq, num_buckets, milliseconds, stream);         \
    }

CPP_HIST(ui8, uint8_t)
CPP_HIST(ui16, uint16_t)
CPP_HIST(ui32, uint32_t)
CPP_HIST(ui64, uint64_t)

#undef CPP_HIST

#define CPP_COARSE_HUFFMAN_ENCODE(Tliteral, Hliteral, Mliteral, T, H, M)                                               \
    template <>                                                                                                        \
    cusz_error_status cusz::cpplaunch_coarse_grained_Huffman_encoding<T, H, M>(                                        \
        T * uncompressed, H * d_internal_coded, size_t const len, uint32_t* d_freq, H* d_book, int const booklen,      \
        H* d_bitstream, M* d_par_metadata, M* h_par_metadata, int const sublen, int const pardeg, int numSMs,          \
        uint8_t** out_compressed, size_t* out_compressed_len, float* time_lossless, cudaStream_t stream)               \
    {                                                                                                                  \
        return claunch_coarse_grained_Huffman_encoding_T##Tliteral##_H##Hliteral##_M##Mliteral(                        \
            uncompressed, d_internal_coded, len, d_freq, d_book, booklen, d_bitstream, d_par_metadata, h_par_metadata, \
            sublen, pardeg, numSMs, out_compressed, out_compressed_len, time_lossless, stream);                        \
    }

CPP_COARSE_HUFFMAN_ENCODE(ui8, ui32, ui32, uint8_t, uint32_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE(ui16, ui32, ui32, uint16_t, uint32_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE(ui32, ui32, ui32, uint32_t, uint32_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE(fp32, ui32, ui32, float, uint32_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE(ui8, ui64, ui32, uint8_t, uint64_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE(ui16, ui64, ui32, uint16_t, uint64_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE(ui32, ui64, ui32, uint32_t, uint64_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE(fp32, ui64, ui32, float, uint64_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE(ui8, ull, ui32, uint8_t, unsigned long long, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE(ui16, ull, ui32, uint16_t, unsigned long long, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE(ui32, ull, ui32, uint32_t, unsigned long long, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE(fp32, ull, ui32, float, unsigned long long, uint32_t);

#undef CPP_COARSE_HUFFMAN_ENCODE

#define CPP_COARSE_HUFFMAN_ENCODE_rev1(Tliteral, Hliteral, Mliteral, T, H, M)                                         \
    template <>                                                                                                       \
    cusz_error_status cusz::cpplaunch_coarse_grained_Huffman_encoding_rev1<T, H, M>(                                  \
        T * uncompressed, size_t const len, hf_book* book_desc, hf_bitstream* bitstream_desc,                         \
        uint8_t** out_compressed, size_t* out_compressed_len, float* time_lossless, cudaStream_t stream)              \
    {                                                                                                                 \
        return claunch_coarse_grained_Huffman_encoding_rev1_T##Tliteral##_H##Hliteral##_M##Mliteral(                  \
            uncompressed, len, book_desc, bitstream_desc, out_compressed, out_compressed_len, time_lossless, stream); \
    }

CPP_COARSE_HUFFMAN_ENCODE_rev1(ui8, ui32, ui32, uint8_t, uint32_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE_rev1(ui16, ui32, ui32, uint16_t, uint32_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE_rev1(ui32, ui32, ui32, uint32_t, uint32_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE_rev1(fp32, ui32, ui32, float, uint32_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE_rev1(ui8, ui64, ui32, uint8_t, uint64_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE_rev1(ui16, ui64, ui32, uint16_t, uint64_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE_rev1(ui32, ui64, ui32, uint32_t, uint64_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE_rev1(fp32, ui64, ui32, float, uint64_t, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE_rev1(ui8, ull, ui32, uint8_t, unsigned long long, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE_rev1(ui16, ull, ui32, uint16_t, unsigned long long, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE_rev1(ui32, ull, ui32, uint32_t, unsigned long long, uint32_t);
CPP_COARSE_HUFFMAN_ENCODE_rev1(fp32, ull, ui32, float, unsigned long long, uint32_t);

#undef CPP_COARSE_HUFFMAN_ENCODE_rev1

#define CPP_COARSE_HUFFMAN_DECODE(Tliteral, Hliteral, Mliteral, T, H, M)                                      \
    template <>                                                                                               \
    cusz_error_status cusz::cpplaunch_coarse_grained_Huffman_decoding<T, H, M>(                               \
        H * d_bitstream, uint8_t * d_revbook, int const revbook_nbyte, M* d_par_nbit, M* d_par_entry,         \
        int const sublen, int const pardeg, T* out_decompressed, float* time_lossless, cudaStream_t stream)   \
    {                                                                                                         \
        return claunch_coarse_grained_Huffman_decoding_T##Tliteral##_H##Hliteral##_M##Mliteral(               \
            d_bitstream, d_revbook, revbook_nbyte, d_par_nbit, d_par_entry, sublen, pardeg, out_decompressed, \
            time_lossless, stream);                                                                           \
    }

CPP_COARSE_HUFFMAN_DECODE(ui8, ui32, ui32, uint8_t, uint32_t, uint32_t);
CPP_COARSE_HUFFMAN_DECODE(ui16, ui32, ui32, uint16_t, uint32_t, uint32_t);
CPP_COARSE_HUFFMAN_DECODE(ui32, ui32, ui32, uint32_t, uint32_t, uint32_t);
CPP_COARSE_HUFFMAN_DECODE(fp32, ui32, ui32, float, uint32_t, uint32_t);
CPP_COARSE_HUFFMAN_DECODE(ui8, ui64, ui32, uint8_t, uint64_t, uint32_t);
CPP_COARSE_HUFFMAN_DECODE(ui16, ui64, ui32, uint16_t, uint64_t, uint32_t);
CPP_COARSE_HUFFMAN_DECODE(ui32, ui64, ui32, uint32_t, uint64_t, uint32_t);
CPP_COARSE_HUFFMAN_DECODE(fp32, ui64, ui32, float, uint64_t, uint32_t);
CPP_COARSE_HUFFMAN_DECODE(ui8, ull, ui32, uint8_t, unsigned long long, uint32_t);
CPP_COARSE_HUFFMAN_DECODE(ui16, ull, ui32, uint16_t, unsigned long long, uint32_t);
CPP_COARSE_HUFFMAN_DECODE(ui32, ull, ui32, uint32_t, unsigned long long, uint32_t);
CPP_COARSE_HUFFMAN_DECODE(fp32, ull, ui32, float, unsigned long long, uint32_t);

#undef CPP_COARSE_HUFFMAN_DECODE
