/**
 * @file cpplaunch_cuda.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-07-27
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef COMPONENT_CALL_KERNEL_HH
#define COMPONENT_CALL_KERNEL_HH

#include "../hf/hf_struct.h"
#include "../kernel/claunch_cuda.h"

namespace cusz {

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_construct_LorenzoI_proto(
    T* const     data,          // input
    dim3 const   data_len3,     //
    double const eb,            // input (config)
    int const    radius,        //
    E* const     eq,            // output
    dim3 const   eq_len3,       //
    T* const     anchor,        //
    dim3 const   anchor_len3,   //
    T*           outlier,       //
    uint32_t*    outlier_idx,   //
    float*       time_elapsed,  // optional
    cudaStream_t stream);       //

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_reconstruct_LorenzoI_proto(
    E*           eq,            // input
    dim3 const   eq_len3,       //
    T*           anchor,        //
    dim3 const   anchor_len3,   //
    T*           outlier,       //
    uint32_t*    outlier_idx,   //
    double const eb,            // input (config)
    int const    radius,        //
    T*           xdata,         // output
    dim3 const   xdata_len3,    //
    float*       time_elapsed,  // optional
    cudaStream_t stream);       //

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_construct_LorenzoI(
    T* const     data,          // input
    dim3 const   data_len3,     //
    double const eb,            // input (config)
    int const    radius,        //
    E* const     eq,            // output
    dim3 const   eq_len3,       //
    T* const     anchor,        //
    dim3 const   anchor_len3,   //
    T*           outlier,       //
    uint32_t*    outlier_idx,   //
    float*       time_elapsed,  // optional
    cudaStream_t stream);       //

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_reconstruct_LorenzoI(
    E*           eq,            // input
    dim3 const   eq_len3,       //
    T*           anchor,        //
    dim3 const   anchor_len3,   //
    T*           outlier,       //
    uint32_t*    outlier_idx,   //
    double const eb,            // input (config)
    int const    radius,        //
    T*           xdata,         // output
    dim3 const   xdata_len3,    //
    float*       time_elapsed,  // optional
    cudaStream_t stream);       //

// 22-10-27 revise later
template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_construct_Spline3(
    bool         NO_R_SEPARATE,
    T*           data,
    dim3 const   len3,
    T*           anchor,
    dim3 const   an_len3,
    E*           eq,
    dim3 const   ec_len3,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

// 22-10-27 revise later
template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_reconstruct_Spline3(
    T*           xdata,
    dim3 const   len3,
    T*           anchor,
    dim3 const   an_len3,
    E*           eq,
    dim3 const   ec_len3,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

template <typename T>
cusz_error_status cpplaunch_histogram(
    T*           in_data,
    size_t       in_len,
    uint32_t*    out_freq,
    int          num_buckets,
    float*       milliseconds,
    cudaStream_t stream);

template <typename T, typename H, typename M>
cusz_error_status cpplaunch_coarse_grained_Huffman_encoding(
    T*           uncompressed,
    H*           d_internal_coded,
    size_t const len,
    uint32_t*    d_freq,
    H*           d_book,
    int const    booklen,
    H*           d_bitstream,
    M*           d_par_metadata,
    M*           h_par_metadata,
    int const    sublen,
    int const    pardeg,
    int          numSMs,
    uint8_t**    out_compressed,
    size_t*      out_compressed_len,
    float*       time_lossless,
    cudaStream_t stream);

template <typename T, typename H, typename M>
cusz_error_status cpplaunch_coarse_grained_Huffman_encoding_rev1(
    T*            uncompressed,
    size_t const  len,
    hf_book*      book_desc,
    hf_bitstream* bitstream_desc,
    uint8_t**     out_compressed,
    size_t*       out_compressed_len,
    float*        time_lossless,
    cudaStream_t  stream);

template <typename T, typename H, typename M>
cusz_error_status cpplaunch_coarse_grained_Huffman_decoding(
    H*           d_bitstream,
    uint8_t*     d_revbook,
    int const    revbook_nbyte,
    M*           d_par_nbit,
    M*           d_par_entry,
    int const    sublen,
    int const    pardeg,
    T*           out_decompressed,
    float*       time_lossless,
    cudaStream_t stream);

}  // namespace cusz

#define CPP_LORENZOI(Tliteral, Eliteral, FPliteral, T, E, FP)                                                         \
    template <>                                                                                                       \
    cusz_error_status cusz::cpplaunch_construct_LorenzoI_proto<T, E, FP>(                                             \
        T* const data, dim3 const len3, double const eb, int const radius, E* const eq, dim3 const eq_len3,           \
        T* const anchor, dim3 const anchor_len3, T* outlier, uint32_t* outlier_idx, float* time_elapsed,              \
        cudaStream_t stream)                                                                                          \
    {                                                                                                                 \
        return claunch_construct_LorenzoI_proto_T##Tliteral##_E##Eliteral##_FP##FPliteral(                            \
            data, len3, anchor, anchor_len3, eq, eq_len3, outlier, eb, radius, time_elapsed, stream);                 \
    }                                                                                                                 \
                                                                                                                      \
    template <>                                                                                                       \
    cusz_error_status cusz::cpplaunch_reconstruct_LorenzoI_proto<T, E, FP>(                                           \
        E * eq, dim3 const eq_len3, T* anchor, dim3 const anchor_len3, T* outlier, uint32_t* outlier_idx,             \
        double const eb, int const radius, T* xdata, dim3 const xdata_len3, float* time_elapsed, cudaStream_t stream) \
    {                                                                                                                 \
        return claunch_reconstruct_LorenzoI_proto_T##Tliteral##_E##Eliteral##_FP##FPliteral(                          \
            xdata, xdata_len3, anchor, anchor_len3, eq, eq_len3, outlier, eb, radius, time_elapsed, stream);          \
    }                                                                                                                 \
                                                                                                                      \
    template <>                                                                                                       \
    cusz_error_status cusz::cpplaunch_construct_LorenzoI<T, E, FP>(                                                   \
        T* const data, dim3 const len3, double const eb, int const radius, E* const eq, dim3 const eq_len3,           \
        T* const anchor, dim3 const anchor_len3, T* outlier, uint32_t* outlier_idx, float* time_elapsed,              \
        cudaStream_t stream)                                                                                          \
    {                                                                                                                 \
        return claunch_construct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                                  \
            data, len3, anchor, anchor_len3, eq, eq_len3, outlier, eb, radius, time_elapsed, stream);                 \
    }                                                                                                                 \
                                                                                                                      \
    template <>                                                                                                       \
    cusz_error_status cusz::cpplaunch_reconstruct_LorenzoI<T, E, FP>(                                                 \
        E * eq, dim3 const eq_len3, T* anchor, dim3 const anchor_len3, T* outlier, uint32_t* outlier_idx,             \
        double const eb, int const radius, T* xdata, dim3 const xdata_len3, float* time_elapsed, cudaStream_t stream) \
    {                                                                                                                 \
        return claunch_reconstruct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                                \
            xdata, xdata_len3, anchor, anchor_len3, eq, eq_len3, outlier, eb, radius, time_elapsed, stream);          \
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

#endif
