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
#include "../kernel/claunch_cuda_proto.h"

namespace cusz {

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_construct_LorenzoI_proto(
    bool         NO_R_SEPARATE,
    T* const     data,
    dim3 const   len3,
    T* const     anchor,
    dim3 const   placeholder_1,
    E* const     errctrl,
    dim3 const   placeholder_2,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_reconstruct_LorenzoI_proto(
    T*           xdata,
    dim3 const   len3,
    T*           anchor,
    dim3 const   placeholder_1,
    E*           errctrl,
    dim3 const   placeholder_2,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_construct_LorenzoI(
    bool         NO_R_SEPARATE,
    T* const     data,
    dim3 const   len3,
    T* const     anchor,
    dim3 const   placeholder_1,
    E* const     errctrl,
    dim3 const   placeholder_2,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_reconstruct_LorenzoI(
    T*           xdata,
    dim3 const   len3,
    T*           anchor,
    dim3 const   placeholder_1,
    E*           errctrl,
    dim3 const   placeholder_2,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_construct_Spline3(
    bool         NO_R_SEPARATE,
    T*           data,
    dim3 const   len3,
    T*           anchor,
    dim3 const   an_len3,
    E*           errctrl,
    dim3 const   ec_len3,
    double const eb,
    int const    radius,
    float*       time_elapsed,
    cudaStream_t stream);

template <typename T, typename E, typename FP>
cusz_error_status cpplaunch_reconstruct_Spline3(
    T*           xdata,
    dim3 const   len3,
    T*           anchor,
    dim3 const   an_len3,
    E*           errctrl,
    dim3 const   ec_len3,
    double const eb,
    int const    radius,
    float*       time_elapsed,
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

#define CPP_CONSTRUCT_LORENZOI_PROTO(Tliteral, Eliteral, FPliteral, T, E, FP)                                   \
    template <>                                                                                                 \
    cusz_error_status cusz::cpplaunch_construct_LorenzoI_proto<T, E, FP>(                                       \
        bool NO_R_SEPARATE, T* const data, dim3 const len3, T* const anchor, dim3 const placeholder_1,          \
        E* const errctrl, dim3 const placeholder_2, double const eb, int const radius, float* time_elapsed,     \
        cudaStream_t stream)                                                                                    \
    {                                                                                                           \
        return claunch_construct_LorenzoI_proto_T##Tliteral##_E##Eliteral##_FP##FPliteral(                      \
            NO_R_SEPARATE, data, len3, anchor, placeholder_1, errctrl, placeholder_2, eb, radius, time_elapsed, \
            stream);                                                                                            \
    }

CPP_CONSTRUCT_LORENZOI_PROTO(fp32, ui8, fp32, float, uint8_t, float);
CPP_CONSTRUCT_LORENZOI_PROTO(fp32, ui16, fp32, float, uint16_t, float);
CPP_CONSTRUCT_LORENZOI_PROTO(fp32, ui32, fp32, float, uint32_t, float);
CPP_CONSTRUCT_LORENZOI_PROTO(fp32, fp32, fp32, float, float, float);

#undef CPP_CONSTRUCT_LORENZOI_PROTO

#define CPP_RECONSTRUCT_LORENZOI_PROTO(Tliteral, Eliteral, FPliteral, T, E, FP)                                \
    template <>                                                                                                \
    cusz_error_status cusz::cpplaunch_reconstruct_LorenzoI_proto<T, E, FP>(                                    \
        T * xdata, dim3 const len3, T* anchor, dim3 const placeholder_1, E* errctrl, dim3 const placeholder_2, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream)                           \
    {                                                                                                          \
        return claunch_reconstruct_LorenzoI_proto_T##Tliteral##_E##Eliteral##_FP##FPliteral(                   \
            xdata, len3, anchor, placeholder_1, errctrl, placeholder_2, eb, radius, time_elapsed, stream);     \
    }

CPP_RECONSTRUCT_LORENZOI_PROTO(fp32, ui8, fp32, float, uint8_t, float);
CPP_RECONSTRUCT_LORENZOI_PROTO(fp32, ui16, fp32, float, uint16_t, float);
CPP_RECONSTRUCT_LORENZOI_PROTO(fp32, ui32, fp32, float, uint32_t, float);
CPP_RECONSTRUCT_LORENZOI_PROTO(fp32, fp32, fp32, float, float, float);

#undef CPP_RECONSTRUCT_LORENZOI_PROTO

#define CPP_CONSTRUCT_LORENZOI(Tliteral, Eliteral, FPliteral, T, E, FP)                                         \
    template <>                                                                                                 \
    cusz_error_status cusz::cpplaunch_construct_LorenzoI<T, E, FP>(                                             \
        bool NO_R_SEPARATE, T* const data, dim3 const len3, T* const anchor, dim3 const placeholder_1,          \
        E* const errctrl, dim3 const placeholder_2, double const eb, int const radius, float* time_elapsed,     \
        cudaStream_t stream)                                                                                    \
    {                                                                                                           \
        return claunch_construct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                            \
            NO_R_SEPARATE, data, len3, anchor, placeholder_1, errctrl, placeholder_2, eb, radius, time_elapsed, \
            stream);                                                                                            \
    }

CPP_CONSTRUCT_LORENZOI(fp32, ui8, fp32, float, uint8_t, float);
CPP_CONSTRUCT_LORENZOI(fp32, ui16, fp32, float, uint16_t, float);
CPP_CONSTRUCT_LORENZOI(fp32, ui32, fp32, float, uint32_t, float);
CPP_CONSTRUCT_LORENZOI(fp32, fp32, fp32, float, float, float);

#undef CPP_CONSTRUCT_LORENZOI

#define CPP_RECONSTRUCT_LORENZOI(Tliteral, Eliteral, FPliteral, T, E, FP)                                      \
    template <>                                                                                                \
    cusz_error_status cusz::cpplaunch_reconstruct_LorenzoI<T, E, FP>(                                          \
        T * xdata, dim3 const len3, T* anchor, dim3 const placeholder_1, E* errctrl, dim3 const placeholder_2, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream)                           \
    {                                                                                                          \
        return claunch_reconstruct_LorenzoI_T##Tliteral##_E##Eliteral##_FP##FPliteral(                         \
            xdata, len3, anchor, placeholder_1, errctrl, placeholder_2, eb, radius, time_elapsed, stream);     \
    }

CPP_RECONSTRUCT_LORENZOI(fp32, ui8, fp32, float, uint8_t, float);
CPP_RECONSTRUCT_LORENZOI(fp32, ui16, fp32, float, uint16_t, float);
CPP_RECONSTRUCT_LORENZOI(fp32, ui32, fp32, float, uint32_t, float);
CPP_RECONSTRUCT_LORENZOI(fp32, fp32, fp32, float, float, float);

#undef CPP_RECONSTRUCT_LORENZOI

#define CPP_CONSTRUCT_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                               \
    template <>                                                                                                      \
    cusz_error_status cusz::cpplaunch_construct_Spline3<T, E, FP>(                                                   \
        bool NO_R_SEPARATE, T* data, dim3 const len3, T* anchor, dim3 const an_len3, E* errctrl, dim3 const ec_len3, \
        double const eb, int const radius, float* time_elapsed, cudaStream_t stream)                                 \
    {                                                                                                                \
        return claunch_construct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                                  \
            NO_R_SEPARATE, data, len3, anchor, an_len3, errctrl, ec_len3, eb, radius, time_elapsed, stream);         \
    }

CPP_CONSTRUCT_SPLINE3(fp32, ui8, fp32, float, uint8_t, float);
CPP_CONSTRUCT_SPLINE3(fp32, ui16, fp32, float, uint16_t, float);
CPP_CONSTRUCT_SPLINE3(fp32, ui32, fp32, float, uint32_t, float);
CPP_CONSTRUCT_SPLINE3(fp32, fp32, fp32, float, float, float);

#undef CPP_CONSTRUCT_SPLINE3

#define CPP_RECONSTRUCT_SPLINE3(Tliteral, Eliteral, FPliteral, T, E, FP)                                            \
    template <>                                                                                                     \
    cusz_error_status cusz::cpplaunch_reconstruct_Spline3<T, E, FP>(                                                \
        T * xdata, dim3 const len3, T* anchor, dim3 const an_len3, E* errctrl, dim3 const ec_len3, double const eb, \
        int const radius, float* time_elapsed, cudaStream_t stream)                                                 \
    {                                                                                                               \
        return claunch_reconstruct_Spline3_T##Tliteral##_E##Eliteral##_FP##FPliteral(                               \
            xdata, len3, anchor, an_len3, errctrl, ec_len3, eb, radius, time_elapsed, stream);                      \
    }

CPP_RECONSTRUCT_SPLINE3(fp32, ui8, fp32, float, uint8_t, float);
CPP_RECONSTRUCT_SPLINE3(fp32, ui16, fp32, float, uint16_t, float);
CPP_RECONSTRUCT_SPLINE3(fp32, ui32, fp32, float, uint32_t, float);
CPP_RECONSTRUCT_SPLINE3(fp32, fp32, fp32, float, float, float);

#undef CPP_RECONSTRUCT_SPLINE3

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
