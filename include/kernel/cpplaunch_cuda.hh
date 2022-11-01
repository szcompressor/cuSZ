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

#include "../cusz/type.h"
#include "../hf/hf_struct.h"

namespace cusz {

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

#endif
