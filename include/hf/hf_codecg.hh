/**
 * @file launch_lossless.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-06-13
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef ABAACE49_2C9E_4E3C_AEFF_B016276142E1
#define ABAACE49_2C9E_4E3C_AEFF_B016276142E1

#include <stdint.h>
#include <stdlib.h>

#include "hf_struct.h"

template <int WIDTH>
struct PackedWordByWidth;

template <>
struct PackedWordByWidth<4> {
    uint32_t word : 24;
    uint32_t bits : 8;
};

template <>
struct PackedWordByWidth<8> {
    uint64_t word : 56;
    uint64_t bits : 8;
};

namespace asz {

template <typename T, typename H, typename M>
void hf_encode_coarse(
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
    uint8_t*&    out_compressed,
    size_t&      out_compressed_len,
    float&       time_lossless,
    cudaStream_t stream);

template <typename T, typename H, typename M>
void hf_encode_coarse_rev1(
    T*            uncompressed,
    size_t const  len,
    hf_book*      book_desc,
    hf_bitstream* bitstream_desc,
    uint8_t*&     out_compressed,      // 22-10-12 buggy
    size_t&       out_compressed_len,  // 22-10-12 buggy
    float&        time_lossless,
    cudaStream_t  stream);

template <typename T, typename H, typename M>
void hf_decode_coarse(
    H*           d_bitstream,
    uint8_t*     d_revbook,
    int const    revbook_nbyte,
    M*           d_par_nbit,
    M*           d_par_entry,
    int const    sublen,
    int const    pardeg,
    T*           out_decompressed,
    float&       time_lossless,
    cudaStream_t stream);

}  // namespace asz

#endif /* ABAACE49_2C9E_4E3C_AEFF_B016276142E1 */
