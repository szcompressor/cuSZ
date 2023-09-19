/**
 * @file hfcodec.hh
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

#include <cstdint>
#include <cstdlib>

#include "hfstruct.h"
#include "hfword.hh"

namespace psz {

template <typename T, typename H, typename M>
void hf_encode_coarse_rev2(
    T* uncompressed, size_t const len, hf_book* book_desc,
    hf_bitstream* bitstream_desc, size_t* outlen_nbit, size_t* outlen_ncell,
    float* time_lossless, void* stream);

template <typename T, typename H, typename M>
void hf_decode_coarse(
    H* d_bitstream, uint8_t* d_revbook, int const revbook_nbyte, M* d_par_nbit,
    M* d_par_entry, int const sublen, int const pardeg, T* out_decompressed,
    float* time_lossless, void* stream);

}  // namespace psz

#endif /* ABAACE49_2C9E_4E3C_AEFF_B016276142E1 */
