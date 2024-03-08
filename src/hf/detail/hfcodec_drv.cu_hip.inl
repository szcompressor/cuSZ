/**
 * @file hfcodec_drv.cu_hip.inl
 * @author Jiannan Tian
 * @brief kernel wrappers; launching Huffman kernels
 * @version 0.3
 * @date 2022-11-02
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "cusz/type.h"
#include "hf/hfbk.cu.hh"
#include "hf/hfcodec.hh"
#include "hf/hfcxx_array.hh"
#include "hf/hfcxx_module.hh"

template <typename E, typename H, typename M>
void psz::hf_encode_coarse_rev2(
    E* uncompressed, size_t const len, hf_book* book_desc,
    hf_bitstream* bitstream_desc, size_t* outlen_nbit, size_t* outlen_ncell,
    float* time_lossless, void* stream)
{
  H* d_buffer = (H*)bitstream_desc->buffer;
  H* d_bitstream = (H*)bitstream_desc->bitstream;
  H* d_book = (H*)book_desc->book;
  auto const booklen = (size_t)book_desc->bklen;
  // to make compiler happy
  auto sublen = (size_t)bitstream_desc->sublen;
  auto pardeg = (size_t)bitstream_desc->pardeg;
  hfpar_description hfpar{sublen, pardeg};
  int const numSMs = bitstream_desc->numSMs;

  auto d_par_nbit = (M*)bitstream_desc->d_metadata->bits;
  auto d_par_ncell = (M*)bitstream_desc->d_metadata->cells;
  auto d_par_entry = (M*)bitstream_desc->d_metadata->entries;

  auto h_par_nbit = (M*)bitstream_desc->h_metadata->bits;
  auto h_par_ncell = (M*)bitstream_desc->h_metadata->cells;
  auto h_par_entry = (M*)bitstream_desc->h_metadata->entries;

  _2403::hf_encode_coarse_phase1<E, H>(
      {uncompressed, len}, {d_book, booklen}, numSMs, {d_buffer, len},
      time_lossless, stream);

  _2403::hf_encode_coarse_phase2<H, M>(
      {d_buffer, len}, hfpar, {d_buffer, len /* placeholder */},
      {d_par_nbit, pardeg}, {d_par_ncell, pardeg}, time_lossless, stream);

  _2403::hf_encode_coarse_phase3(
      {d_par_nbit, pardeg}, {d_par_ncell, pardeg}, {d_par_entry, pardeg},
      hfpar, {h_par_nbit, pardeg}, {h_par_ncell, pardeg},
      {h_par_entry, pardeg}, outlen_nbit, outlen_ncell, nullptr, stream);

  _2403::hf_encode_coarse_phase4<H, M>(
      {d_buffer, len}, {d_par_entry, pardeg}, {d_par_ncell, pardeg}, hfpar,
      {d_bitstream, len}, time_lossless, stream);
}

template <typename E, typename H, typename M>
void psz::hf_decode_coarse(
    H* d_bitstream, uint8_t* d_revbook, size_t const revbook_nbyte,
    M* d_par_nbit, M* d_par_entry, size_t const sublen, size_t const pardeg,
    E* out, float* time_lossless, void* stream)
{
  _2403::hf_decode_coarse<E, H, M>(
      {d_bitstream, 0}, hfarray_cxx<uint8_t>{d_revbook, revbook_nbyte},
      {d_par_nbit, pardeg}, {d_par_entry, pardeg}, {sublen, pardeg}, {out, 0},
      time_lossless, stream);
}
