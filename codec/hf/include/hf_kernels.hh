#ifndef ED744237_D186_4707_B7FC_1B931FFC5CBB
#define ED744237_D186_4707_B7FC_1B931FFC5CBB

#include <cstdint>

#include "hf.h"
#include "phf_array.hh"

namespace phf::cuhip {

// @brief a namespace-like class for batch template instantiations; a rewrite of
// hf_kernels.{hh,cc}; all the included wrapped kernels/methods are `static`
// @tparam E input type
// @tparam H intermediate type for Huffman coding
template <typename E, typename H>
class modules {
  // metadata, e.g., saved index for parallel operations
  using M = PHF_METADATA;

 public:
  static void GPU_coarse_encode_phase1(
      E* in_data, const size_t data_len, H* in_book, const u4 book_len, const int numSMs,
      H* out_bitstream, void* stream);

  static void GPU_coarse_encode_phase2(
      H* in_data, const size_t data_len, phf::par_config hfpar, H* deflated, M* par_nbit,
      M* par_ncell, void* stream);

  static void GPU_fine_encode_phase1_2(
      E* in, const size_t len, H* book, const u4 bklen, H* bitstream, M* par_nbit, M* par_ncell,
      const u4 nblock, E* brval, u4* bridx, u4* brnum, void* stream);

  static void GPU_coarse_encode_phase3_sync(
      phf::par_config hfpar, M* d_par_nbit, M* h_par_nbit, M* d_par_ncell, M* h_par_ncell,
      M* d_par_entry, M* h_par_entry, size_t* outlen_nbit, size_t* outlen_ncell,
      float* time_cpu_time, void* stream);

  static void GPU_coarse_encode_phase4(
      H* in_buf, const size_t len, M* par_entry, M* par_ncell, phf::par_config hfpar, H* bitstream,
      const size_t max_bitstream_len, void* stream);

  static void GPU_coarse_encode(
      E* in_data, size_t data_len, H* in_book, u4 book_len, int numSMs, phf::par_config hfpar,
      // internal buffers
      H* d_scratch4, M* d_par_nbit, M* h_par_nbit, M* d_par_ncell, M* h_par_ncell, M* d_par_entry,
      M* h_par_entry, H* d_bitstream4, size_t bitstream_max_len,
      // output
      size_t* out_total_nbit, size_t* out_total_ncell, void* stream);

  static void GPU_fine_encode(
      E* in_data, size_t data_len, H* in_book, u4 book_len, phf::par_config hfpar,
      // internal buffers
      H* d_scratch4, M* d_par_nbit, M* h_par_nbit, M* d_par_ncell, M* h_par_ncell, M* d_par_entry,
      M* h_par_entry, H* d_bitstream4, size_t bitstream_max_len, E* d_brval, u4* d_bridx,
      u4* d_brnum,
      // output
      size_t* out_total_nbit, size_t* out_total_ncell, void* stream);

  static void GPU_coarse_decode(
      H* in_bitstream, uint8_t* in_revbook, size_t const revbook_len, M* in_par_nbit,
      M* in_par_entry, size_t const sublen, size_t const pardeg, E* out_decoded, void* stream);

  static void GPU_scatter(E* val, u4* idx, const u4 h_num, E* out, void* stream);
};

}  // namespace phf::cuhip

#endif /* ED744237_D186_4707_B7FC_1B931FFC5CBB */
