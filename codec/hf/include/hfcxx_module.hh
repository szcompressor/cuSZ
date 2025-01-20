#ifndef ED744237_D186_4707_B7FC_1B931FFC5CBB
#define ED744237_D186_4707_B7FC_1B931FFC5CBB

#include <cstdint>

#include "hf.h"
#include "phf_array.hh"

namespace phf::cuhip {

/**
 * @brief a namespace-like class for batch template instantiations; a rewrite
 * of hfcodec.{hh,cc}; all the included wrapped kernels/methods are `static`
 *
 * @tparam E input type, e.g., error-quantization code in psz
 * @tparam H intermediate type for Huffman coding
 */
template <typename E, typename H>
class modules {
  // metadata, e.g., saved index for parallel operations
  using M = PHF_METADATA;

 public:
  static void GPU_coarse_encode_phase1(
      phf::array<E> in, phf::array<H> book, const int numSMs, phf::array<H> out, void* stream);

  static void GPU_coarse_encode_phase1_collect_metadata(
      phf::array<E> in, phf::array<H> book, const int numSMs, phf::array<H> out,
      phf::array<M> par_nbit, phf::array<M> par_ncell, phf::par_config hfpar, void* stream);

  static void GPU_coarse_encode_phase2(
      phf::array<H> in, phf::par_config hfpar, phf::array<H> deflated, phf::array<M> par_nbit,
      phf::array<M> par_ncell, void* stream);

  static void GPU_coarse_encode_phase3_sync(
      phf::array<M> d_par_nbit, phf::array<M> d_par_ncell,
      phf::array<M> d_par_entry,  //
      phf::par_config hfpar,      //
      phf::array<M> h_par_nbit, phf::array<M> h_par_ncell,
      phf::array<M> h_par_entry,                  //
      size_t* outlen_nbit, size_t* outlen_ncell,  //
      float* time_cpu_time, void* stream);

  static void GPU_coarse_encode_phase4(
      phf::array<H> buf, phf::array<M> par_entry, phf::array<M> par_ncell, phf::par_config hfpar,
      phf::array<H> bitstream, void* stream);

  static void GPU_coarse_decode(
      H* in_bitstream, uint8_t* in_revbook, size_t const revbook_len, M* in_par_nbit,
      M* in_par_entry, size_t const sublen, size_t const pardeg, E* out_decoded, void* stream);

  static void GPU_experimental_scatter(phf::sparse<E> compact, E* out, void* stream);
};

}  // namespace phf::cuhip

#endif /* ED744237_D186_4707_B7FC_1B931FFC5CBB */
