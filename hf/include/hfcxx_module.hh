#ifndef ED744237_D186_4707_B7FC_1B931FFC5CBB
#define ED744237_D186_4707_B7FC_1B931FFC5CBB

#include <cstdint>

#include "hfcxx_array.hh"

namespace _2403 {

/**
 * @brief a namespace-like class for batch template instantiations; a rewrite
 * of hfcodec.{hh,cc}; all the included wrapped kernels/methods are `static`
 *
 * @tparam E input type, e.g., error-quantization code in psz
 * @tparam H intermediate type for Huffman coding
 * @tparam M metadata, e.g., saved index for parallel operations
 * @tparam TIMING default true to replicate the original
 */
template <typename E, typename H, typename M = uint32_t, bool TIMING = true>
class phf_kernel_wrapper {
 public:
  static void phf_coarse_encode_phase1(
      hfarray_cxx<E> in, hfarray_cxx<H> book, const int numSMs,
      hfarray_cxx<H> out, float* time_lossless, void* stream);

  static void phf_coarse_encode_phase1_collect_metadata(
      hfarray_cxx<E> in, hfarray_cxx<H> book, const int numSMs,
      hfarray_cxx<H> out, hfarray_cxx<M> par_nbit, hfarray_cxx<M> par_ncell,
      hfpar_description hfpar, float* time_lossless, void* stream);

  static void phf_coarse_encode_phase2(
      hfarray_cxx<H> in, hfpar_description hfpar, hfarray_cxx<H> deflated,
      hfarray_cxx<M> par_nbit, hfarray_cxx<M> par_ncell, float* time_lossless,
      void* stream);

  static void phf_coarse_encode_phase3(
      hfarray_cxx<M> d_par_nbit, hfarray_cxx<M> d_par_ncell,
      hfarray_cxx<M> d_par_entry,  //
      hfpar_description hfpar,     //
      hfarray_cxx<M> h_par_nbit, hfarray_cxx<M> h_par_ncell,
      hfarray_cxx<M> h_par_entry,                 //
      size_t* outlen_nbit, size_t* outlen_ncell,  //
      float* time_cpu_time, void* stream);

  static void phf_coarse_encode_phase4(
      hfarray_cxx<H> buf, hfarray_cxx<M> par_entry, hfarray_cxx<M> par_ncell,
      hfpar_description hfpar, hfarray_cxx<H> bitstream, float* time_lossless,
      void* stream);

  static void phf_coarse_decode(
      hfarray_cxx<H> bitstream, hfarray_cxx<uint8_t> revbook,
      hfarray_cxx<M> par_nbit, hfarray_cxx<M> par_entry,
      hfpar_description hfpar, hfarray_cxx<E> out, float* time_lossless,
      void* stream);

  static void phf_scatter_adhoc(
      hfcompact_cxx<E> compact, E* out, float* milliseconds, void* stream);
};

}  // namespace _2403

#endif /* ED744237_D186_4707_B7FC_1B931FFC5CBB */
