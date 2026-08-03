#ifndef HFR_PBK_DECODER_HH
#define HFR_PBK_DECODER_HH

#include <cstddef>
#include <cstdint>

namespace phf::module {

// per-block GPU Huffman inflate (naive).
// u1 for PBK25_R128 pool, u2 for runtime-built rvbk on E=u2
template <typename E, typename H = uint32_t, typename Storage = uint8_t, int Magnitude = 10>
struct HFR_PBK_decoder {
  // pbk_packed_headers: 2 u4 per block (w0 packs nbit|encid, w1 = entry).
  template <typename Eout = E>
  static int GPU_kernel(
      H* in_pbk_bitstream, size_t pbk_bitstream_len, uint8_t* in_revbooks_r128_25,
      int revbook_nbyte, uint32_t const* pbk_packed_headers, int pbk_pardeg, size_t data_len,
      Eout* out_decoded, uint8_t* out_incomp_flag, void* stream);
};

}  // namespace phf::module

#endif  // HFR_PBK_DECODER_HH
