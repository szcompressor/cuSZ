// HFR-PBKC CPU reference decoder.

#include <cstddef>
#include <cstdint>

#include "hfr.hh"
#include "single_inflate.inl"

namespace phf::module {

template <typename T, int Magnitude, int ReduceTimes, typename Hf, uint16_t Radius>
void HFR_PBKC_decode<T, Magnitude, ReduceTimes, Hf, Radius>::CPU_kernel(
    Hf const* in_bitstream, break_t const* in_breaks, header_t const* in_headers, size_t nblock,
    uint8_t const* revbooks, size_t one_revbook_nbyte, T* out_eq, size_t out_len,
    int* opt_n_incomp_blocks)
{
  using BreakCell = psz::HFR_PBK_Breaks<Radius>;
  constexpr size_t ChunkSize = 1u << Magnitude;
  constexpr uint32_t NumBooks = 25;
  (void)in_breaks;

  int n_incomp = 0;

  for (size_t b = 0; b < nblock; b++) {
    auto h = in_headers[b];
    auto block_out = out_eq + b * ChunkSize;
    auto block_slot = in_bitstream + (h.entry / sizeof(Hf));
    auto br_slot = (BreakCell const*)block_slot;
    auto bs_slot = block_slot + h.n_breaks;

    const auto block_len =
        (size_t)((b + 1) * ChunkSize <= out_len ? ChunkSize : out_len - b * ChunkSize);

    if ((uint32_t)h.enc_id == (uint32_t)psz::HFR_PBK_Constants::CodeIncompBreaks) {
      auto raw = (T const*)bs_slot;
      for (size_t i = 0; i < block_len; i++) block_out[i] = raw[i];
      ++n_incomp;
      continue;
    }

    if ((uint32_t)h.enc_id >= NumBooks) continue;

    auto rvbk = const_cast<uint8_t*>(revbooks + (size_t)h.enc_id * one_revbook_nbyte);

    T scratch[ChunkSize] = {};
    phf::single_thread_inflate<T, Hf, uint8_t>(
        const_cast<Hf*>(bs_slot), scratch, rvbk, (int)h.bits);
    for (size_t i = 0; i < block_len; i++) block_out[i] = scratch[i];

    for (uint32_t k = 0; k < h.n_breaks; k++) {
      auto cell = br_slot[k];
      block_out[cell.idx] = (T)cell.val;
    }
  }

  if (opt_n_incomp_blocks) *opt_n_incomp_blocks = n_incomp;
}

}  // namespace phf::module

// instantiations
template struct phf::module::HFR_PBKC_decode<uint8_t, 10, 2, uint32_t, 128>;
template struct phf::module::HFR_PBKC_decode<uint16_t, 10, 2, uint32_t, 128>;
