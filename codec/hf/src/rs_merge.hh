#ifndef AF64552A_247F_47E8_BFFA_EFC88E0491EF
#define AF64552A_247F_47E8_BFFA_EFC88E0491EF

#include <stdexcept>

#include "hf_impl.hh"

using u4 = uint32_t;

// ---------------------------------------------------------------------------
// Compile-time configuration for one HFReVISIT instantiation.
// ---------------------------------------------------------------------------
template <typename _T, int _Magnitude, int _ReduceTimes, typename _Hf = u4>
struct HFReVISIT_config {
  using T = _T;
  using Hf = _Hf;
  using W = HuffmanWord<sizeof(Hf)>;

  static constexpr u4 Magnitude = _Magnitude;
  static constexpr u4 ReduceTimes = _ReduceTimes;
  static constexpr u4 ShuffleTimes = Magnitude - ReduceTimes;
  static constexpr u4 ChunkSize = 1u << Magnitude;    // input elements per block
  static constexpr u4 ShardSize = 1u << ReduceTimes;  // elements per thread
  static constexpr u4 NumShards = 1u << ShuffleTimes;
  static constexpr u4 BlockDim = NumShards;  // == threads per block
  static constexpr u4 BITWIDTH = sizeof(Hf) * 8;
};

namespace phf {

// Alt. code holds ShardSize (= 2^ReduceTimes) copies of the shortest one.
// TODO: need revision per EIP reference to decrease the overhead
template <typename Hf = u4>
void make_altcode(Hf* bk, u2 bklen, int reduce_times, Hf& alt_code, u4& alt_bitcount)
{
  using W = HuffmanWord<sizeof(Hf)>;

  const auto shard_size = 1 << reduce_times;

  // Find the shortest available codeword in the runtime codebook.
  Hf shortest = bk[0];
  auto shortest_w = reinterpret_cast<W*>(&shortest);
  for (u2 i = 1; i < bklen; i++) {
    auto cand = bk[i];
    auto cand_w = reinterpret_cast<W*>(&cand);
    if (cand_w->bitcount < shortest_w->bitcount) {
      shortest = cand;
      shortest_w = reinterpret_cast<W*>(&shortest);
    }
  }

  if (shortest_w->bitcount * shard_size > (sizeof(Hf) * 8))
    throw std::runtime_error("make_altcode: alt-code would exceed bitwidth.");

  alt_bitcount = 0;
  alt_code = 0;
  for (auto i = 0; i < shard_size; i++) {
    auto offset = W::BITWIDTH - (i + 1) * shortest_w->bitcount;
    alt_code |= (shortest_w->prefix_code << offset);
    alt_bitcount += shortest_w->bitcount;
  }
}

}  // namespace phf

namespace phf::module {

// ---------------------------------------------------------------------------
// HFReVISIT encoder module.
//
// GPU_kernel:
//   in/len        — quantization codes (host pointer to device memory)
//   bk/bklen      — dynamic Huffman codebook on device
//   alt_code/alt_bitcount — replacement codeword for breaking shards
//   dn_out        — dense bitstream output (device, size ≥ ChunkSize * nblocks)
//   dn_bitcount   — per-block bit counts (device, size = nblocks)
//   sp_val/sp_idx — sparse buffer for spilled (breaking) symbols
//   sp_count      — global counter for sparse buffer entries (device, size = 1)
//   stream        — CUDA/HIP stream
// ---------------------------------------------------------------------------
template <typename T, int Magnitude, int ReduceTimes, bool UseScan = false, typename Hf = u4>
struct HFReVISIT_encode {
  static int GPU_kernel(
      T* in, size_t len, Hf* bk, u4 bklen, Hf alt_code, u4 alt_bitcount, Hf* dn_out,
      u4* dn_bitcount, T* sp_val, u4* sp_idx, u4* sp_count, void* stream);
};

}  // namespace phf::module

#endif /* AF64552A_247F_47E8_BFFA_EFC88E0491EF */
