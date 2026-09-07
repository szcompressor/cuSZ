// [[not a public header]]

#ifndef HFD26_HH
#define HFD26_HH

#include <cstddef>
#include <cstdint>

using u1 = uint8_t;
using u2 = uint16_t;
using u4 = uint32_t;

namespace phf {

struct LutEntry {  // 8-bit LUT entry like Yamamoto et al.
  u2 symbol;
  u1 length;  // ==0 if codeword exceeds 8 bits
  u1 pad;
};
static_assert(sizeof(LutEntry) == 4, "LutEntry must be 4 bytes");

template <int Mag, int EinBytes>
struct hfd26_geometry {
  static constexpr auto ChunkSize = 1 << Mag;

  // offsets: one u2 per SymsPerShard
  static constexpr auto SymsPerShard = 16;  // lane-wise
  static constexpr auto ShardsPerChunk = ChunkSize / SymsPerShard;

  static constexpr auto StagedBytes = 8192;
  static constexpr auto StagedSymbols = StagedBytes / EinBytes;
  static constexpr auto _chunks = StagedSymbols / ChunkSize;

  static constexpr auto ChunksPerBlock = _chunks > 0 ? _chunks : 1;
  static constexpr auto BlockDim = ChunksPerBlock * ShardsPerChunk;
  static constexpr auto LB_NBlk = 1024 / BlockDim;
  static constexpr auto BsStageWords = 128 * (ChunkSize / 1024);  // block-wide

  static_assert(ChunkSize / 1024 >= 1, "1Ki, 2Ki, 4Ki (tested by far), ...");
  static_assert(BlockDim <= 1024, "must fit lowest HW-allowed block size");
  static_assert(ShardsPerChunk % 32 == 0, "must align to whole warps");
};

namespace cpu_ref {

template <typename H, typename KStorage>
void build_lut(u1 const* rvbk, LutEntry* out_lut);

template <typename E, typename H, typename KStorage>
void shard_inflate_lut(
    H const* bs_base, int bit_start, int bit_end, LutEntry const* lut, u1 const* rvbk, E* out,
    int max_out);

template <typename H>
int walk_n(H const* bs, H const* first, int i, int bit_end, int count);

}  // namespace cpu_ref

}  // namespace phf

namespace phf::module {

template <typename E, typename H = u4, typename Storage = u1, int Magnitude = 10>
struct HFD26 {
  // during init
  static int build_lut(
      u1 const* rvbks_g, int rvbk_nbyte, int num_books, phf::LutEntry* lut_d, void* stream);

  // decoding
  template <typename Eout = E>
  static int decode_fused(
      H* in_bitstream, size_t bitstream_len, u1* in_RVBKs, int RVBK_nbyte,
      u4 const* packed_headers, phf::LutEntry const* lut, int pardeg, size_t data_len,
      Eout* out_decoded, u1* out_incomp_flag, void* stream);
};

}  // namespace phf::module

#endif  // HFD26_HH
