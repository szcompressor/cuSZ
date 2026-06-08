#ifndef AF64552A_247F_47E8_BFFA_EFC88E0491EF
#define AF64552A_247F_47E8_BFFA_EFC88E0491EF

// HFR family interface declarations (encode/decode entry points + LAGO-concat helpers).

#include <cstddef>
#include <cstdint>

#include "hfr-pbk.hh"
#include "hf_impl.hh"

using u4 = uint32_t;

template <typename _T, int _Magnitude, int _ReduceTimes, typename _Hf = u4>
struct HFR_Config {
  using T = _T;
  using Hf = _Hf;
  using W = HuffmanWord<sizeof(Hf)>;

  static constexpr u4 Magnitude = _Magnitude;
  static constexpr u4 ReduceTimes = _ReduceTimes;
  static constexpr u4 ShuffleTimes = Magnitude - ReduceTimes;
  static constexpr u4 ChunkSize = 1u << Magnitude;
  static constexpr u4 ShardSize = 1u << ReduceTimes;
  static constexpr u4 NumShards = 1u << ShuffleTimes;
  static constexpr u4 BlockDim = NumShards;
  static constexpr u4 BITWIDTH = sizeof(Hf) * 8;
};

template <
    typename _T, int _Magnitude, int _ReduceTimes, typename _Hf = uint32_t, uint16_t _Radius = 128>
struct _HFR_PBK_Config_Base : HFR_Config<_T, _Magnitude, _ReduceTimes, _Hf> {
  static constexpr uint16_t Radius = _Radius;
  static constexpr uint16_t BookLen = _Radius * 2;
  static constexpr uint8_t NumBooks = 25;
};

namespace phf {

template <typename Hf = u4>
void make_altcode_single(Hf* bk, u2 bklen, int reduce_times, Hf& alt_code, u4& alt_bitcount)
{
  using W = HuffmanWord<sizeof(Hf)>;

  const auto shard_size = 1 << reduce_times;

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

  if (shortest_w->bitcount * shard_size > (sizeof(Hf) * 8)) {
    alt_code = 0;
    alt_bitcount = 0;
    return;
  }

  alt_code = (Hf)shortest_w->prefix_code;
  alt_bitcount = shortest_w->bitcount;
}

// Max RT supported by this codebook (avoids merge breaks).
template <typename Hf = u4>
int compute_rt_max_single(Hf const* bk, u2 bklen)
{
  using W = HuffmanWord<sizeof(Hf)>;
  if (bklen == 0) return 0;

  Hf shortest = bk[0];
  auto shortest_w = reinterpret_cast<W*>(&shortest);
  for (u2 i = 1; i < bklen; i++) {
    auto cand = bk[i];
    auto cand_w = reinterpret_cast<W*>(&cand);
    if (cand_w->bitcount and cand_w->bitcount < shortest_w->bitcount) {
      shortest = cand;
      shortest_w = reinterpret_cast<W*>(&shortest);
    }
  }
  if (shortest_w->bitcount == 0) return 0;

  const u4 BW = sizeof(Hf) * 8;
  const u4 cap = BW / shortest_w->bitcount;
  int rt = 0;
  while ((1u << (rt + 1)) <= cap) rt++;
  return rt;
}

// alt_code + bitcount + RT clamped by compressibility.
template <typename Hf = u4>
void make_altcode_and_rt(
    Hf const* bk, u2 bklen, int rt_ceil, Hf& alt_code, u4& alt_bitcount, int& effective_rt)
{
  const int rt_max = compute_rt_max_single<Hf>(bk, bklen);
  effective_rt = rt_max < rt_ceil ? rt_max : rt_ceil;

  using W = HuffmanWord<sizeof(Hf)>;
  Hf shortest = bk[0];
  auto shortest_w = reinterpret_cast<W*>(&shortest);
  for (u2 i = 1; i < bklen; i++) {
    auto cand = bk[i];
    auto cand_w = reinterpret_cast<W*>(&cand);
    if (cand_w->bitcount and cand_w->bitcount < shortest_w->bitcount) {
      shortest = cand;
      shortest_w = reinterpret_cast<W*>(&shortest);
    }
  }
  alt_code = (Hf)shortest_w->prefix_code;
  alt_bitcount = shortest_w->bitcount;
}

template <
    typename T, int Magnitude, int ReduceTimes, typename Hf = uint32_t, uint16_t Radius = 128>
using HFR_PBKC_Config = _HFR_PBK_Config_Base<T, Magnitude, ReduceTimes, Hf, Radius>;

template <
    typename T, int Magnitude, int ReduceTimes, typename Hf = uint32_t, uint16_t Radius = 128>
using HFR_PBKGO_Config = _HFR_PBK_Config_Base<T, Magnitude, ReduceTimes, Hf, Radius>;

}  // namespace phf

namespace phf::module {

template <typename T, int Magnitude, int ReduceTimes, bool UseScan = false, typename Hf = u4>
struct HFR_encoder {
  static int GPU_kernel_v1(
      T* in, size_t len, Hf* bk, u4 bklen, Hf alt_code, u4 alt_bitcount, Hf* dn_bitstream,
      u4* dn_bitcount, psz::HFR_PBK_Breaks<128>* sp_breaks, u4* sp_count, u4* par_brnum,
      u4* par_broffset, uint8_t* par_encid, void* stream);

  template <uint16_t Radius = 128>
  static int GPU_kernel_v2(
      T* in_eq, size_t len, Hf* runtime_book, Hf* dn_bitstream,
      psz::_future::bheader<T, Radius>* dn_headers, void* stream, RMerge rm, SMerge sm);
};

struct pack_bheader_backport {
  static int GPU_kernel(
      uint32_t const* par_nbit, uint32_t const* par_entry, uint32_t* out_headers, int pardeg,
      int sizeof_Hf, void* stream);
};

struct unpack_bheader_backport {
  static int GPU_kernel(
      uint32_t const* in_headers, uint32_t* par_nbit, uint32_t* par_entry, int pardeg,
      int sizeof_Hf, void* stream);
};

struct reduce_total_nbit {
  static int GPU_kernel(u4 const* par_nbit, u4 pardeg, u4* total_nbit, void* stream);
};

template <typename E>
struct pack_pbk_metadata {
  using bheader_t = psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius>;
  static int GPU_kernel(
      bheader_t const* pbk_headers, u4 pardeg, u4* par_nbit, u4* par_ncell, uint8_t* par_encid,
      void* stream);
};

// Pack bheader[] -> 2-word AoS headers (for paths that bypass LAGO concat).
template <typename E>
struct pack_packed_headers {
  using bheader_t = psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius>;
  static int GPU_kernel(
      bheader_t const* pbk_headers, uint32_t* out_headers, uint32_t sizeof_Hf, int pardeg,
      void* stream);
};

}  // namespace phf::module

namespace phf {

template <int BlockDim>
struct concat_via_scatter_ppc {
  static int GPU_kernel(
      uint32_t const* par_ncell, uint32_t* par_entry, uint32_t const* dn_in, uint32_t* dn_out,
      uint32_t ChunkSize, int pardeg, uint32_t* scan_partial_aggregate, uint32_t* scan_incl_prefix,
      int* scan_tile_status, uint32_t* opt_d_total_words, void* stream);
};

template <typename E, int BlockDim>
struct _future_concat_via_scatter {
  using bheader_t = psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius>;
  static int GPU_kernel(
      bheader_t const* bheaders, uint32_t* par_entry, uint32_t const* dn_in, uint32_t* dn_out,
      uint32_t* out_packed_headers, uint32_t sizeof_Hf, uint32_t ChunkSize, int pardeg,
      uint32_t* scan_partial_aggregate, uint32_t* scan_incl_prefix, int* scan_tile_status,
      uint32_t* opt_d_total_words, void* stream);
};

}  // namespace phf

namespace phf::module {

template <
    typename T, int Magnitude, int ReduceTimes, typename Hf = uint32_t, uint16_t Radius = 128>
struct HFR_PBKC_encode {
  using header_t = psz::_future::bheader<T, Radius>;
  using break_t = psz::HFR_PBK_Breaks<Radius>;

  static int GPU_kernel(
      T* in_eq, size_t len, Hf* dram_pbk, Hf* dn_bitstream, header_t* dn_headers, void* stream,
      RMerge rm, SMerge sm);
};

template <
    typename T, int Magnitude = 10, int ReduceTimes = 2, typename Hf = uint32_t,
    uint16_t Radius = 128>
struct HFR_PBKC_decode {
  using header_t = psz::_future::bheader<T, Radius>;
  using break_t = psz::HFR_PBK_Breaks<Radius>;

  static void CPU_kernel(
      Hf const* in_bitstream, break_t const* in_breaks, header_t const* in_headers, size_t nblock,
      uint8_t const* revbooks, size_t one_revbook_nbyte, T* out_eq, size_t out_len,
      int* opt_n_incomp_blocks = nullptr);
};

template <
    typename T, int Magnitude, int ReduceTimes, typename Hf = uint32_t, uint16_t Radius = 128>
struct HFR_PBKGO_encode {
  using header_t = psz::_future::bheader<T, Radius>;
  using break_t = psz::HFR_PBK_Breaks<Radius>;

  static int max_blocks_per_sm();

  // Needs a caller-owned, pre-zeroed d_state buffer (pardeg u4 words).
  static int GPU_kernel(
      T* in_eq, size_t len, Hf* dram_pbk, Hf* dn_bitstream, uint32_t* dn_packed_headers,
      uint32_t* d_total_cells, uint32_t* d_state, int max_resident_blocks, void* stream, RMerge rm,
      SMerge sm);
};

template <
    typename T, int Magnitude = 10, int ReduceTimes = 2, typename Hf = uint32_t,
    uint16_t Radius = 128>
struct HFR_PBKGO_decode {
  using header_t = psz::_future::bheader<T, Radius>;
  using break_t = psz::HFR_PBK_Breaks<Radius>;

  static void CPU_kernel(
      Hf const* in_bitstream, break_t const* in_breaks, header_t const* in_headers, size_t nblock,
      uint8_t const* revbooks, size_t one_revbook_nbyte, T* out_eq, size_t out_len,
      int* opt_n_incomp_blocks = nullptr);
};

}  // namespace phf::module

#endif /* AF64552A_247F_47E8_BFFA_EFC88E0491EF */
