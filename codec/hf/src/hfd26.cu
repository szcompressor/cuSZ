#include <cstdint>
#include <type_traits>

#include "_future/block_scan.cuh"
#include "hfd26.hh"
#include "hfr-pbk.hh"
#include "mem/cxx_v.cuh"

namespace phf {

using u1 = u1;
using u2 = u2;
using u4 = u4;

using psz::unpack_par_dense;
using psz::unpack_par_encid;
using psz::unpack_par_end_words;
using psz::unpack_par_entry_words;
using psz::unpack_par_nunpred;

using _ptb::_v;

// u2 shard offsets: log2_ceil(4Ki) (hfd26_geometry::BsStageWords)
static_assert(
    psz::log2_ceil(hfd26_geometry<12, 4>::ChunkSize * 4) <= 16,
    "shard offsets (u2) can't cover the largest chunk at 4 bits/symbol");

// build the per-book 8-bit primary LUT
template <typename H, typename Storage>
__global__ void KCU_hfd26_build_lut(
    u1 const* rvbks_g, int rvbk_nbyte, int num_books, LutEntry* out_lut)
{
  const auto book = blockIdx.x;
  const auto p = threadIdx.x;
  if (book >= num_books or p >= 256) return;
  constexpr auto H_TYPE_BITS = sizeof(H) * 8;
  u1 const* rvbk = rvbks_g + book * rvbk_nbyte;
  auto first = reinterpret_cast<H const*>(rvbk);
  auto entry = first + H_TYPE_BITS;
  auto keys = reinterpret_cast<Storage const*>(rvbk + sizeof(H) * (2 * H_TYPE_BITS));
  LutEntry e = {0, 0, 0};
#pragma unroll
  for (auto L = 1; L <= 8; ++L) {
    H v = (H)((u4)p >> (8 - L));
    H first_L = first[L];
    H count_L = entry[L + 1] - entry[L];
    if (v >= first_L && v < first_L + count_L) {
      e.symbol = (u2)keys[entry[L] + v - first_L];
      e.length = (u1)L;
      break;
    }
  }
  out_lut[(size_t)book * 256 + p] = e;
}

template <typename E, typename H, typename Storage>
__device__ __forceinline__ void shard_inflate_lut(
    H const* input, int bit_start, int bit_end, LutEntry const* lut_chunk, u1 const* rvbk, E* out,
    int max_out)
{
  constexpr auto H_TYPE_BITS = sizeof(H) * 8;
  if (bit_end <= bit_start) return;
  auto first = reinterpret_cast<H const*>(rvbk);
  auto entry = first + H_TYPE_BITS;
  auto keys = reinterpret_cast<Storage const*>(rvbk + sizeof(H) * (2 * H_TYPE_BITS));
  auto i = bit_start, idx_out = 0;
  while (i < bit_end and idx_out < max_out) {
    if (bit_end - i >= 8) {
      const auto word_idx = i >> 5;
      const auto word_bit = i & 31;
      H w0 = input[word_idx];
      H w1 = (word_bit > 24) ? input[word_idx + 1] : (H)0;
      H combined = (w0 << word_bit) | ((word_bit == 0) ? (H)0 : (w1 >> (H_TYPE_BITS - word_bit)));
      auto peek = (u4)((combined >> 24) & 0xFFu);
      LutEntry e = lut_chunk[peek];
      if (e.length > 0) {
        out[idx_out++] = (E)e.symbol;
        i += e.length;
        continue;
      }
      H v = (H)peek;
      auto L = 8;
      while (v < first[L]) {  // terminiate on L=H_TYPE_BITS
        ++L;
        if (L > H_TYPE_BITS) break;  // not for valid books
        const auto bit_idx = i + L - 1;
        const auto wi = bit_idx >> 5;
        const auto bi = bit_idx & 31;
        H next_bit = (input[wi] >> (H_TYPE_BITS - 1 - bi)) & (H)0x1u;
        v = (H)((v << 1) | next_bit);
      }
      out[idx_out++] = (E)keys[entry[L] + v - first[L]];
      i += L;
    }
    else {
      auto idx_byte = i >> 5;
      auto idx_bit = i & 31;
      H bufr = input[idx_byte];
      H v = (bufr >> (H_TYPE_BITS - 1 - idx_bit)) & (H)0x1u;
      auto l = 1;
      while (v < first[l]) {
        ++i;
        idx_byte = i >> 5;
        idx_bit = i & 31;
        if (idx_bit == 0) bufr = input[idx_byte];
        H next_bit = (bufr >> (H_TYPE_BITS - 1 - idx_bit)) & (H)0x1u;
        v = (H)((v << 1) | next_bit);
        ++l;
      }
      out[idx_out++] = (E)keys[entry[l] + v - first[l]];
      ++i;
    }
  }
}

// granularity: one codeword: LUT or bit-by-bit
template <typename H>
__device__ __forceinline__ int hfd26_lut_next(
    H const* bs, H const* first, LutEntry const* lut, int i, int const bit_end)
{
  constexpr auto H_TYPE_BITS = sizeof(H) * 8;
  if (bit_end - i >= 8) {
    const auto word_idx = i >> 5;
    const auto word_bit = i & 31;
    H const w0 = bs[word_idx];
    H const w1 = (word_bit > 24) ? bs[word_idx + 1] : (H)0;
    H const combined =
        (w0 << word_bit) | ((word_bit == 0) ? (H)0 : (w1 >> (H_TYPE_BITS - word_bit)));
    const auto peek = (u4)((combined >> 24) & 0xFFu);
    LutEntry const e = lut[peek];
    if (e.length > 0) return i + e.length;
    H v = (H)peek;
    auto L = 8;
    while (v < first[L]) {
      ++L;
      if (L > H_TYPE_BITS) return i + H_TYPE_BITS;  // abnormal; valid books never get here
      const auto bit_idx = i + L - 1;
      H const next = (bs[bit_idx >> 5] >> (H_TYPE_BITS - 1 - (bit_idx & 31))) & (H)0x1u;
      v = (H)((v << 1) | next);
    }
    return i + L;
  }
  auto idx_word = i >> 5, idx_bit = i & 31;
  H bufr = bs[idx_word];
  H v = (bufr >> (H_TYPE_BITS - 1 - idx_bit)) & (H)0x1u;
  auto l = 1;
  while (v < first[l]) {
    ++i;
    idx_word = i >> 5;
    idx_bit = i & 31;
    if (idx_bit == 0) bufr = bs[idx_word];
    H next = (bufr >> (H_TYPE_BITS - 1 - idx_bit)) & (H)0x1u;
    v = (H)((v << 1) | next);
    ++l;
    if (l > H_TYPE_BITS) return i;  // abnormal, stop
  }
  return i + 1;
}

// decode, possibly cross-codeword; count returned via `n_syms`
template <typename H>
__device__ __forceinline__ int hfd26_decode_until(
    H const* bs, H const* first, LutEntry const* lut, int start, int limit, int bit_end,
    int& n_syms)
{
  auto i = start, n = 0;
  while (i < limit and i < bit_end) {
    i = hfd26_lut_next(bs, first, lut, i, bit_end);
    ++n;
  }
  n_syms = n;
  return i;
}

// self-sync-like
template <typename H, int NumSegs>
__device__ __forceinline__ void hfd26_staircase_sync(
    H const* bs, H const* first, LutEntry const* lut, int const seg_bits, int const bit_end,
    int const l_id, const bool is_worker, u2* s_end, u2* s_syms, int* s_unsynced)
{
  auto cur_i = 0, n_syms = 0;
  bool synced = false;

  if (is_worker) {
    cur_i = hfd26_decode_until(
        bs, first, lut, l_id * seg_bits, min((l_id + 1) * seg_bits, bit_end), bit_end, n_syms);
    s_end[l_id] = (u2)cur_i;
    s_syms[l_id] = (u2)n_syms;
  }
  if (l_id == 0) *s_unsynced = NumSegs;  // one per worker; workers subtract on sync
  __syncthreads();

  for (auto t = 1; t < NumSegs; ++t) {
    if (is_worker) {
      const auto cs = l_id + t;
      if (cs < NumSegs and not synced) {
        const auto limit = min((cs + 1) * seg_bits, bit_end);
        const auto end_bit = hfd26_decode_until(bs, first, lut, cur_i, limit, bit_end, n_syms);
        if ((u2)end_bit == s_end[cs]) {
          synced = true;
          atomicSub(s_unsynced, 1);
        }
        else {
          s_end[cs] = (u2)end_bit;
          s_syms[cs] = (u2)n_syms;
        }
        cur_i = end_bit;
      }
    }
    __syncthreads();
    if (*s_unsynced == 0) break;
  }
}

// search s_prefix for the segment owning it
// template <typename H, int NumSegs>
// __device__ __forceinline__ int hfd26_shard_offset(
//     H const* bs, H const* first, LutEntry const* lut, int const* s_prefix, u2 const* s_end,
//     int const target, int const bit_end)
// {
//   auto lo = 0, hi = NumSegs - 1, s = 0;
//   while (lo <= hi) {
//     const auto mid = (lo + hi) / 2;
//     if (s_prefix[mid] <= target) {
//       s = mid;
//       lo = mid + 1;
//     }
//     else
//       hi = mid - 1;
//   }
//   auto pos = (s == 0) ? 0 : s_end[s - 1];
//   for (auto r = target - s_prefix[s]; r > 0 and pos < bit_end; --r)
//     pos = hfd26_lut_next(bs, first, lut, pos, bit_end);
//   return pos;
// }

template <typename Ein, typename Eout, int Stride>
__device__ __forceinline__ void hfd26_bypass_copy(
    Ein const* raw, Eout* out, size_t const valid, const bool is_incompunpred, int const l_id)
{
  if constexpr (sizeof(Ein) == 4 and sizeof(Eout) == 4) {
    if (((uintptr_t)raw % 16u) == 0u and ((uintptr_t)out % 16u) == 0u and (valid % 4u) == 0u) {
      using VIn = _v<Ein, 4>;
      using VOut = _v<Eout, 4>;
      const auto* raw_v = reinterpret_cast<VIn const*>(raw);
      auto* out_v = reinterpret_cast<VOut*>(out);
      if (is_incompunpred)
        for (u4 i = l_id; i < valid / 4u; i += Stride) {
          VIn const r = raw_v[i];
          VOut w;
#pragma unroll
          for (auto j = 0; j < 4; ++j) w[j] = (Eout)psz::incomp_unpack<Ein>(r[j]);
          out_v[i] = w;
        }
      else
        for (u4 i = l_id; i < valid / 4u; i += Stride) {
          VIn const r = raw_v[i];
          VOut w;
#pragma unroll
          for (auto j = 0; j < 4; ++j) w[j] = (Eout)r[j];
          out_v[i] = w;
        }
      return;
    }
  }
  if (is_incompunpred)
    for (u4 i = l_id; i < valid; i += Stride) out[i] = (Eout)psz::incomp_unpack<Ein>(raw[i]);
  else
    for (u4 i = l_id; i < valid; i += Stride) out[i] = (Eout)raw[i];
}

template <typename Ein, typename H, typename Storage, typename Eout, int Mag>
__global__
__launch_bounds__(hfd26_geometry<Mag, sizeof(Ein)>::ShardsPerChunk) void KCU_hfd26_fused(
    H* in_bs, size_t const bs_len, u1* in_rvbks, int const rvbk_nbyte, u4 const* bheaders,
    LutEntry const* lut, int const pardeg, size_t const data_len, Eout* out_decoded,
    u1* out_incomp_flag)
{
  using psz::OutlierCell;
  using G = hfd26_geometry<Mag, sizeof(Ein)>;
  using C = psz::HFR_PBK_Constants;
  using BreakCell = psz::HFR_PBK_Breaks<C::Radius>;

  constexpr auto ChunkSize = G::ChunkSize;
  constexpr auto ShardsPerChunk = G::ShardsPerChunk;
  // constexpr auto SymsPerShard = G::SymsPerShard;
  constexpr auto BsStageWords = G::BsStageWords;
  constexpr auto NumBooks = C::NumBooks;
  constexpr auto LutEntries = 256;

  constexpr auto NumSegs = (Mag == 10) ? ShardsPerChunk : ShardsPerChunk / 2;
  constexpr auto NumWarps = NumSegs / 32;
  static_assert(NumSegs <= 1024, "one chunk per block; NumSegs must fit a CUDA block");
  static_assert(NumSegs % 32 == 0, "NumSegs must be a whole number of warps");
  using SymStage = std::conditional_t<(sizeof(Ein) > 2), u2, Ein>;

  // shard: 16 symbols = fixed-size output
  // segment: seg_bits bits = fixed-size input
  // symbols   0    16    32    48    64    80
  // shards    |--0--|--1--|--2--|--3--|--4--|
  // segments  |---- 0 ----|------ 1 ------|--- 2 ---
  //                  ^                 ^
  //             shard 1 crosses     shard 3 crosses
  //             segs 0 and 1        segs 1 and 2
  // constexpr bool SegInflate = true;
  constexpr bool AllWorkers = (NumSegs == ShardsPerChunk);

  __shared__ SymStage s_decoded[ChunkSize];
  __shared__ H s_bs_stage[BsStageWords];
  __shared__ __align__(16) LutEntry s_lut[LutEntries];
  __shared__ u2 s_end[NumSegs];
  __shared__ u2 s_syms[NumSegs];
  __shared__ int s_warp_totals[NumWarps];
  __shared__ int s_unsynced;

  const auto gid = blockIdx.x;
  const auto l_id = threadIdx.x;

  const bool is_worker = AllWorkers or l_id < NumSegs;
  const bool is_oob = (gid >= pardeg);

  const auto w0 = is_oob ? 0u : bheaders[2 * gid + 0];
  const auto bk_id = unpack_par_encid<Mag>(w0);

  const bool is_incompunpred = not is_oob and (bk_id == C::CodeIncompUnpred);
  const bool is_bypass = is_oob or (bk_id >= (u4)NumBooks);
  if (out_incomp_flag and not is_oob and l_id == 0)
    out_incomp_flag[gid] = is_incompunpred ? 1u : 0u;

  const auto dn_words = is_oob ? 0u : unpack_par_dense<Mag>(w0);
  const auto nunpred = is_oob ? 0u : unpack_par_nunpred<Mag>(w0);

  const auto _blk_start = is_oob ? 0u : unpack_par_entry_words<H>(bheaders, gid);
  const auto _blk_end =
      is_oob ? _blk_start : unpack_par_end_words<H>(bheaders, gid, pardeg, bs_len);
  const auto _total_units = _blk_end - _blk_start;
  const auto nbreaks = is_bypass ? 0u : _total_units - dn_words - psz::pbk_unpred_words(nunpred);
  const auto _blk = in_bs + _blk_start;
  const auto breaks = (BreakCell const*)_blk;
  const auto blk_bs = _blk + nbreaks;
  const auto unpred = (OutlierCell const*)(is_bypass ? (_blk + dn_words) : (blk_bs + dn_words));
  const auto sym_start = (size_t)gid * ChunkSize;
  const auto valid =
      is_oob ? 0u : (data_len - sym_start < ChunkSize ? data_len - sym_start : ChunkSize);
  Eout* out_blk = out_decoded + sym_start;
  SymStage* out_shmem = s_decoded;

  // `is_bypass` indiacates two paths of different barrier counts.
  if (is_bypass) {
    if (not is_oob)
      hfd26_bypass_copy<Ein, Eout, ShardsPerChunk>(
          reinterpret_cast<Ein const*>(_blk), out_blk, valid, is_incompunpred, l_id);
  }
  else {
    u1 const* rvbk = in_rvbks + bk_id * (u4)rvbk_nbyte;

    {  // stage per-blk book
      using _lut_t = _v<u4, 4>;
      constexpr auto NV = LutEntries / 4;
      const auto* src_v = reinterpret_cast<_lut_t const*>(lut + bk_id * LutEntries);
      auto* lut_v = reinterpret_cast<_lut_t*>(s_lut);
      for (auto k = l_id; k < NV; k += ShardsPerChunk) lut_v[k] = src_v[k];
    }

    // stage dense bits
    const bool use_stage = dn_words <= BsStageWords;
    if (use_stage)
      for (u4 w = (u4)l_id; w < dn_words; w += (u4)ShardsPerChunk) s_bs_stage[w] = blk_bs[w];
    __syncthreads();

    H const* bs = use_stage ? (H const*)s_bs_stage : blk_bs;
    H const* first = reinterpret_cast<H const*>(rvbk);
    const auto bit_end = dn_words * 32u;

    const auto seg_bits = (bit_end + NumSegs - 1) / NumSegs;
    hfd26_staircase_sync<H, NumSegs>(
        bs, first, s_lut, seg_bits, bit_end, l_id, is_worker, s_end, s_syms, &s_unsynced);

    auto val = 0;
    const auto lane = l_id & 31, warp_id = l_id >> 5;
    {
      if (is_worker) {
        // s_syms is refreshed regardless
        const auto seg_start = (l_id == 0) ? 0 : s_end[l_id - 1];
        hfd26_decode_until(bs, first, s_lut, seg_start, s_end[l_id], bit_end, val);
        s_syms[l_id] = (u2)val;
        val = block_scan::warp_incl_scan_u32(val);
        if (lane == 31) s_warp_totals[warp_id] = val;
      }
      __syncthreads();
      if (is_worker and warp_id == 0) {
        int const warp_sum_incl =
            block_scan::warp_incl_scan_u32((lane < NumWarps) ? s_warp_totals[lane] : 0);
        if (lane < NumWarps) s_warp_totals[lane] = warp_sum_incl;
      }
      __syncthreads();
    }

    // if constexpr (SegInflate) {
    if (is_worker) {
      const auto warp_prefix = (warp_id == 0) ? 0 : s_warp_totals[warp_id - 1];
      const auto out_pos = val + warp_prefix - s_syms[l_id];
      const auto seg_start = (l_id == 0) ? 0 : s_end[l_id - 1];
      if (out_pos < ChunkSize)
        shard_inflate_lut<SymStage, H, Storage>(
            bs, seg_start, s_end[l_id], s_lut, rvbk, out_shmem + out_pos, ChunkSize - out_pos);
    }
    // }
    // else {  // old version: unused for now
    //   __shared__ int s_prefix[NumSegs];
    //   __shared__ u2 s_shard_ofst[ShardsPerChunk];

    //   if (is_worker) {
    //     const auto warp_prefix = (warp_id == 0) ? 0 : s_warp_totals[warp_id - 1];
    //     s_prefix[l_id] = val + warp_prefix - s_syms[l_id];
    //   }
    //   __syncthreads();

    //   // one shard per thread
    //   s_shard_ofst[l_id] =
    //       (l_id == 0) ? (u2)0
    //                   : (u2)hfd26_shard_offset<H, NumSegs>(
    //                         bs, first, s_lut, s_prefix, s_end, l_id * SymsPerShard, bit_end);
    //   __syncthreads();

    //   const auto bit_start = s_shard_ofst[l_id];
    //   const auto shard_bit_end = (l_id + 1 < ShardsPerChunk) ? s_shard_ofst[l_id + 1] : bit_end;
    //   shard_inflate_lut<SymStage, H, Storage>(
    //       bs, bit_start, shard_bit_end, s_lut, rvbk, out_shmem + l_id * SymsPerShard,
    //       SymsPerShard);
    // }
    __syncthreads();

    for (u4 k = l_id; k < nbreaks; k += ShardsPerChunk) {
      auto cell = breaks[k];
      if ((u4)cell.idx < valid) out_shmem[cell.idx] = (SymStage)cell.val;
    }
    __syncthreads();

    for (u4 i = l_id; i < valid; i += ShardsPerChunk) out_blk[i] = (Eout)out_shmem[i];
  }
  __syncthreads();

  if (not is_oob and not is_incompunpred)
    for (u4 k = l_id; k < nunpred; k += ShardsPerChunk) {
      auto cell = unpred[k];
      if (cell.idx < valid) out_blk[cell.idx] = (Eout)cell.val;
    }
}

}  // namespace phf

namespace phf::module {

template <typename E, typename H, typename Storage, int Mag>
template <typename Eout>
int HFD26<E, H, Storage, Mag>::decode_fused(
    H* in_bs, size_t bs_len, u1* in_RVBKs, int RVBK_nbyte, u4 const* bheaders,
    phf::LutEntry const* lut, int pardeg, size_t data_len, Eout* out_decoded, u1* out_incomp_flag,
    void* stream)
{
  if (pardeg <= 0) return 0;
  using G = phf::hfd26_geometry<Mag, sizeof(E)>;
  dim3 grid(pardeg, 1, 1);
  dim3 block(G::ShardsPerChunk, 1, 1);
  phf::KCU_hfd26_fused<E, H, Storage, Eout, Mag><<<grid, block, 0, (cudaStream_t)stream>>>(
      in_bs, bs_len, in_RVBKs, RVBK_nbyte, bheaders, lut, pardeg, data_len, out_decoded,
      out_incomp_flag);
  return 0;
}

// aux, one-time
template <typename E, typename H, typename Storage, int Mag>
int HFD26<E, H, Storage, Mag>::build_lut(
    u1 const* rvbks_g, int rvbk_nbyte, int num_books, phf::LutEntry* lut_d, void* stream)
{
  if (num_books <= 0) return 0;
  dim3 grid(num_books, 1, 1);
  dim3 block(256, 1, 1);
  phf::KCU_hfd26_build_lut<H, Storage>
      <<<grid, block, 0, (cudaStream_t)stream>>>(rvbks_g, rvbk_nbyte, num_books, lut_d);
  return 0;
}

// instantiation
#define HFD26_CELL(EIN, STORAGE, MAG)                           \
  template struct HFD26<EIN, u4, STORAGE, MAG>;                 \
  template int HFD26<EIN, u4, STORAGE, MAG>::decode_fused<EIN>( \
      u4*, size_t, u1*, int, u4 const*, phf::LutEntry const*, int, size_t, EIN*, u1*, void*);
#define HFD26_CELL_VALUED(EIN, STORAGE, MAG, EOUT)               \
  template int HFD26<EIN, u4, STORAGE, MAG>::decode_fused<EOUT>( \
      u4*, size_t, u1*, int, u4 const*, phf::LutEntry const*, int, size_t, EOUT*, u1*, void*);

#define HFD26_MAGNITUDE_SET(MAG)     \
  HFD26_CELL(u1, u1, MAG)            \
  HFD26_CELL(u2, u1, MAG)            \
  HFD26_CELL(u2, u2, MAG)            \
  HFD26_CELL(u4, u1, MAG)            \
  HFD26_CELL(u4, u4, MAG)            \
  HFD26_CELL_VALUED(u2, u1, MAG, f4) \
  HFD26_CELL_VALUED(u2, u1, MAG, f8) \
  HFD26_CELL_VALUED(u2, u2, MAG, f4) \
  HFD26_CELL_VALUED(u2, u2, MAG, f8) \
  HFD26_CELL_VALUED(u4, u1, MAG, f4) \
  HFD26_CELL_VALUED(u4, u1, MAG, f8) \
  HFD26_CELL_VALUED(u4, u4, MAG, f4) \
  HFD26_CELL_VALUED(u4, u4, MAG, f8)

HFD26_MAGNITUDE_SET(10)
HFD26_MAGNITUDE_SET(11)
HFD26_MAGNITUDE_SET(12)

#undef HFD26_MAGNITUDE_SET
#undef HFD26_CELL_VALUED
#undef HFD26_CELL

}  // namespace phf::module
