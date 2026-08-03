#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <type_traits>

#include "_future/block_scan.cuh"
#include "_future/warp_top1.cuh"
#include "c_type.h"
#include "hf_impl.hh"
#include "hfr-pbk.hh"

using Hf = u4;
using W = HuffmanWord<sizeof(Hf)>;
using Constants = psz::HFR_PBK_Constants;

// forward declarations
namespace hfr_pbk {

// slot policies: .reserve + .commit
struct slot_fixed_stride;
struct slot_lago_ticket;
struct slot_decoupled_lookback;

// clang-format off

// book selection
template <int TileDim, int PbkNumBooks, typename Header_v3> __forceinline__ __device__ void find_proper_book(volatile u4* s_top1_counts, volatile Header_v3* v3_bheader, size_t data_len, u4 chunk_id);
template <int PbkBookLen, typename Header_v3>               __forceinline__ __device__ void load_proper_book(u4& ReduceTimes, volatile u4* s_book, u4* pbk, volatile Header_v3* v3_bheader);

// reduce-merge
template <int ChunkSize, int BitWidth, int Radius, int BaseSeq, int MergeSize, typename Header_v3>    __forceinline__ __device__ void rmerge_sync__v7_const_shardsize_iter(size_t const data_len, u4 const chunk_id, volatile u4* s_book, int p_eq[BaseSeq], psz::HFR_PBK_Breaks<Radius>* br2_val_idx, volatile u4* s_v3_incomp, volatile Header_v3* v3_bheader, u4* r_reduced, u4* r_bits);
template <int ChunkSize, int BitWidth, int Radius, int BaseRT, typename Header_v3, int RT>            __forceinline__ __device__ void rmerge__v7_wrapper                  (u4 const reduce_times, size_t const data_len, u4 const chunk_id, volatile u4* s_book, int* p_eq, psz::HFR_PBK_Breaks<Radius>* br2_val_idx, volatile u4* s_v3_incomp, volatile Header_v3* v3_bheader, u4* r_reduced, u4* r_bits);

// shuffle-merge
template <int ChunkSize, int BitWidth, int BaseSeq, int MergeSize> __forceinline__ __device__ void smerge_sync__v7_const_shardsize_iter(volatile u4* s_reduced, volatile u4* s_bitcount, u4* r_reduced, u4* r_bits);
template <int ChunkSize, int BitWidth, int BaseRT, int RT>         __forceinline__ __device__ void smerge__v7_wrapper                  (u4 const reduce_times, volatile u4* s_reduced, volatile u4* s_bitcount, u4* r_reduced, u4* r_bits);

// version dispatch (state bundled in MergeCtx<C>)
template <class C> struct MergeCtx;
template <RMerge V, class C>        __forceinline__ __device__ void dispatch_rmerge(MergeCtx<C> cx);
template <SMerge V, class C>        __forceinline__ __device__ void dispatch_smerge(MergeCtx<C> cx);
template <class Launch>             __host__ void dispatch_merge_host(RMerge rm, SMerge sm, Launch&& launch);

// block/bitstream write
template <typename Slot, typename BreaksRouter, typename Hf, typename Header, typename BreakCell>       __forceinline__ __device__ void write_pbk_bitstream_v2(u4 b, volatile u4* s_bitcount, volatile Hf* s_reduced, u1* dn_base, volatile Header* bheader, volatile BreakCell* s_breaks, Slot slot, BreaksRouter router, psz::OutlierCell const* block_outliers = nullptr);

// incomp fallback
template <typename bheader_v3_t, typename T, int TileDim, int Seq, int NumThreads, size_t SymCodec> __device__ __forceinline__ void _handle_incomp_non_final                (volatile bheader_v3_t* s_v3_bheader, u1* pbk_bs_v3, size_t stride_bytes, volatile T* s_data, size_t const data_len, u4 const id_base);
template <typename bheader_v3_t, typename T, int TileDim, int Seq, int NumThreads, size_t SymCodec> __device__ __forceinline__ bool handle_incomp_non_final_and_signal_exit (u4* s_incomp, bheader_v3_t* s_v3_bheader, u1* pbk_bs_v3, size_t stride_bytes, T* s_data, size_t const data_len, u4 const id_base, bheader_v3_t* v3_headers);

// eq load + top-1 count
template <typename T, int ChunkSize, int ShardSize, int NumThreads, int Radius, bool R1>  __forceinline__ __device__ void load_eq_and_count_top1_v2   (T const* in_eq, size_t data_len, u4 id_base, int* p_eq, volatile u4* s_top1_counts);

// breaks routers
template <typename BreakCell, typename Off>     struct _router_inline_breaks;

// clang-format on

}  // namespace hfr_pbk

namespace hfr_pbk {

// LUT: to select book based on the top1 probability
static __device__ __constant__ float probs_lookup[] = {
    0.99531072, 0.89518499, 0.74531221, 0.58350813, 0.54362822, 0.42192072, 0.34299222, 0.35969448,
    0.3462581,  0.28855956, 0.25986451, 0.22411603, 0.18871215, 0.15629803, 0.12847738, 0.10806277,
    0.09283233, 0.08313504, 0.07545929, 0.06983808, 0.06501415, 0.0612177,  0.05761271, 0.05464315,
    0.05169559, -1,         -1,         -1,         -1,         -1,         -1,         -1};

// LUT: to select merge level based on the top1 probability
static __device__ __constant__ int merge_lookup[]{4, 4, 4, 4,  3,  3,  3,  3,  3,  3, 3,
                                                  3, 2, 2, 2,  2,  2,  2,  2,  2,  2, 2,
                                                  2, 2, 2, -1, -1, -1, -1, -1, -1, -1};

struct slot_fixed_stride {
  u4 stride_bytes;

  template <typename Header>
  __device__ __forceinline__ void reserve(
      u4 b, u4 /*p_wbytes*/, volatile u4* s_start_off, volatile Header* bheader) const
  {
    if (threadIdx.x == 0) {
      u4 start = b * stride_bytes;
      *s_start_off = start;
      bheader->entry = start;
    }
    __syncthreads();
  }

  __forceinline__ __device__ void commit(u4 /*b*/) const {}
};

struct slot_lago_ticket {  // do-wait-write
  u4* d_cursor;
  u4* d_ticket;

  template <typename Header>
  __forceinline__ __device__ void reserve(
      u4 b, u4 p_wbytes, volatile u4* s_start_off, volatile Header* bheader) const
  {
    if (threadIdx.x == 0) {
      (void)b;
      u4 start = atomicAdd(d_cursor, p_wbytes);
      *s_start_off = start;
      bheader->entry = start;
    }
    __syncthreads();
  }

  __forceinline__ __device__ void commit(u4 /*logical_bid*/) const {}
};

// Decoupled-lookback per-block prefix scan slot. Requires cooperative launch.
struct slot_decoupled_lookback {
  volatile u4* d_state;  // [nblock] packed (flag:2 | value:30)

  static __device__ __forceinline__ u4 pack(u4 v, u4 f) { return (f << 30) | (v & 0x3FFFFFFFu); }
  static __device__ __forceinline__ u4 unp_v(u4 p) { return p & 0x3FFFFFFFu; }
  static __device__ __forceinline__ u4 unp_f(u4 p) { return p >> 30; }

  template <typename Header>
  __forceinline__ __device__ void reserve(
      u4 b, u4 p_wbytes, volatile u4* s_start_off, volatile Header* bheader) const
  {
    // Decoupled-lookback tile-status states (CUB-aligned naming).
    constexpr u4 INVALID = 0, PARTIAL = 1, INCLUSIVE = 2;
    __shared__ u4 s_excl_prefix;

    if (threadIdx.x == 0) {
      d_state[b] = pack(p_wbytes, PARTIAL);
      __threadfence();
    }
    __syncthreads();

    if (b == 0) {
      if (threadIdx.x == 0) {
        d_state[0] = pack(p_wbytes, INCLUSIVE);
        __threadfence();
        s_excl_prefix = 0;
      }
      __syncthreads();
      *s_start_off = s_excl_prefix;
      bheader->entry = s_excl_prefix;
      return;
    }

    if (threadIdx.x < 32) {
      u4 excl_prefix = 0;
      int predecessor_idx = (int)b - 1 - (int)threadIdx.x;
      bool done = false;
      while (not done) {
        u4 predecessor_status, value;
        if (predecessor_idx >= 0) {
          u4 st = d_state[predecessor_idx];
          while (unp_f(st) == INVALID) st = d_state[predecessor_idx];  // WaitForValid
          predecessor_status = unp_f(st);
          value = unp_v(st);
        }
        else {
          predecessor_status = INCLUSIVE;
          value = 0;  // synthetic OOB stop past tile 0
        }

        u4 incl_mask = __ballot_sync(0xffffffffu, predecessor_status == INCLUSIVE);
        int tail_lane = (incl_mask != 0u) ? (__ffs(incl_mask) - 1) : -1;

        u4 window_aggregate;
        if (tail_lane >= 0) {
          window_aggregate = ((int)threadIdx.x <= tail_lane) ? value : 0u;
          done = true;
        }
        else {
          window_aggregate = value;
          predecessor_idx -= 32;
        }
#pragma unroll
        for (int s = 16; s > 0; s >>= 1)
          window_aggregate += __shfl_xor_sync(0xffffffffu, window_aggregate, s);
        excl_prefix += window_aggregate;
      }

      if (threadIdx.x == 0) {
        d_state[b] = pack(excl_prefix + p_wbytes, INCLUSIVE);
        __threadfence();
        s_excl_prefix = excl_prefix;
      }
    }
    __syncthreads();
    *s_start_off = s_excl_prefix;
    bheader->entry = s_excl_prefix;
  }

  __forceinline__ __device__ void commit(u4 /*b*/) const {}
};

// utils: find proper pbk based on tiny histogram
template <int PbkNumBooks>
__forceinline__ __device__ int _pbk_argmin(float prob)
{
  float min_diff = fabsf(probs_lookup[threadIdx.x] - prob);
  int min_idx = threadIdx.x;
  unsigned mask = __ballot_sync(0xffffffff, threadIdx.x < PbkNumBooks);
  for (int offset = 16; offset > 0; offset /= 2) {
    float shfl_min = __shfl_down_sync(mask, min_diff, offset);
    int shfl_idx = __shfl_down_sync(mask, min_idx, offset);
    if (shfl_min < min_diff and threadIdx.x < PbkNumBooks) min_diff = shfl_min, min_idx = shfl_idx;
  }
  return min_idx;  // valid in lane 0
}

// utils: find proper pbk based on tiny histogram (per-block top-1 prob).
template <int TileDim, int PbkNumBooks, typename Header_v3>
__forceinline__ __device__ void find_proper_book(
    volatile u4* s_top1_counts, volatile Header_v3* v3_bheader, size_t data_len, u4 chunk_id)
{
  if (threadIdx.x < 32) {
    float prob = 0.0;
    if (threadIdx.x == 0) {
      size_t valid = min((size_t)TileDim, data_len - (size_t)TileDim * chunk_id);
      prob = *s_top1_counts * 1.0 / valid;
    }
    prob = __shfl_sync(0xffffffff, prob, 0);
    int min_idx = _pbk_argmin<PbkNumBooks>(prob);
    if (threadIdx.x == 0) v3_bheader->enc_id = min_idx;
  }
  __syncthreads();
}

// utils: load the proper pbk
template <int PbkBookLen, typename Header_v3>
__forceinline__ __device__ void load_proper_book(
    u4& ReduceTimes, volatile u4* s_book, u4* pbk, volatile Header_v3* v3_bheader)
{
  auto tree_idx = 0u;

  if ((threadIdx.x & 31) == 0) tree_idx = v3_bheader->enc_id;

  tree_idx = __shfl_sync(0xffffffff, tree_idx, 0);

  if ((threadIdx.x & 31) == 0) ReduceTimes = min(merge_lookup[tree_idx], ReduceTimes);
  ReduceTimes = __shfl_sync(0xffffffff, ReduceTimes, 0);

  for (auto idx = threadIdx.x; idx < PbkBookLen; idx += blockDim.x)
    s_book[idx] = pbk[idx + tree_idx * PbkBookLen];

  __syncthreads();  // make s_book[] writes visible to reduce_merge
}

// reduce-merge: fixed BaseSeq threads, Iters=BaseSeq/MergeSize words each.
template <int ChunkSize, int BitWidth, int Radius, int BaseSeq, int MergeSize, typename Header_v3>
__forceinline__ __device__ void rmerge_sync__v7_const_shardsize_iter(
    size_t const data_len, u4 const chunk_id, volatile u4* s_book, int p_eq[BaseSeq],
    psz::HFR_PBK_Breaks<Radius>* br2_val_idx, volatile u4* s_v3_incomp,
    volatile Header_v3* v3_bheader, u4* r_reduced, u4* r_bits)
{
  static_assert(BaseSeq % MergeSize == 0, "BaseSeq must be a multiple of MergeSize.");
  constexpr int Iters = BaseSeq / MergeSize;
  auto bitcount_of = [](Hf* _w) { return reinterpret_cast<W*>(_w)->bitcount; };
  auto entry = [&]() -> size_t { return ChunkSize * chunk_id; };
  auto allowed_len = [&]() { return min((size_t)ChunkSize, data_len - entry()); };

#pragma unroll
  for (int it = 0; it < Iters; it++) {
    const int wbase = it * MergeSize;

    Hf r_code[MergeSize];
    u4 r_width[MergeSize];
    u4 r_valid[MergeSize];
#pragma unroll
    for (int j = 0; j < MergeSize; j++) {
      const size_t idx = (size_t)threadIdx.x * BaseSeq + wbase + j;
      // valid eq is < 2*Radius (book domain)
      // restrict s_book in-range on malformed input
      auto p_val = s_book[max(min(p_eq[wbase + j], 2 * Radius - 1), 0)];
      const auto sym_bits = bitcount_of(&p_val);
      r_code[j] = p_val << (BitWidth - sym_bits);
      r_width[j] = sym_bits;
      r_valid[j] = (idx < allowed_len()) ? 1u : 0u;
    }

    u4 r_offset[MergeSize];
    u4 acc = 0;
#pragma unroll
    for (int j = 0; j < MergeSize; j++) {
      r_offset[j] = acc;
      acc += r_width[j] * r_valid[j];
    }
    Hf p_reduced = 0x0;
    u4 p_bits = acc;

#pragma unroll
    for (int j = 0; j < MergeSize; j++) p_reduced |= (r_code[j] >> r_offset[j]);

  if (p_bits > BitWidth) {
      p_bits = 0u;
      p_reduced = 0x0;
      auto p_val_ref = s_book[Radius];
      const auto sym_bits_ref = bitcount_of(&p_val_ref);
#pragma unroll
      for (int j = 0; j < MergeSize; j++) {
        const u4 br_lidx = (u4)((size_t)threadIdx.x * BaseSeq + wbase + j);
        auto p_val = s_book[max(min(p_eq[wbase + j], 2 * Radius - 1), 0)];
        auto sym_bits = bitcount_of(&p_val);
        if (sym_bits > (BitWidth / MergeSize)) {
          auto _l_br_idx = atomicAdd(const_cast<u4*>(s_v3_incomp), 1 << 16);
          auto l_br_idx = (_l_br_idx & Constants::MASK_BREAKS) >> 16;
          constexpr u4 MaxNumBreaks = ChunkSize / 16 - 1;  // runtime; FIXME: conslidate with outer HFR_PBK_C*::MaxNumBreaks
          if (l_br_idx < MaxNumBreaks) {
            br2_val_idx[l_br_idx] = {(u2)p_eq[wbase + j], (u2)br_lidx};
            p_val = p_val_ref;
            sym_bits = sym_bits_ref;
          }
          else
            atomicOr(const_cast<u4*>(s_v3_incomp), true);
        }
        p_val <<= (BitWidth - sym_bits);
        p_reduced |= (p_val >> p_bits);
        p_bits += sym_bits * (br_lidx < allowed_len());
      }
    }

    r_reduced[it] = p_reduced;
    r_bits[it] = p_bits;
  }

  __syncthreads();
  if (threadIdx.x == 0)
    if ((s_v3_incomp[0] & Constants::MASK_TF) == 0)
      v3_bheader->n_breaks = (s_v3_incomp[0] & Constants::MASK_BREAKS) >> 16;
}

// concatenate bitstream (v2: Iters=1 only).
// s_reduced >= ChunkSize/MergeSize + 1.
template <int ChunkSize, int BitWidth, int BaseSeq, int MergeSize>
__forceinline__ __device__ void smerge_sync__v7_const_shardsize_iter(
    volatile u4* s_reduced, volatile u4* s_bitcount, u4* r_reduced, u4* r_bits)
{
  static_assert(BitWidth == 32, "assumes BitWidth = 32 (u4 words).");
  static_assert(BaseSeq % MergeSize == 0, "BaseSeq must be a multiple of MergeSize.");
  constexpr int Iters = BaseSeq / MergeSize;
  constexpr int NumThreads = ChunkSize / BaseSeq;
  constexpr int NumWarps = NumThreads / 32;
  constexpr int MaxWords = ChunkSize / MergeSize;  // words this M can touch

  __shared__ u4 s_warp_totals[NumWarps];

  for (int i = threadIdx.x; i <= MaxWords; i += NumThreads) s_reduced[i] = 0;

  u4 thread_total = 0;
  u4 local_excl[Iters];
#pragma unroll
  for (int it = 0; it < Iters; it++) {
    local_excl[it] = thread_total;
    thread_total += r_bits[it];
  }

  u4 base_excl =
      phf::block_scan::block_incl_scan_u32<NumThreads>(thread_total, s_warp_totals) - thread_total;
  const u4 block_total = s_warp_totals[NumWarps - 1];

#pragma unroll
  for (int it = 0; it < Iters; it++) {
    if (r_bits[it] == 0) continue;
    const u4 bit_off = base_excl + local_excl[it];
    const u4 lo_word = bit_off >> 5;
    const u4 used__bits = bit_off & (BitWidth - 1);
    const u4 unused_bits = BitWidth - used__bits;
    atomicOr(const_cast<u4*>(s_reduced + lo_word), r_reduced[it] >> used__bits);
    if (used__bits > 0)
      atomicOr(const_cast<u4*>(s_reduced + lo_word + 1), r_reduced[it] << unused_bits);
  }

  if (threadIdx.x == 0) s_bitcount[0] = block_total;
  __syncthreads();
}

// Dispatch runtime reduce_times to constexpr MergeSize = 1<<RT (only M <= 1<<BaseRT).
template <int ChunkSize, int BitWidth, int Radius, int BaseRT, typename Header_v3, int RT = BaseRT>
__forceinline__ __device__ void rmerge__v7_wrapper(
    u4 const reduce_times, size_t const data_len, u4 const chunk_id, volatile u4* s_book,
    int* p_eq, psz::HFR_PBK_Breaks<Radius>* br2_val_idx, volatile u4* s_v3_incomp,
    volatile Header_v3* v3_bheader, u4* r_reduced, u4* r_bits)
{
  if (reduce_times == (u4)RT)
    rmerge_sync__v7_const_shardsize_iter<
        ChunkSize, BitWidth, Radius, (1 << BaseRT), (1 << RT), Header_v3>(
        data_len, chunk_id, s_book, p_eq, br2_val_idx, s_v3_incomp, v3_bheader, r_reduced, r_bits);
  else if constexpr (RT > 1)
    rmerge__v7_wrapper<ChunkSize, BitWidth, Radius, BaseRT, Header_v3, RT - 1>(
        reduce_times, data_len, chunk_id, s_book, p_eq, br2_val_idx, s_v3_incomp, v3_bheader,
        r_reduced, r_bits);
}

template <int ChunkSize, int BitWidth, int BaseRT, int RT = BaseRT>
__forceinline__ __device__ void smerge__v7_wrapper(
    u4 const reduce_times, volatile u4* s_reduced, volatile u4* s_bitcount, u4* r_reduced,
    u4* r_bits)
{
  if (reduce_times == (u4)RT)
    smerge_sync__v7_const_shardsize_iter<ChunkSize, BitWidth, (1 << BaseRT), (1 << RT)>(
        s_reduced, s_bitcount, r_reduced, r_bits);
  else if constexpr (RT > 1)
    smerge__v7_wrapper<ChunkSize, BitWidth, BaseRT, RT - 1>(
        reduce_times, s_reduced, s_bitcount, r_reduced, r_bits);
}

// RMerge/SMerge symbols are hfr-pbk_ver.hh.
// bundled context for per-block merge + a version tag.
template <class C>
struct MergeCtx {
  using Header = typename C::bheader_t;
  using BreakCell = psz::HFR_PBK_Breaks<C::Radius>;
  size_t data_len;
  u4 chunk_id;
  u4 reduce_times;
  volatile u4* s_book;
  int* p_eq;
  BreakCell* s_breaks;
  volatile u4* s_v3_incomp;
  volatile Header* bheader;
  volatile u4* s_reduced;
  volatile u4* s_bitcount;
  u4* r_reduced;
  u4* r_bits;
};

template <RMerge V, class C>
__forceinline__ __device__ void dispatch_rmerge(MergeCtx<C> cx)
{
  constexpr int ChunkSize = C::ChunkSize;
  constexpr int BitWidth = C::BITWIDTH;
  constexpr int Radius = C::Radius;
  // BaseSeq = 1<<BaseRT; runtime reduce_times selects MergeSize.
  // Iters = BaseSeq/MergeSize = 1 << IterLog
  constexpr int BaseRT = (int)C::ReduceTimes + (int)C::IterLog;
  using Header = typename MergeCtx<C>::Header;
  static_assert(V == RMerge::v7, "release build: v7 only");

  rmerge__v7_wrapper<ChunkSize, BitWidth, Radius, BaseRT, Header>(
      cx.reduce_times, cx.data_len, cx.chunk_id, cx.s_book, cx.p_eq, cx.s_breaks, cx.s_v3_incomp,
      cx.bheader, cx.r_reduced, cx.r_bits);
}

template <SMerge V, class C>
__forceinline__ __device__ void dispatch_smerge(MergeCtx<C> cx)
{
  constexpr int ChunkSize = C::ChunkSize;
  constexpr int BitWidth = C::BITWIDTH;
  constexpr int BaseRT = (int)C::ReduceTimes + (int)C::IterLog;
  static_assert(V == SMerge::v7, "release build: v7 only");
  smerge__v7_wrapper<ChunkSize, BitWidth, BaseRT>(
      cx.reduce_times, cx.s_reduced, cx.s_bitcount, cx.r_reduced, cx.r_bits);
}

template <class Launch>
__host__ void dispatch_merge_host(RMerge rm, SMerge sm, Launch&& launch)
{
  (void)rm;
  (void)sm;
  launch(
      std::integral_constant<RMerge, RMerge::v7>{}, std::integral_constant<SMerge, SMerge::v7>{});
}

namespace {

template <typename bheader_v3_t, typename T, int TileDim, int Seq, int NumThreads, size_t SymCodec>
__device__ __forceinline__ void _handle_incomp_non_final(
    volatile bheader_v3_t* s_v3_bheader, u1* pbk_bs_v3, size_t stride_bytes, volatile T* s_data,
    size_t const data_len, u4 const id_base)
{
  using incomp_eq_t = __half;
  __shared__ size_t s_loc_thisblk;

  if (threadIdx.x == 0) {
    s_v3_bheader->enc_id = SymCodec;
    s_v3_bheader->n_unpred = 0;
    s_v3_bheader->n_breaks = 0;
    s_v3_bheader->dense = (TileDim * sizeof(incomp_eq_t) * 8) >> 5;  // words
    s_loc_thisblk = (size_t)blockIdx.x * stride_bytes;
    s_v3_bheader->entry = s_loc_thisblk;
  }
  __syncthreads();

  size_t p_loc_thisblk;
  if ((threadIdx.x & 31) == 0) p_loc_thisblk = s_loc_thisblk;
  p_loc_thisblk = __shfl_sync(0xffffffff, p_loc_thisblk, 0);

#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto l_id = threadIdx.x + ix * NumThreads;
    auto id = id_base + l_id;
    if (id < data_len) {
      ((__half*)(pbk_bs_v3 + p_loc_thisblk))[l_id] = __float2half((float)s_data[l_id]);
    }
  }
}

}  // namespace

template <typename bheader_v3_t, typename T, int TileDim, int Seq, int NumThreads, size_t SymCodec>
__device__ __forceinline__ bool handle_incomp_non_final_and_signal_exit(
    u4* s_incomp, bheader_v3_t* s_v3_bheader, u1* pbk_bs_v3, size_t stride_bytes, T* s_data,
    size_t const data_len, u4 const id_base, bheader_v3_t* v3_headers)
{
  u4 p_incomp{0};
  if ((threadIdx.x & 31) == 0) p_incomp = *s_incomp & Constants::MASK_TF;
  __syncthreads();
  p_incomp = __shfl_sync(0xffffffff, p_incomp, 0);
  if (p_incomp) {
    _handle_incomp_non_final<bheader_v3_t, T, TileDim, Seq, NumThreads, SymCodec>(
        s_v3_bheader, pbk_bs_v3, stride_bytes, s_data, data_len, id_base);
    if (threadIdx.x == 0) v3_headers[blockIdx.x] = *s_v3_bheader;
    return true;
  }
  return false;
}

// fused gmem->register load + top-1 count (skips the s_eq_in transit)
// R1=true: coalesced SoA-u32 per-warp transit for BytesPerThread > 16.
template <typename T, int ChunkSize, int ShardSize, int NumThreads, int Radius, bool R1 = false>
__forceinline__ __device__ void load_eq_and_count_top1_v2(
    T const* in_eq, size_t data_len, u4 id_base, int* p_eq, volatile u4* s_top1_counts)
{
  constexpr u4 BytesPerThread = ShardSize * (u4)sizeof(T);
  static_assert(
      BytesPerThread >= 4 ? (BytesPerThread & 3) == 0 : true,
      "fused load (>=4 bytes/thread) must be a multiple of 4; <4 uses scalar fallback");

  u4 thp_top1_count = 0;

  auto emit = [&](T const* sym) {
#pragma unroll
    for (auto ix = 0; ix < ShardSize; ix++) {
      p_eq[ix] = (int)sym[ix];
      psz::warp_top1_count(p_eq[ix] == Radius, thp_top1_count);
    }
  };

  if ((size_t)id_base + ChunkSize > data_len) {
#pragma unroll
    for (auto ix = 0; ix < ShardSize; ix++) {
      auto idx = threadIdx.x * ShardSize + ix;
      auto id = (size_t)id_base + idx;
      bool valid = (id < data_len);
      p_eq[ix] = valid ? (int)in_eq[id] : 0;
      psz::warp_top1_count(valid and (p_eq[ix] == Radius), thp_top1_count);
    }
  }
  else if constexpr (BytesPerThread < 4) {  // low-RT (ShardSize*sizeof(T) in {1,2}): scalar
#pragma unroll
    for (auto ix = 0; ix < ShardSize; ix++) {
      auto idx = threadIdx.x * ShardSize + ix;
      p_eq[ix] = (int)in_eq[(size_t)id_base + idx];
      psz::warp_top1_count(p_eq[ix] == Radius, thp_top1_count);
    }
  }
  else if constexpr (BytesPerThread == 4) {
    u4 v = reinterpret_cast<u4 const*>(in_eq + id_base)[threadIdx.x];
    emit(reinterpret_cast<T const*>(&v));
  }
  else if constexpr (BytesPerThread == 8) {
    uint2 v = reinterpret_cast<uint2 const*>(in_eq + id_base)[threadIdx.x];
    emit(reinterpret_cast<T const*>(&v));
  }
  else if constexpr (BytesPerThread == 16) {
    uint4 v = reinterpret_cast<uint4 const*>(in_eq + id_base)[threadIdx.x];
    emit(reinterpret_cast<T const*>(&v));
  }
  else if constexpr (not R1) {  // stride-2 fast path (default)
    constexpr u4 N = BytesPerThread / 16;
    auto src = reinterpret_cast<uint4 const*>(in_eq + id_base);
    uint4 v[N];
#pragma unroll
    for (u4 i = 0; i < N; i++) v[i] = src[N * threadIdx.x + i];
    emit(reinterpret_cast<T const*>(&v[0]));
  }
  else {  // R1: warp-coalesced + SoA-u32 per-warp transit
    constexpr u4 N = BytesPerThread / 16;
    constexpr u4 NumWarps = NumThreads / 32;
    __shared__ u4 s_x[NumWarps][32 * N], s_y[NumWarps][32 * N], s_z[NumWarps][32 * N],
        s_w[NumWarps][32 * N];

    const u4 lane = threadIdx.x & 31, warp_id = threadIdx.x >> 5;
    const u4 warp_base = warp_id * 32 * N;
    auto src = reinterpret_cast<uint4 const*>(in_eq + id_base);

#pragma unroll
    for (u4 i = 0; i < N; i++) {
      uint4 v = src[warp_base + i * 32 + lane];  // coalesced gmem
      u4 slot = i * 32 + lane;
      s_x[warp_id][slot] = v.x;
      s_y[warp_id][slot] = v.y;
      s_z[warp_id][slot] = v.z;
      s_w[warp_id][slot] = v.w;
    }
    __syncwarp();

    uint4 own[N];
#pragma unroll
    for (u4 i = 0; i < N; i++) {
      u4 slot = lane * N + i;
      own[i] = {s_x[warp_id][slot], s_y[warp_id][slot], s_z[warp_id][slot], s_w[warp_id][slot]};
    }
    emit(reinterpret_cast<T const*>(&own[0]));
  }

  if ((threadIdx.x & 31) == 0) atomicAdd(const_cast<u4*>(s_top1_counts), thp_top1_count);
  __syncthreads();
}

// generalized per-block writer: pairs slot policy with breaks router.
template <typename BreakCell, typename Off = ull>
struct _router_inline_breaks {
  using offset_t = Off;

  __device__ __forceinline__ Off breaks_bytes(u4 n_breaks) const
  {
    return (Off)n_breaks * sizeof(BreakCell);
  }
  __device__ __forceinline__ BreakCell* breaks_base(u1* block_entry) const
  {
    return (BreakCell*)block_entry;
  }
  __device__ __forceinline__ u1* bitstream_base(u1* block_entry, Off breaks_bytes_) const
  {
    return block_entry + breaks_bytes_;
  }
};

// Offset type deduced from BreaksRouter::offset_t (u4 for detached, ull for inline).
template <typename Slot, typename BreaksRouter, typename Hf, typename Header, typename BreakCell>
__forceinline__ __device__ void write_pbk_bitstream_v2(
    u4 b, volatile u4* s_bitcount, volatile Hf* s_reduced, u1* dn_base, volatile Header* bheader,
    volatile BreakCell* s_breaks, Slot slot, BreaksRouter router,
    psz::OutlierCell const* block_outliers)
{
  using Off = typename BreaksRouter::offset_t;
  using psz::OutlierCell;
  __shared__ u4 s_wunits;
  __shared__ Off s_wloc;

  if (threadIdx.x == 0) {
    u4 p_bc = s_bitcount[0];
    bheader->dense = (p_bc + 31u) >> 5;  // bits -> 32-bit words
    s_wunits = bheader->dense;
  }
  __syncthreads();

  Off breaks_bytes_ = router.breaks_bytes(bheader->n_breaks);
  u4 const n_unpred = block_outliers ? bheader->n_unpred : 0u;
  // unpred trails the bitstream; pad the section to a word so the next block's entry stays aligned.
  Off unpred_bytes_ = ((Off)n_unpred * (Off)sizeof(OutlierCell) + 3u) & ~(Off)3u;
  Off p_wbytes = (Off)s_wunits * sizeof(Hf) + breaks_bytes_ + unpred_bytes_;
  slot.reserve(b, p_wbytes, &s_wloc, bheader);

  auto block_entry = dn_base + s_wloc;
  auto breaks_base = router.breaks_base(block_entry);
  auto bs_base = (Hf*)router.bitstream_base(block_entry, breaks_bytes_);
  // [breaks | bitstream | unpred]
  auto unpred_base = (OutlierCell*)((u1*)bs_base + (Off)s_wunits * sizeof(Hf));

  // breaks cap can exceed blockDim (4Ki: 255 vs 128 threads) -> strided copy
  for (u4 i = threadIdx.x; i < (u4)bheader->n_breaks; i += blockDim.x)
    breaks_base[i] = const_cast<BreakCell*>(s_breaks)[i];

#pragma unroll
  for (auto i = threadIdx.x; i < s_wunits; i += blockDim.x) bs_base[i] = (Hf)s_reduced[i];

  if (threadIdx.x < n_unpred)
    unpred_base[threadIdx.x] =
        block_outliers[(size_t)b * Header::C::MaxNumUnpred + threadIdx.x];

  slot.commit(b);
}

}  // namespace hfr_pbk

// boilerplate
#define HFR_PBK_USING_HELPERS()                        \
  using hfr_pbk::find_proper_book;                     \
  using hfr_pbk::load_proper_book;                     \
  using hfr_pbk::rmerge_sync__v7_const_shardsize_iter; \
  using hfr_pbk::smerge_sync__v7_const_shardsize_iter; \
  using hfr_pbk::rmerge__v7_wrapper;                   \
  using hfr_pbk::smerge__v7_wrapper;                   \
  using hfr_pbk::MergeCtx;                             \
  using hfr_pbk::dispatch_rmerge;                      \
  using hfr_pbk::dispatch_smerge;                      \
  using hfr_pbk::dispatch_merge_host;

#define HFR_PBK_TYPEDEFS_AND_CONSTEXPRS(C)                                     \
  using T = typename C::T;                                                     \
  using Hf = typename C::Hf;                                                   \
  using KC = psz::_parameterized_hfr_pbk_constants<(size_t)C::Magnitude>;      \
  using Header = psz::_future::bheader<T, C::Radius, (size_t)C::Magnitude>;    \
  using BreakCell = psz::HFR_PBK_Breaks<C::Radius>;                            \
  constexpr auto ChunkSize = C::ChunkSize;                                     \
  constexpr auto ShardSize = C::ShardSize;                                     \
  constexpr auto NumThreads = C::BlockDim;                                     \
  constexpr auto BitWidth = C::BITWIDTH;                                       \
  constexpr auto BookLen = C::BookLen;                                         \
  constexpr auto NumBooks = C::NumBooks;                                       \
  constexpr auto Radius = C::Radius;                                           \
  constexpr auto ShuffleTimes = C::ShuffleTimes;

#define HFR_PBK_SHARED_AND_RESET()                                          \
  __shared__ alignas(16) T s_eq_in[ChunkSize];                              \
  __shared__ Hf s_book[BookLen];                                            \
  constexpr auto ReducedSize = ChunkSize / (ShardSize >= 2 ? 2 : 1) + 1;    \
  __shared__ Hf s_reduced[ReducedSize]; /* v7: +1 for lo_word+1 failsafe */ \
  __shared__ u4 s_bitcount[NumThreads];                                     \
  __shared__ u4 s_top1_counts;                                              \
  __shared__ u4 s_v3_incomp;                                                \
  __shared__ Header s_bheader;                                              \
  __shared__ BreakCell s_breaks[KC::MaxNumBreaks];                          \
  if (threadIdx.x == 0) s_bheader = {};                                     \
  if (threadIdx.x == 32) s_top1_counts = 0;                                 \
  if (threadIdx.x == 64 % NumThreads) s_v3_incomp = 0;
