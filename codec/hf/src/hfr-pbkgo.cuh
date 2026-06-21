// HFR-PBK_Compat: blockwise, globally ordering resolved in one kernel
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "_future/hfr-pbk.cuh"   // shared LUTs + find_proper_book / load_proper_book
#include "_future/warp_top1.cuh"  // psz::warp_top1_count
#include "hf_impl.hh"
#include "hfr.hh"

using u1 = uint8_t;
using u2 = uint16_t;
using u4 = uint32_t;
using ull = unsigned long long;

namespace phf {

HFR_PBK_USING_HELPERS()
using hfr_pbk::_router_inline_breaks;
using hfr_pbk::slot_decoupled_lookback;
using hfr_pbk::slot_lago_ticket;
using hfr_pbk::write_pbk_bitstream_v2;

// Write raw payload when this block is incompressible.
template <
    typename T, int ChunkSize, int ShardSize, int NumThreads, typename Hf, typename Header,
    typename Slot>
__forceinline__ __device__ void handle_incomp_pbkgo(
    u4 b, volatile Header* bheader, Hf* dn_bitstream, T* in_eq, size_t const data_len,
    u4 const id_base, Slot slot)
{
  __shared__ u4 s_start_off;
  __shared__ u4 s_wunits;

  if (threadIdx.x == 0) {
    bheader->enc_id = (u4)psz::HFR_PBK_Constants::CodeIncompBreaks;
    bheader->n_breaks = 0;
    bheader->bits = ChunkSize * sizeof(T) * 8;
    s_wunits = (bheader->bits + 31u) >> 5;
  }
  __syncthreads();

  slot.reserve(b, s_wunits * sizeof(Hf), &s_start_off, bheader);

  auto bs_base = (Hf*)((u1*)dn_bitstream + s_start_off);
  auto base = (T*)bs_base;

#pragma unroll
  for (auto ix = 0; ix < ShardSize; ix++) {
    auto l_id = threadIdx.x + ix * NumThreads;
    auto id = id_base + l_id;
    if (id < data_len) base[l_id] = in_eq[id];
  }

  slot.commit(b);
}

// min-blocks/SM launch hint: 8 on server tiers (sm_70/80/90), 6 elsewhere.
#if __CUDA_ARCH__ == 700 || __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900
#define PBKGO_MIN_BLOCKS_PER_SM 8
#else
#define PBKGO_MIN_BLOCKS_PER_SM 6
#endif

// last logical block also publishes total_cells, thread-0-only.
template <typename Header, typename Hf>
__forceinline__ __device__ void emit_packed_and_total(
    u4 b, u4 nblock, volatile Header const* bheader, u4 p_wbytes, u4* dn_packed_headers,
    u4* d_total_cells)
{
  if (threadIdx.x != 0) return;
  u4 const bits_v = bheader->bits;
  u4 const enc_id_v = bheader->enc_id;
  u4 const entry_v = bheader->entry;  // byte offset
  dn_packed_headers[2 * b + 0] = (bits_v << 14) | (enc_id_v << 9);
  dn_packed_headers[2 * b + 1] = entry_v;
  if (b == nblock - 1) *d_total_cells = (entry_v + p_wbytes) / (u4)sizeof(Hf);
}

template <class C, RMerge RM = RMerge::v7, SMerge SM = SMerge::v7>
__global__ __launch_bounds__(C::BlockDim, PBKGO_MIN_BLOCKS_PER_SM) void KCU_HFR_PBKGO_encode(
    typename C::T* in_eq, size_t data_len, typename C::Hf* dram_pbk, typename C::Hf* dn_bitstream,
    u4* dn_packed_headers, u4* d_total_cells, u4* d_state)
{
  static_assert(merge_compatible(RM, SM), "RMerge/SMerge data-handoff contract mismatch");
  HFR_PBK_TYPEDEFS_AND_CONSTEXPRS(C);
  HFR_PBK_SHARED_AND_RESET();
  __syncthreads();

  slot_decoupled_lookback slot{(volatile u4*)d_state};

  const u4 nblock = (u4)((data_len + ChunkSize - 1) / ChunkSize);
  const u4 b = (u4)blockIdx.x;
  if (b >= nblock) return;
  auto const id_base = b * ChunkSize;

  int p_eq[ShardSize];
  hfr_pbk::load_eq_and_count_top1_v2<T, ChunkSize, ShardSize, NumThreads, Radius>(
      in_eq, data_len, id_base, p_eq, &s_top1_counts);

  u4 reduce_times = C::ReduceTimes;
  find_proper_book<ChunkSize, NumBooks>(&s_top1_counts, &s_bheader, data_len, b);
  load_proper_book<BookLen>(reduce_times, s_book, dram_pbk, &s_bheader);

  constexpr int MaxIters = ShardSize / 2;
  u4 r_reduced[MaxIters], r_bits[MaxIters];
  MergeCtx<C> cx{data_len,     b,          reduce_times, (volatile u4*)s_book, p_eq,      s_breaks,
                 &s_v3_incomp, &s_bheader, s_reduced,    s_bitcount,           r_reduced, r_bits};
  dispatch_rmerge<RM>(cx);

  {
    u4 p_incomp = 0;
    if ((threadIdx.x & 31) == 0) p_incomp = s_v3_incomp & psz::HFR_PBK_Constants::MASK_TF;
    __syncthreads();
    p_incomp = __shfl_sync(0xffffffff, p_incomp, 0);
    if (p_incomp) {
      handle_incomp_pbkgo<T, ChunkSize, ShardSize, NumThreads>(
          b, &s_bheader, dn_bitstream, in_eq, data_len, id_base, slot);
      u4 const p_wbytes = ((s_bheader.bits + 31u) >> 5) * (u4)sizeof(Hf);
      emit_packed_and_total<Header, Hf>(
          b, nblock, &s_bheader, p_wbytes, dn_packed_headers, d_total_cells);
      return;
    }
  }

  dispatch_smerge<SM>(cx);
  write_pbk_bitstream_v2(
      b, s_bitcount, s_reduced, (u1*)dn_bitstream, &s_bheader, s_breaks, slot,
      _router_inline_breaks<BreakCell, u4>{});
  u4 const p_wbytes = ((s_bheader.bits + 31u) >> 5) * (u4)sizeof(Hf) +
                      (u4)s_bheader.n_breaks * (u4)sizeof(BreakCell);
  emit_packed_and_total<Header, Hf>(
      b, nblock, &s_bheader, p_wbytes, dn_packed_headers, d_total_cells);
}

}  // namespace phf

namespace phf::module {

template <typename T, int Magnitude, int ReduceTimes, typename Hf, uint16_t Radius>
int HFR_PBKGO_encode<T, Magnitude, ReduceTimes, Hf, Radius>::max_blocks_per_sm()
{
  using C = phf::HFR_PBKGO_Config<T, Magnitude, ReduceTimes, Hf, Radius>;
  int n = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &n, phf::KCU_HFR_PBKGO_encode<C>, (int)C::BlockDim, 0);
  return n;
}

template <typename T, int Magnitude, int ReduceTimes, typename Hf, uint16_t Radius>
int HFR_PBKGO_encode<T, Magnitude, ReduceTimes, Hf, Radius>::GPU_kernel(
    T* in_eq, size_t len, Hf* dram_pbk, Hf* dn_bitstream, uint32_t* dn_packed_headers,
    uint32_t* d_total_cells, uint32_t* d_state, int max_resident_blocks, void* stream, RMerge rm,
    SMerge sm)
{
  using C = phf::HFR_PBKGO_Config<T, Magnitude, ReduceTimes, Hf, Radius>;

  constexpr auto nthread = C::BlockDim;
  (void)max_resident_blocks;  // unused: one-shot launch now sizes grid to nblock
  const u4 nblock = (u4)((len + C::ChunkSize - 1) / C::ChunkSize);
  const dim3 grid(nblock, 1, 1);
  const dim3 block((u4)nthread, 1, 1);

  dispatch_merge_host(rm, sm, [&](auto rm_tag, auto sm_tag) {
    constexpr RMerge RM = decltype(rm_tag)::value;
    constexpr SMerge SM = decltype(sm_tag)::value;
    phf::KCU_HFR_PBKGO_encode<C, RM, SM><<<grid, block, 0, (cudaStream_t)stream>>>(
        in_eq, len, dram_pbk, dn_bitstream, dn_packed_headers, d_total_cells, d_state);
  });
  return 0;
}

}  // namespace phf::module

// Instantiation macros — caller TUs invoke; this .inl alone instantiates nothing.
#define __INSTANTIATE_HFR_PBKGO(T, MAG, RED, RAD) \
  template struct phf::module::HFR_PBKGO_encode<T, MAG, RED, uint32_t, RAD>;

// 1-arg form: fan out u1/u2 at canonical MAG=10, RAD=128 (mirrors __INSTANTIATE_RSMERGE_1).
#define __INSTANTIATE_HFR_PBKGO_1(RED)           \
  __INSTANTIATE_HFR_PBKGO(uint8_t, 10, RED, 128) \
  __INSTANTIATE_HFR_PBKGO(uint16_t, 10, RED, 128)
