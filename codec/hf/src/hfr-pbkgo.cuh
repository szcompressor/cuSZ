#ifndef PHF_HFR_PBKGO_C_CUH
#define PHF_HFR_PBKGO_C_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "_future/hfr-pbk.cuh"
#include "_future/hfr_incomp_fb.cuh"
#include "_future/warp_top1.cuh"
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
using phf::hfr_helpers::blk_incomp_fb;

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
  using KC = typename Header::C;
  constexpr u4 EncIdShift = (u4)(KC::BitsMaxNumUnpred + KC::BitsMaxNumBreaks);
  constexpr u4 DenseShift = EncIdShift + (u4)KC::BitsEncId;
  constexpr u4 UnpredMask = (1u << KC::BitsMaxNumUnpred) - 1u;
  u4 const dense_v = bheader->dense;  // words
  u4 const enc_id_v = bheader->enc_id;
  u4 const n_unpred_v = bheader->n_unpred;
  u4 const entry_v = bheader->entry;  // byte offset
  dn_packed_headers[2 * b + 0] =
      (dense_v << DenseShift) | (enc_id_v << EncIdShift) | (n_unpred_v & UnpredMask);
  dn_packed_headers[2 * b + 1] = entry_v;
  if (b == nblock - 1) *d_total_cells = (entry_v + p_wbytes) / (u4)sizeof(Hf);
}

template <class C>
__global__ __launch_bounds__(C::BlockDim, PBKGO_MIN_BLOCKS_PER_SM) void KCU_HFR_PBKGO_encode(
    typename C::T* in_eq, size_t data_len, typename C::Hf* dram_pbk, typename C::Hf* dn_bitstream,
    typename C::bheader_t* dn_headers,
    psz::OutlierCell* block_outliers,
    u4* dn_packed_headers, u4* d_total_cells, u4* d_state)
{
  HFR_PBK_TYPEDEFS_AND_CONSTEXPRS(C);
  HFR_PBK_SHARED_AND_RESET();
  __syncthreads();

  slot_decoupled_lookback slot{(volatile u4*)d_state};

  const u4 nblock = (u4)((data_len + ChunkSize - 1) / ChunkSize);
  const u4 b = (u4)blockIdx.x;
  if (b >= nblock) return;
  auto const id_base = b * ChunkSize;
  __shared__ u4 s_pre_encid;
  // keep predictor unpred + pre-set enc_id (unpred-incomp flag).
  if (threadIdx.x == 0) {
    s_bheader.n_unpred = dn_headers[b].n_unpred;
    s_pre_encid = dn_headers[b].enc_id;
  }
  __syncthreads();

  // unpred-incomp: enc_id=31, eq already carries the raw candidate bits; bypass Huffman.
  if (s_pre_encid == (u4)psz::HFR_PBK_Constants::CodeIncompUnpred) {
    blk_incomp_fb<T, ChunkSize, ShardSize, NumThreads>(
        &s_bheader, dn_bitstream, in_eq, data_len, id_base, slot,
        (u4)psz::HFR_PBK_Constants::CodeIncompUnpred);
    u4 const p_wbytes = s_bheader.dense * (u4)sizeof(Hf);
    if (threadIdx.x == 0) dn_headers[b] = s_bheader;
    emit_packed_and_total<Header, Hf>(
        b, nblock, &s_bheader, p_wbytes, dn_packed_headers, d_total_cells);
    return;
  }

  int p_eq[ShardSize];
  hfr_pbk::load_eq_and_count_top1_v2<T, ChunkSize, ShardSize, NumThreads, Radius>(
      in_eq, data_len, id_base, p_eq, &s_top1_counts);

  u4 reduce_times = C::ReduceTimes;
  find_proper_book<ChunkSize, NumBooks>(&s_top1_counts, &s_bheader, data_len, b);
  load_proper_book<BookLen>(reduce_times, s_book, dram_pbk, &s_bheader);

  constexpr int MaxIters = ShardSize / 2;
  u4 r_reduced[MaxIters], r_bits[MaxIters];
  _merge_ctx<C> cx{data_len,     b,          reduce_times, (volatile u4*)s_book, p_eq,      s_breaks,
                   &s_v3_incomp, &s_bheader, s_reduced,    s_bitcount,           r_reduced, r_bits};
  dispatch_rmerge<'b'>(cx);

  {
    u4 p_incomp = 0;
    if ((threadIdx.x & 31) == 0) p_incomp = s_v3_incomp & psz::HFR_PBK_Constants::MASK_TF;
    __syncthreads();
    p_incomp = __shfl_sync(0xffffffff, p_incomp, 0);
    if (p_incomp) {
      blk_incomp_fb<T, ChunkSize, ShardSize, NumThreads>(
          &s_bheader, dn_bitstream, in_eq, data_len, id_base, slot,
          (u4)psz::HFR_PBK_Constants::CodeIncompBreaks, block_outliers);
      u4 const p_wbytes =
          s_bheader.dense * (u4)sizeof(Hf) + psz::pbk_unpred_bytes((u4)s_bheader.n_unpred);
      if (threadIdx.x == 0)
        dn_headers[b] = s_bheader;  // uniform bheader output (entry don't-care)
      emit_packed_and_total<Header, Hf>(
          b, nblock, &s_bheader, p_wbytes, dn_packed_headers, d_total_cells);
      return;
    }
  }

  dispatch_smerge<'b'>(cx);
  write_pbk_bitstream_v2(
      b, s_bitcount, s_reduced, (u1*)dn_bitstream, &s_bheader, s_breaks, slot,
      _router_inline_breaks<BreakCell, u4>{}, block_outliers);
  u4 const p_wbytes = s_bheader.dense * (u4)sizeof(Hf) +
                      (u4)s_bheader.n_breaks * (u4)sizeof(BreakCell) +
                      psz::pbk_unpred_bytes((u4)s_bheader.n_unpred);
  if (threadIdx.x == 0) dn_headers[b] = s_bheader;  // uniform bheader output (entry don't-care)
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
    T* in_eq, size_t len, Hf* dram_pbk, Hf* dn_bitstream, header_t* dn_headers,
    psz::OutlierCell* block_outliers,
    uint32_t* dn_packed_headers, uint32_t* d_total_cells, uint32_t* d_state,
    int max_resident_blocks, void* stream)
{
  using C = phf::HFR_PBKGO_Config<T, Magnitude, ReduceTimes, Hf, Radius>;

  constexpr auto nthread = C::BlockDim;
  (void)max_resident_blocks;  // unused: one-shot launch now sizes grid to nblock
  const u4 nblock = (u4)((len + C::ChunkSize - 1) / C::ChunkSize);
  const dim3 grid(nblock, 1, 1);
  const dim3 block((u4)nthread, 1, 1);

  phf::KCU_HFR_PBKGO_encode<C><<<grid, block, 0, (cudaStream_t)stream>>>(
      in_eq, len, dram_pbk, dn_bitstream, dn_headers, block_outliers,
      dn_packed_headers, d_total_cells, d_state);
  return 0;
}

}  // namespace phf::module

#endif /* PHF_HFR_PBKGO_C_CUH */
