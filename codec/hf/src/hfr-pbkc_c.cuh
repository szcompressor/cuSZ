// HFR-PBK_Compat: blockwise, otherwise compat with HFR; global ordering via concat
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "_future/hfr-pbk.cuh"
#include "_future/hfr_handle_incomp.cuh"
#include "_future/warp_top1.cuh"
#include "hf_impl.hh"
#include "hfr.hh"

namespace phf {

HFR_PBK_USING_HELPERS()
using hfr_pbk::_router_inline_breaks;
using hfr_pbk::slot_fixed_stride;
using hfr_pbk::write_pbk_bitstream_v2;
using phf::hfr_helpers::handle_incomp_block;

template <class C, RMerge RM = RMerge::v7, SMerge SM = SMerge::v7>
__global__ void KCU_HFR_PBKC_encode(
    typename C::T* in_eq, size_t data_len, typename C::Hf* dram_pbk, typename C::Hf* dn_bitstream,
    psz::_future::bheader<typename C::T, C::Radius>* dn_headers)
{
  static_assert(merge_compatible(RM, SM), "RMerge/SMerge data-handoff contract mismatch");
  HFR_PBK_TYPEDEFS_AND_CONSTEXPRS(C);
  HFR_PBK_SHARED_AND_RESET();

  constexpr u4 MaxBytesPerBlock = ChunkSize * (u4)sizeof(Hf) +
                                  (u4)psz::HFR_PBK_Constants::MaxNumBreaks * (u4)sizeof(BreakCell);
  slot_fixed_stride slot{MaxBytesPerBlock};

  auto const id_base = (u4)blockIdx.x * ChunkSize;

  int p_eq[ShardSize];
  hfr_pbk::load_eq_and_count_top1_v2<T, ChunkSize, ShardSize, NumThreads, Radius>(
      in_eq, data_len, id_base, p_eq, &s_top1_counts);

  u4 reduce_times = C::ReduceTimes;
  find_proper_book<ChunkSize, NumBooks, Header>(&s_top1_counts, &s_bheader, data_len, blockIdx.x);
  load_proper_book<BookLen, Header>(reduce_times, (volatile u4*)s_book, dram_pbk, &s_bheader);

  constexpr int MaxIters = ShardSize / 2;
  u4 r_reduced[MaxIters], r_bits[MaxIters];
  MergeCtx<C> cx{data_len,  (u4)blockIdx.x, reduce_times, (volatile u4*)s_book,
                 p_eq,      s_breaks,       &s_v3_incomp, &s_bheader,
                 s_reduced, s_bitcount,     r_reduced,    r_bits};
  dispatch_rmerge<RM>(cx);

  {
    u4 p_incomp = 0;
    if ((threadIdx.x & 31) == 0) p_incomp = s_v3_incomp & psz::HFR_PBK_Constants::MASK_TF;
    __syncthreads();
    p_incomp = __shfl_sync(0xffffffff, p_incomp, 0);
    if (p_incomp) {
      handle_incomp_block<T, ChunkSize, ShardSize, NumThreads>(
          &s_bheader, dn_bitstream, in_eq, data_len, id_base, slot);
      if (threadIdx.x == 0) dn_headers[blockIdx.x] = s_bheader;
      return;
    }
  }

  dispatch_smerge<SM>(cx);
  write_pbk_bitstream_v2(
      blockIdx.x, s_bitcount, s_reduced, (u1*)dn_bitstream, &s_bheader, s_breaks, slot,
      _router_inline_breaks<BreakCell, u4>{});
  if (threadIdx.x == 0) dn_headers[blockIdx.x] = s_bheader;
}

}  // namespace phf

namespace phf::module {

template <typename T, int Magnitude, int ReduceTimes, typename Hf, uint16_t Radius>
int HFR_PBKC_encode<T, Magnitude, ReduceTimes, Hf, Radius>::GPU_kernel(
    T* in_eq, size_t len, Hf* dram_pbk, Hf* dn_bitstream, header_t* dn_headers, void* stream,
    RMerge rm, SMerge sm)
{
  using C = phf::HFR_PBKC_Config<T, Magnitude, ReduceTimes, Hf, Radius>;

  constexpr auto nthread = C::BlockDim;
  const auto nblock = (u4)((len - 1) / C::ChunkSize + 1);

  dispatch_merge_host(rm, sm, [&](auto rm_tag, auto sm_tag) {
    constexpr RMerge RM = decltype(rm_tag)::value;
    constexpr SMerge SM = decltype(sm_tag)::value;
    phf::KCU_HFR_PBKC_encode<C, RM, SM><<<nblock, nthread, 0, (cudaStream_t)stream>>>(
        in_eq, len, dram_pbk, dn_bitstream, dn_headers);
  });

  return 0;
}

}  // namespace phf::module

// Instantiation macros — caller TUs invoke; this .inl alone instantiates nothing.
#define __INSTANTIATE_HFR_PBK_COMPAT(T, MAG, RED, RAD) \
  template struct phf::module::HFR_PBKC_encode<T, MAG, RED, uint32_t, RAD>;

// 1-arg form: fan out u1/u2 at canonical MAG=10, RAD=128 (mirrors __INSTANTIATE_RSMERGE_1).
#define __INSTANTIATE_HFR_PBKC_1(RED)                 \
  __INSTANTIATE_HFR_PBK_COMPAT(uint8_t, 10, RED, 128) \
  __INSTANTIATE_HFR_PBK_COMPAT(uint16_t, 10, RED, 128)
