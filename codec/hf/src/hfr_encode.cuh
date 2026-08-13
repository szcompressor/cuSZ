// HFR: single rt-book; global ordering via concat
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "_future/hfr-pbk.cuh"
#include "_future/hfr_incomp_fb.cuh"
#include "_future/warp_top1.cuh"
#include "hf_impl.hh"
#include "hfr.hh"

namespace phf {

HFR_PBK_USING_HELPERS()
using hfr_pbk::_router_inline_breaks;
using hfr_pbk::slot_fixed_stride;
using hfr_pbk::write_pbk_bitstream_v2;
using phf::hfr_helpers::blk_incomp_fb;

template <class C>
__global__ void KCU_HFR_encode(
    typename C::T* in_eq, size_t data_len, typename C::Hf* runtime_book,
    typename C::Hf* dn_bitstream, typename C::bheader_t* dn_headers,
    psz::OutlierCell* block_outliers)
{
  HFR_PBK_TYPEDEFS_AND_CONSTEXPRS(C);
  HFR_PBK_SHARED_AND_RESET();

  // fixed per-block stride
  constexpr u4 MaxBytesPerBlock = ChunkSize * (u4)sizeof(Hf) +
                                  (u4)KC::MaxNumBreaks * (u4)sizeof(BreakCell) +
                                  (u4)KC::MaxUnpredBytes;
  slot_fixed_stride slot{MaxBytesPerBlock};

  auto const id_base = (u4)blockIdx.x * ChunkSize;
  __shared__ u4 s_pre_encid;
  // keep predictor unpred + enc_id (unpred-incomp flag).
  if (threadIdx.x == 0) {
    s_bheader.n_unpred = dn_headers[blockIdx.x].n_unpred;
    s_pre_encid = dn_headers[blockIdx.x].enc_id;
  }
  __syncthreads();

  // unpred-incomp: enc_id=31, eq already carries the raw candidate bits; bypass Huffman.
  if (s_pre_encid == (u4)psz::HFR_PBK_Constants::CodeIncompUnpred) {
    blk_incomp_fb<T, ChunkSize, ShardSize, NumThreads>(
        &s_bheader, dn_bitstream, in_eq, data_len, id_base, slot,
        (u4)psz::HFR_PBK_Constants::CodeIncompUnpred);
    if (threadIdx.x == 0) dn_headers[blockIdx.x] = s_bheader;
    return;
  }

  int p_eq[ShardSize];
  hfr_pbk::load_eq_and_count_top1_v2<T, ChunkSize, ShardSize, NumThreads, Radius>(
      in_eq, data_len, id_base, p_eq, &s_top1_counts);

  // single fixed runtime book (enc_id=0) as opposed to -PBK variants
  if (threadIdx.x == 0) s_bheader.enc_id = 0;
  for (int i = threadIdx.x; i < BookLen; i += NumThreads) s_book[i] = runtime_book[i];
  __syncthreads();

  constexpr int MaxIters = (ShardSize + 1) / 2;  // >=1 so RT=0 (ShardSize=1) holds one word
  u4 r_reduced[MaxIters], r_bits[MaxIters];
  u4 reduce_times = C::ReduceTimes;  // single fixed book: never clamped
  _merge_ctx<C> cx{data_len,  (u4)blockIdx.x, reduce_times, (volatile u4*)s_book,
                   p_eq,      s_breaks,       &s_v3_incomp, &s_bheader,
                   s_reduced, s_bitcount,     r_reduced,    r_bits};
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
      if (threadIdx.x == 0) dn_headers[blockIdx.x] = s_bheader;
      return;
    }
  }

  dispatch_smerge<'b'>(cx);
  write_pbk_bitstream_v2(
      blockIdx.x, s_bitcount, s_reduced, (u1*)dn_bitstream, &s_bheader, s_breaks, slot,
      _router_inline_breaks<BreakCell, u4>{}, block_outliers);
  if (threadIdx.x == 0) dn_headers[blockIdx.x] = s_bheader;
}

}  // namespace phf

namespace phf::module {

template <typename T, int Magnitude, int ReduceTimes, bool UseScan, typename Hf>
int HFR_encoder<T, Magnitude, ReduceTimes, UseScan, Hf>::GPU_kernel_v2(
    T* in_eq, size_t len, Hf* runtime_book, Hf* dn_bitstream, bheader_t* dn_headers,
    psz::OutlierCell* block_outliers, void* stream)
{
  using C = phf::HFR_PBKC_Config<T, Magnitude, ReduceTimes, Hf>;

  constexpr auto nthread = C::BlockDim;
  const auto nblock = (u4)((len - 1) / C::ChunkSize + 1);

  phf::KCU_HFR_encode<C><<<nblock, nthread, 0, (cudaStream_t)stream>>>(
      in_eq, len, runtime_book, dn_bitstream, dn_headers, block_outliers);

  return 0;
}

}  // namespace phf::module
