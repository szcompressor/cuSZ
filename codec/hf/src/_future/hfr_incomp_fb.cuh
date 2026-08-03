#ifndef PHF_HFR_HANDLE_INCOMP_CU_INL
#define PHF_HFR_HANDLE_INCOMP_CU_INL

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "hfr-pbk.hh"

namespace phf::hfr_helpers {

template <
    typename T, int ChunkSize, int ShardSize, int NumThreads, typename Hf, typename Header,
    typename Slot>
__forceinline__ __device__ void blk_incomp_fb(
    volatile Header* bheader, Hf* dn_bitstream, T* in_eq, size_t const data_len, u4 const id_base,
    Slot slot, u4 const code = (u4)psz::HFR_PBK_Constants::CodeIncompBreaks,
    psz::OutlierCell const* block_outliers = nullptr)
{
  using psz::OutlierCell;
  __shared__ u4 s_start_off;
  __shared__ u4 s_wunits;    // content + unpred
  __shared__ u4 s_content;   // raw eq words (unpred and breaks alike)
  __shared__ u4 s_n_unpred;  // per-block outliers

  bool const is_unpred = (code == (u4)psz::HFR_PBK_Constants::CodeIncompUnpred);

  if (threadIdx.x == 0) {
    // append cells conditionally
    s_n_unpred = (is_unpred or not block_outliers) ? 0u : (u4)bheader->n_unpred;
    bheader->enc_id = code;
    bheader->n_unpred = s_n_unpred;
    bheader->n_breaks = 0;
    bheader->dense = (ChunkSize * sizeof(T) * 8) >> 5;
    s_content = bheader->dense;
    s_wunits = s_content + psz::pbk_unpred_words(s_n_unpred);
  }
  __syncthreads();

  slot.reserve(blockIdx.x, s_wunits * sizeof(Hf), &s_start_off, bheader);

  auto bs_base = (Hf*)((uint8_t*)dn_bitstream + s_start_off);

  // unpred: eq slot carries the predictor's incomp_pack(candidate) bits (1d: prequant, 2d/3d:
  // candidate, spline: s_recon). breaks: eq slot carries the raw quant code. Either way, a
  // straight bit copy: the predictor pre-encoded the fallback into eq's own width.
#pragma unroll
  for (auto ix = 0; ix < ShardSize; ix++) {
    auto l_id = threadIdx.x + ix * NumThreads;
    auto id = id_base + l_id;
    if (id < data_len) ((T*)bs_base)[l_id] = in_eq[id];
  }

  // breaks: append the per-block outliers after raw eq
  if (not is_unpred and block_outliers and threadIdx.x < s_n_unpred) {
    auto cell_base = (OutlierCell*)((u1*)bs_base + (size_t)s_content * sizeof(Hf));
    cell_base[threadIdx.x] =
        block_outliers[(size_t)blockIdx.x * Header::C::MaxNumUnpred + threadIdx.x];
  }

  slot.commit(blockIdx.x);
}

}  // namespace phf::hfr_helpers

#endif  // PHF_HFR_HANDLE_INCOMP_CU_INL
