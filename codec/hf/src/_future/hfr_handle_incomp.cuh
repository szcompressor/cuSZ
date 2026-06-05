// Shared incomp-fallback writer for HFR v2 + PBK-Compat.
#ifndef PHF_HFR_HANDLE_INCOMP_CU_INL
#define PHF_HFR_HANDLE_INCOMP_CU_INL

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "hfr-pbk.hh"  // psz::HFR_PBK_Constants::CodeIncompBreaks

namespace phf::hfr_helpers {

template <
    typename T, int ChunkSize, int ShardSize, int NumThreads, typename Hf, typename Header,
    typename Slot>
__forceinline__ __device__ void handle_incomp_block(
    volatile Header* bheader, Hf* dn_bitstream, T* in_eq, size_t const data_len,
    uint32_t const id_base, Slot slot)
{
  __shared__ uint32_t s_start_off;
  __shared__ uint32_t s_wunits;

  if (threadIdx.x == 0) {
    bheader->enc_id = (uint32_t)psz::HFR_PBK_Constants::CodeIncompBreaks;
    bheader->n_breaks = 0;
    bheader->bits = ChunkSize * sizeof(T) * 8;
    s_wunits = (bheader->bits + 31u) >> 5;
  }
  __syncthreads();

  slot.reserve(blockIdx.x, s_wunits * sizeof(Hf), &s_start_off, bheader);

  auto bs_base = (Hf*)((uint8_t*)dn_bitstream + s_start_off);
  auto base = (T*)bs_base;

#pragma unroll
  for (auto ix = 0; ix < ShardSize; ix++) {
    auto l_id = threadIdx.x + ix * NumThreads;
    auto id = id_base + l_id;
    if (id < data_len) base[l_id] = in_eq[id];
  }

  slot.commit(blockIdx.x);
}

}  // namespace phf::hfr_helpers

#endif  // PHF_HFR_HANDLE_INCOMP_CU_INL
