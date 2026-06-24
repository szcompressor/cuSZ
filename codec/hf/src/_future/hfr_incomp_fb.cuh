#ifndef PHF_HFR_HANDLE_INCOMP_CU_INL
#define PHF_HFR_HANDLE_INCOMP_CU_INL

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "_future/incomp_redo.cuh"
#include "hfr-pbk.hh"

namespace phf::hfr_helpers {

template <
    typename T, int ChunkSize, int ShardSize, int NumThreads, typename Hf, typename Header,
    typename Slot>
__forceinline__ __device__ void blk_incomp_fb(
    volatile Header* bheader, Hf* dn_bitstream, T* in_eq, size_t const data_len,
    uint32_t const id_base, Slot slot,
    uint32_t const code = (uint32_t)psz::HFR_PBK_Constants::CodeIncompBreaks,
    f4 const* incomp_data = nullptr, _ptb::compact_cell<f4, u2> const* block_outliers = nullptr,
    IncompRedo rc = {})
{
  using OutlierCell = _ptb::compact_cell<f4, u2>;
  __shared__ uint32_t s_start_off;
  __shared__ uint32_t s_wunits;    // content + unpred
  __shared__ uint32_t s_content;   // f4 for unpred, raw eq for breaks
  __shared__ uint32_t s_n_unpred;  // per-block outliers

  bool const is_unpred = (code == (uint32_t)psz::HFR_PBK_Constants::CodeIncompUnpred);

  if (threadIdx.x == 0) {
    // append cells conditionally
    s_n_unpred = (is_unpred or not block_outliers) ? 0u : (uint32_t)bheader->n_unpred;
    bheader->enc_id = code;
    bheader->n_unpred = s_n_unpred;
    bheader->n_breaks = 0;
    bheader->bits = ChunkSize * (is_unpred ? (uint32_t)sizeof(f4) : (uint32_t)sizeof(T)) * 8;
    s_content = (bheader->bits + 31u) >> 5;
    s_wunits = s_content + psz::pbk_unpred_words(s_n_unpred);
  }
  __syncthreads();

  slot.reserve(blockIdx.x, s_wunits * sizeof(Hf), &s_start_off, bheader);

  auto bs_base = (Hf*)((uint8_t*)dn_bitstream + s_start_off);

#pragma unroll
  for (auto ix = 0; ix < ShardSize; ix++) {
    auto l_id = threadIdx.x + ix * NumThreads;
    auto id = id_base + l_id;
    if (id < data_len) {
      if (is_unpred) {
        // tile-order (2D): the chunk is a 32x32 tile, so map the in-tile offset
        // back to the linear gid before recomputing the delta from in_data.
        size_t gid = id;
        bool in_range = true;
        if (rc.nd_tile and rc.kind == IncompPredKind::Lorenzo2D) {
          u4 tiles_x = ((u4)rc.leapy + 31u) / 32u;
          u4 bx = blockIdx.x % tiles_x, by = blockIdx.x / tiles_x;
          u4 gix = bx * 32u + (l_id % 32u), giy = by * 32u + (l_id / 32u);
          in_range = (gix < rc.leapy and giy < rc.dimy);  // partial boundary tile
          gid = (size_t)giy * rc.leapy + gix;
        }
        else if (rc.nd_tile and rc.kind == IncompPredKind::Lorenzo3D) {
          // 32x8x8 tile == two 1Ki chunks; HF block = 2*tile + half.
          u4 tile = blockIdx.x >> 1, half = blockIdx.x & 1u;
          u4 off = half * 1024u + l_id;  // in-tile offset 0..2047 (z*256 + y*32 + x)
          u4 lx = off % 32u, ly = (off % 256u) / 32u, lz = off / 256u;
          u4 tiles_x = ((u4)rc.leapy + 31u) / 32u, tiles_y = (rc.dimy + 7u) / 8u;
          u4 bx = tile % tiles_x, by = (tile / tiles_x) % tiles_y, bz = tile / (tiles_x * tiles_y);
          u4 gix = bx * 32u + lx, giy = by * 8u + ly, giz = bz * 8u + lz;
          in_range = (gix < rc.leapy and giy < rc.dimy and giz < rc.dimz);
          gid = (size_t)giz * rc.leapz + (size_t)giy * rc.leapy + gix;
        }
        ((f4*)bs_base)[l_id] = not in_range ? (f4)0
                               : (rc.kind != IncompPredKind::None)
                                   ? psz::incomp_redo::dispatch(rc, gid)
                                   : incomp_data[gid];
      }
      else
        ((T*)bs_base)[l_id] = in_eq[id];
    }
  }

  // breaks: append the per-block outliers after raw eq
  if (not is_unpred and block_outliers and threadIdx.x < s_n_unpred) {
    auto cell_base = (OutlierCell*)((u1*)bs_base + (size_t)s_content * sizeof(Hf));
    cell_base[threadIdx.x] =
        block_outliers[(size_t)blockIdx.x * psz::HFR_PBK_Constants::MaxNumUnpred + threadIdx.x];
  }

  slot.commit(blockIdx.x);
}

}  // namespace phf::hfr_helpers

#endif  // PHF_HFR_HANDLE_INCOMP_CU_INL
