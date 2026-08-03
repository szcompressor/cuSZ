#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "_future/scan_lookback.cuh"
#include "_future/scan_lookback.hh"
#include "hfr.hh"

using u1 = uint8_t;
using u4 = uint32_t;

namespace phf {

// pass-2: fully-parallel scatter. Each block copies its slice.
template <int BlockDim>
__global__ void KCU_concat_via_scatter_packed(
    u4 const* __restrict__ par_ncell, u4 const* __restrict__ par_entry,
    u4 const* __restrict__ dn_in, u4* __restrict__ dn_out, u4 ChunkSize, int pardeg)
{
  const int b = (int)blockIdx.x;
  if (b >= pardeg) return;

  __shared__ u4 s_ncell;
  __shared__ u4 s_entry;
  if (threadIdx.x == 0) {
    s_ncell = par_ncell[b];
    s_entry = par_entry[b];
  }
  __syncthreads();

  u4 const ncell = s_ncell;
  u4 const src_base = (u4)b * ChunkSize;
  u4 const dst_base = s_entry;
  for (u4 i = threadIdx.x; i < ncell; i += BlockDim) dn_out[dst_base + i] = dn_in[src_base + i];
}

// scatter + inline 2-word header/block.
template <typename E, int BlockDim, int Magnitude = 10>
__global__ void KCU_concat_via_scatter(
    psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius, (size_t)Magnitude> const* __restrict__
        bheaders,
    u4 const* __restrict__ par_entry, u4 const* __restrict__ dn_in, u4* __restrict__ dn_out,
    u4* __restrict__ out_headers, u4 sizeof_Hf, u4 ChunkSize, int pardeg)
{
  const int b = (int)blockIdx.x;
  if (b >= pardeg) return;

  using KC = psz::_parameterized_hfr_pbk_constants<(size_t)Magnitude>;
  constexpr u4 EncIdShift = (u4)(KC::BitsMaxNumUnpred + KC::BitsMaxNumBreaks);
  constexpr u4 DenseShift = EncIdShift + (u4)KC::BitsEncId;
  constexpr u4 UnpredMask = (1u << KC::BitsMaxNumUnpred) - 1u;

  __shared__ u4 s_ncell;
  __shared__ u4 s_entry;
  if (threadIdx.x == 0) {
    auto const h = bheaders[b];
    u4 const dense = h.dense;  // words
    u4 const n_breaks = (u4)h.n_breaks;
    u4 const enc_id = (u4)h.enc_id;
    s_ncell = dense + n_breaks + psz::pbk_unpred_words((u4)h.n_unpred);
    u4 const entry = par_entry[b];
    s_entry = entry;
    // Emit 2-word header inline (n_unpred in the low bits).
    out_headers[2 * b + 0] =
        (dense << DenseShift) | (enc_id << EncIdShift) | ((u4)h.n_unpred & UnpredMask);
    out_headers[2 * b + 1] = entry * sizeof_Hf;
  }
  __syncthreads();

  u4 const ncell = s_ncell;
  u4 const src_base = (u4)b * ChunkSize;
  u4 const dst_base = s_entry;
  for (u4 i = threadIdx.x; i < ncell; i += BlockDim) dn_out[dst_base + i] = dn_in[src_base + i];
}

// load per-block ncell from bheader[i].{dense, n_breaks}.
template <typename E, int Magnitude = 10>
struct LoadNcellFromBheader {
  psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius, (size_t)Magnitude> const* __restrict__ p;
  __device__ __forceinline__ u4 operator()(int i) const
  {
    auto const h = p[i];
    return h.dense + (u4)h.n_breaks + psz::pbk_unpred_words((u4)h.n_unpred);
  }
};

using bheader_hfr2 = psz::_future::bheader<u4, psz::HFR_PBK_Constants::Radius>;

__global__ void KCU_pack_bheader_backport(  // HF-rev2: par_nbit/par_entry -> future-bheader AoS
    u4 const* __restrict__ par_nbit, u4 const* __restrict__ par_entry,
    u4* __restrict__ out_headers, int pardeg, u4 sizeof_Hf)
{
  const int b = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (b >= pardeg) return;
  bheader_hfr2 bh{};
  bh.dense = par_nbit[b];  // HF-rev2: bit count (see bheader field note)
  bh.entry = par_entry[b] * sizeof_Hf;
  reinterpret_cast<bheader_hfr2*>(out_headers)[b] = bh;
}

__global__ void KCU_unpack_bheader_backport(  // HF-rev2: future-bheader AoS -> par_nbit/par_entry
    u4 const* __restrict__ in_headers, u4* __restrict__ par_nbit, u4* __restrict__ par_entry,
    int pardeg, u4 sizeof_Hf)
{
  const int b = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (b >= pardeg) return;
  auto const bh = reinterpret_cast<bheader_hfr2 const*>(in_headers)[b];
  par_nbit[b] = bh.dense;
  par_entry[b] = bh.entry / sizeof_Hf;
}

}  // namespace phf

namespace phf::module {

int pack_bheader_backport::GPU_kernel(
    uint32_t const* par_nbit, uint32_t const* par_entry, uint32_t* out_headers, int pardeg,
    int sizeof_Hf, void* stream)
{
  if (pardeg <= 0) return 0;
  constexpr int BlockDim = 256;
  dim3 grid((unsigned)((pardeg + BlockDim - 1) / BlockDim), 1, 1);
  dim3 block(BlockDim, 1, 1);
  phf::KCU_pack_bheader_backport<<<grid, block, 0, (cudaStream_t)stream>>>(
      par_nbit, par_entry, out_headers, pardeg, (u4)sizeof_Hf);
  return 0;
}

int unpack_bheader_backport::GPU_kernel(
    uint32_t const* in_headers, uint32_t* par_nbit, uint32_t* par_entry, int pardeg, int sizeof_Hf,
    void* stream)
{
  if (pardeg <= 0) return 0;
  constexpr int BlockDim = 256;
  dim3 grid((unsigned)((pardeg + BlockDim - 1) / BlockDim), 1, 1);
  dim3 block(BlockDim, 1, 1);
  phf::KCU_unpack_bheader_backport<<<grid, block, 0, (cudaStream_t)stream>>>(
      in_headers, par_nbit, par_entry, pardeg, (u4)sizeof_Hf);
  return 0;
}

}  // namespace phf::module

namespace phf {

template <int BlockDim>
int concat_via_scatter_ppc<BlockDim>::GPU_kernel(
    u4 const* par_ncell, u4* par_entry, u4 const* dn_in, u4* dn_out, u4 ChunkSize, int pardeg,
    u4* scan_partial_aggregate, u4* scan_incl_prefix, int* scan_tile_status, u4* opt_d_total_words,
    void* stream)
{
  if (pardeg <= 0) return 0;
  auto cstream = (cudaStream_t)stream;

  // pass-1: decoupled-lookback scan (state pre-init'd by hf_buf).
  psz::scan_lookback::launch_scan(
      par_ncell, par_entry, pardeg, scan_partial_aggregate, scan_incl_prefix, scan_tile_status,
      opt_d_total_words, cstream);

  // pass-2: scatter.
  dim3 grid2((unsigned)pardeg, 1, 1);
  dim3 block2((unsigned)BlockDim, 1, 1);
  phf::KCU_concat_via_scatter_packed<BlockDim>
      <<<grid2, block2, 0, cstream>>>(par_ncell, par_entry, dn_in, dn_out, ChunkSize, pardeg);

  return 0;
}

template <typename E, int BlockDim, int Magnitude>
int _future_concat_via_scatter<E, BlockDim, Magnitude>::GPU_kernel(
    bheader_t const* bheaders, u4* par_entry, u4 const* dn_in, u4* dn_out, u4* out_packed_headers,
    u4 sizeof_Hf, u4 ChunkSize, int pardeg, u4* scan_partial_aggregate, u4* scan_incl_prefix,
    int* scan_tile_status, u4* opt_d_total_words, void* stream)
{
  if (pardeg <= 0) return 0;
  auto cstream = (cudaStream_t)stream;

  // pass-1: scan reads ncell
  psz::scan_lookback::launch_scan_typed(
      phf::LoadNcellFromBheader<E, Magnitude>{bheaders}, par_entry, pardeg,
      scan_partial_aggregate, scan_incl_prefix, scan_tile_status, opt_d_total_words, cstream);

  // pass-2: fused scatter
  dim3 grid2((unsigned)pardeg, 1, 1);
  dim3 block2((unsigned)BlockDim, 1, 1);
  phf::KCU_concat_via_scatter<E, BlockDim, Magnitude><<<grid2, block2, 0, cstream>>>(
      bheaders, par_entry, dn_in, dn_out, out_packed_headers, sizeof_Hf, ChunkSize, pardeg);

  return 0;
}

}  // namespace phf

// Host wrapper for the scan init, callable from non-CUDA TUs (e.g., hf_buf.cc).
namespace psz::scan_lookback {
void launch_init_host(
    uint32_t* d_partial_aggregate, uint32_t* d_incl_prefix, int* d_tile_status, int num_tiles,
    void* stream)
{
  launch_init(d_partial_aggregate, d_incl_prefix, d_tile_status, num_tiles, (cudaStream_t)stream);
}
}  // namespace psz::scan_lookback
