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

// Ported from bleeding-edge: scatter + inline 2-word header emit per block.
template <typename E, int BlockDim>
__global__ void KCU_concat_via_scatter(
    psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius> const* __restrict__ bheaders,
    u4 const* __restrict__ par_entry, u4 const* __restrict__ dn_in, u4* __restrict__ dn_out,
    u4* __restrict__ out_headers, u4 sizeof_Hf, u4 ChunkSize, int pardeg)
{
  const int b = (int)blockIdx.x;
  if (b >= pardeg) return;

  __shared__ u4 s_ncell;
  __shared__ u4 s_entry;
  if (threadIdx.x == 0) {
    auto const h = bheaders[b];
    u4 const bits = h.bits;
    u4 const n_breaks = (u4)h.n_breaks;
    u4 const enc_id = (u4)h.enc_id;
    // ncell (4-byte cells) = ceil(bits/32) + n_breaks (BreakCell = 1 word for R≤128).
    s_ncell = (bits + 31u) / 32u + n_breaks;
    u4 const entry = par_entry[b];
    s_entry = entry;
    // Emit 2-word header inline.
    out_headers[2 * b + 0] = (bits << 14) | (enc_id << 9);
    out_headers[2 * b + 1] = entry * sizeof_Hf;
  }
  __syncthreads();

  u4 const ncell = s_ncell;
  u4 const src_base = (u4)b * ChunkSize;
  u4 const dst_base = s_entry;
  for (u4 i = threadIdx.x; i < ncell; i += BlockDim) dn_out[dst_base + i] = dn_in[src_base + i];
}

// Scan input adapter: load per-block ncell from bheader[i].{bits, n_breaks}.
template <typename E>
struct LoadNcellFromBheader {
  psz::_future::bheader<E, psz::HFR_PBK_Constants::Radius> const* __restrict__ p;
  __device__ __forceinline__ u4 operator()(int i) const
  {
    auto const h = p[i];
    return (h.bits + 31u) / 32u + (u4)h.n_breaks;
  }
};

// Sum par_nbit[0..pardeg) -> *total_nbit via per-block atomics.
template <int BlockDim>
__global__ void KCU_reduce_total_nbit(u4 const* par_nbit, u4 pardeg, u4* total_nbit)
{
  __shared__ u4 s_partial;
  if (threadIdx.x == 0) s_partial = 0;
  __syncthreads();
  u4 local = 0;
  for (u4 i = blockIdx.x * BlockDim + threadIdx.x; i < pardeg; i += BlockDim * gridDim.x)
    local += par_nbit[i];
  atomicAdd(&s_partial, local);
  __syncthreads();
  if (threadIdx.x == 0 and s_partial > 0) atomicAdd(total_nbit, s_partial);
}

// HF_rev2: pack (par_nbit, par_entry) -> AoS bheader_backport[] (bits:32 + entry:32).
__global__ void KCU_pack_bheader_backport(
    u4 const* __restrict__ par_nbit, u4 const* __restrict__ par_entry,
    u4* __restrict__ out_headers, int pardeg, u4 sizeof_Hf)
{
  const int b = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (b >= pardeg) return;
  out_headers[2 * b + 0] = par_nbit[b];
  out_headers[2 * b + 1] = par_entry[b] * sizeof_Hf;
}

// HF_rev2: unpack AoS bheader_backport[] -> (par_nbit, par_entry) on decode.
__global__ void KCU_unpack_bheader_backport(
    u4 const* __restrict__ in_headers, u4* __restrict__ par_nbit, u4* __restrict__ par_entry,
    int pardeg, u4 sizeof_Hf)
{
  const int b = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (b >= pardeg) return;
  par_nbit[b] = in_headers[2 * b + 0];
  par_entry[b] = in_headers[2 * b + 1] / sizeof_Hf;
}

}  // namespace phf

namespace phf::module {

int reduce_total_nbit::GPU_kernel(u4 const* par_nbit, u4 pardeg, u4* total_nbit, void* stream)
{
  if (pardeg == 0) return 0;
  constexpr int BlockDim = 256;
  // Cap grid: enough to oversubscribe but not waste — atomic adds dominate.
  int grid = (int)((pardeg + BlockDim - 1) / BlockDim);
  if (grid > 32) grid = 32;
  phf::KCU_reduce_total_nbit<BlockDim>
      <<<grid, BlockDim, 0, (cudaStream_t)stream>>>(par_nbit, pardeg, total_nbit);
  return 0;
}

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

template <typename E, int BlockDim>
int _future_concat_via_scatter<E, BlockDim>::GPU_kernel(
    bheader_t const* bheaders, u4* par_entry, u4 const* dn_in, u4* dn_out, u4* out_packed_headers,
    u4 sizeof_Hf, u4 ChunkSize, int pardeg, u4* scan_partial_aggregate, u4* scan_incl_prefix,
    int* scan_tile_status, u4* opt_d_total_words, void* stream)
{
  if (pardeg <= 0) return 0;
  auto cstream = (cudaStream_t)stream;

  // pass-1: scan reads ncell directly from bheader (no separate par_ncell buffer).
  psz::scan_lookback::launch_scan_typed(
      phf::LoadNcellFromBheader<E>{bheaders}, par_entry, pardeg, scan_partial_aggregate,
      scan_incl_prefix, scan_tile_status, opt_d_total_words, cstream);

  // pass-2: fused scatter — writes 2-word header + scatters payload.
  dim3 grid2((unsigned)pardeg, 1, 1);
  dim3 block2((unsigned)BlockDim, 1, 1);
  phf::KCU_concat_via_scatter<E, BlockDim><<<grid2, block2, 0, cstream>>>(
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

#define __INSTANTIATE_PHF_CONCAT_VIA_SCATTER_PPC(BD) \
  template struct phf::concat_via_scatter_ppc<BD>;
#define __INSTANTIATE_PHF_FUTURE_CONCAT_VIA_SCATTER(E, BD) \
  template struct phf::_future_concat_via_scatter<E, BD>;
