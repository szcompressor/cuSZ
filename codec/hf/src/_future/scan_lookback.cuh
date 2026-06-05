#ifndef PSZ_HFR_PBK_SCAN_LOOKBACK_CU_INL
#define PSZ_HFR_PBK_SCAN_LOOKBACK_CU_INL

// Decoupled-lookback prefix scan (CUB-aligned naming).

#include "c_type.h"

namespace psz::scan_lookback {

constexpr int BLOCK_THREADS = 256;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / 32;

// CUB-aligned ScanTileStatus values.
constexpr int INVALID = 0;
constexpr int PARTIAL = 1;
constexpr int INCLUSIVE = 2;

__global__ void k_scan_init(
    volatile u4* d_partial_aggregate, volatile u4* d_incl_prefix, volatile int* d_tile_status,
    int num_tiles)
{
  int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid > num_tiles) return;

  if (tid == 0) {
    d_partial_aggregate[0] = 0u;
    d_incl_prefix[0] = 0u;
    d_tile_status[0] = INCLUSIVE;
  }
  else {
    d_partial_aggregate[tid] = 0u;
    d_incl_prefix[tid] = 0u;
    d_tile_status[tid] = INVALID;
  }
}

// Identity load: backwards-compat for callers that already have a flat u4* per-block ncell.
struct LoadFromU4 {
  u4 const* __restrict__ p;
  __device__ __forceinline__ u4 operator()(int i) const { return p[i]; }
};

template <typename Load>
__global__ void k_scan_lookback_typed(
    Load load, u4* __restrict__ par_entry, int num_items, volatile u4* d_partial_aggregate,
    volatile u4* d_incl_prefix, volatile int* d_tile_status, u4* opt_d_total)
{
  const int tile_id = (int)blockIdx.x;
  const int lane = (int)(threadIdx.x & 31);
  const int warp_id = (int)(threadIdx.x >> 5);

  __shared__ u4 s_warp_totals[WARPS_PER_BLOCK];
  __shared__ u4 s_block_aggregate;
  __shared__ u4 s_excl_prefix;

  u4 items[ITEMS_PER_THREAD];
  const int base = tile_id * TILE_SIZE + (int)threadIdx.x * ITEMS_PER_THREAD;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int idx = base + i;
    items[i] = (idx < num_items) ? load(idx) : 0u;
  }

  u4 thread_incl[ITEMS_PER_THREAD];
  u4 thread_sum = 0u;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    thread_sum += items[i];
    thread_incl[i] = thread_sum;
  }

  u4 warp_incl = thread_sum;
#pragma unroll
  for (int s = 1; s < 32; s <<= 1) {
    u4 v = __shfl_up_sync(0xffffffff, warp_incl, s);
    if (lane >= s) warp_incl += v;
  }
  u4 warp_excl = warp_incl - thread_sum;

  if (lane == 31) s_warp_totals[warp_id] = warp_incl;
  __syncthreads();

  if (warp_id == 0) {
    u4 wt = (lane < WARPS_PER_BLOCK) ? s_warp_totals[lane] : 0u;
    u4 wt_incl = wt;
#pragma unroll
    for (int s = 1; s < WARPS_PER_BLOCK; s <<= 1) {
      u4 v = __shfl_up_sync(0xffffffff, wt_incl, s);
      if (lane >= s) wt_incl += v;
    }
    u4 wt_excl = wt_incl - wt;
    if (lane < WARPS_PER_BLOCK) s_warp_totals[lane] = wt_excl;
    if (lane == WARPS_PER_BLOCK - 1) s_block_aggregate = wt_incl;
  }
  __syncthreads();

  u4 warp_block_base = s_warp_totals[warp_id];

  // thread-0-only; other threads wait at the trailing __syncthreads.
  if (threadIdx.x == 0) {
    u4 block_aggregate = s_block_aggregate;

    d_partial_aggregate[tile_id + 1] = block_aggregate;
    __threadfence();
    d_tile_status[tile_id + 1] = PARTIAL;
    __threadfence();

    u4 excl_prefix = 0u;
    int predecessor_idx = tile_id;
    while (predecessor_idx > 0) {
      int predecessor_status;
      do {
        predecessor_status = d_tile_status[predecessor_idx];
        __threadfence();
      } while (predecessor_status == INVALID);

      if (predecessor_status == INCLUSIVE) {
        excl_prefix += d_incl_prefix[predecessor_idx];
        __threadfence();
        break;
      }
      excl_prefix += d_partial_aggregate[predecessor_idx];
      predecessor_idx--;
      __threadfence();
    }

    d_incl_prefix[tile_id + 1] = excl_prefix + block_aggregate;
    __threadfence();
    d_tile_status[tile_id + 1] = INCLUSIVE;
    __threadfence();

    s_excl_prefix = excl_prefix;

    if (opt_d_total and tile_id == gridDim.x - 1) *opt_d_total = excl_prefix + block_aggregate;
  }
  __syncthreads();

  u4 const tile_excl_prefix = s_excl_prefix;
  u4 const thread_excl_in_tile = tile_excl_prefix + warp_block_base + warp_excl;

#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
    int idx = base + i;
    if (idx < num_items) {
      u4 elem_excl = thread_excl_in_tile + (thread_incl[i] - items[i]);
      par_entry[idx] = elem_excl;
    }
  }
}

template <typename Load>
inline int launch_scan_typed(
    Load load, u4* d_par_entry, int num_items, volatile u4* d_partial_aggregate,
    volatile u4* d_incl_prefix, volatile int* d_tile_status, u4* opt_d_total, cudaStream_t stream)
{
  if (num_items <= 0) return 0;
  int num_tiles = (num_items + TILE_SIZE - 1) / TILE_SIZE;

  // Caller pre-inits scan state (buf-init + post-encode reset).
  k_scan_lookback_typed<Load><<<num_tiles, BLOCK_THREADS, 0, stream>>>(
      load, d_par_entry, num_items, d_partial_aggregate, d_incl_prefix, d_tile_status,
      opt_d_total);

  return 0;
}

// Backwards-compat overload: flat u4* per-block ncell input.
inline int launch_scan(
    u4 const* d_par_ncell, u4* d_par_entry, int num_items, volatile u4* d_partial_aggregate,
    volatile u4* d_incl_prefix, volatile int* d_tile_status, u4* opt_d_total, cudaStream_t stream)
{
  return launch_scan_typed(
      LoadFromU4{d_par_ncell}, d_par_entry, num_items, d_partial_aggregate, d_incl_prefix,
      d_tile_status, opt_d_total, stream);
}

inline int launch_init(
    volatile u4* d_partial_aggregate, volatile u4* d_incl_prefix, volatile int* d_tile_status,
    int num_tiles, cudaStream_t stream)
{
  if (num_tiles <= 0) return 0;
  constexpr int INIT_BLOCK = 256;
  int g = ((num_tiles + 1) + INIT_BLOCK - 1) / INIT_BLOCK;
  k_scan_init<<<g, INIT_BLOCK, 0, stream>>>(
      d_partial_aggregate, d_incl_prefix, d_tile_status, num_tiles);
  return 0;
}

}  // namespace psz::scan_lookback

#endif
