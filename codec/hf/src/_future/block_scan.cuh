#ifndef PHF_BLOCK_SCAN_INL
#define PHF_BLOCK_SCAN_INL

#include <cstdint>

namespace phf::block_scan {

__device__ __forceinline__ uint32_t warp_incl_scan_u32(uint32_t x)
{
#pragma unroll
  for (int s = 1; s < 32; s <<= 1) {
    uint32_t y = __shfl_up_sync(0xffffffffu, x, s);
    if ((threadIdx.x & 31) >= s) x += y;
  }
  return x;
}

// NumThreads <= 32*32 (two rounds)
// block_total = s_warp_totals[NumWarps-1]
template <int NumThreads>
__device__ __forceinline__ uint32_t block_incl_scan_u32(uint32_t x, uint32_t* s_warp_totals)
{
  static_assert(
      NumThreads > 0 and NumThreads <= 1024 and (NumThreads % 32) == 0,
      "NumThreads must be in (0, 1024] and a multiple of 32.");
  constexpr int NumWarps = NumThreads / 32;
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  // for all warps
  uint32_t warp_prefix = warp_incl_scan_u32(x);
  if (lane == 31) s_warp_totals[warp_id] = warp_prefix;
  __syncthreads();

  // for one warp 0
  if (warp_id == 0) {
    uint32_t v = (lane < NumWarps) ? s_warp_totals[lane] : 0u;
    v = warp_incl_scan_u32(v);
    if (lane < NumWarps) s_warp_totals[lane] = v;
  }
  __syncthreads();

  // propagate to all elements
  uint32_t prior = (warp_id > 0) ? s_warp_totals[warp_id - 1] : 0u;
  return prior + warp_prefix;
}

}  // namespace phf::block_scan

#endif  // PHF_BLOCK_SCAN_INL
