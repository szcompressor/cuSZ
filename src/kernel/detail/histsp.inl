/**
 * @file hist_sp.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-05-18
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

//                    -2 -1  0 +1 +2
//                     |<-R->|<-R->|
// p_hist              |<----K---->|       K=2R+1
// s_hist |<-----offset------|------offset------>|
//
// Multiple warps:the large shmem use (relative to #thread).

template <
    typename T, int NWARP = 4, int CHUNK = 16384, typename FQ = uint32_t,
    int OUTLEN = 1024, int K = 5, bool COMPLETE = true>
__global__ void histsp_multiwarp(
    T* in, uint32_t inlen, FQ* out, uint32_t outlen, int offset = 0)
{
  static_assert(K % 2 == 1, "K must be odd.");
  constexpr auto R = (K - 1) / 2;

  // small & big local hist based on presumed access category
  __shared__ FQ s_hist[OUTLEN];
  // cannot scale according to compressiblity becasuse of the register pressure
  // there should be offline optimal
  FQ p_hist[K] = {0};

  auto __gidx = [&](auto tix) { return blockIdx.x * CHUNK + tix; };
  constexpr auto BLOCK_DIM = 32 * NWARP;

  // accessor

  for (auto i = threadIdx.x; i < CHUNK; i += BLOCK_DIM) {
    auto gidx = __gidx(i);
    if (gidx < inlen) {
      auto sym = (int)in[gidx] - offset;

      if (2 * abs(sym) + 1 <= K) {
        p_hist[sym + R] += 1;  // more possible
      }
      else {
        atomicAdd(&s_hist[sym + offset], 1);  // less possible
      }
    }
  }
  __syncthreads();

  // fix-up
  for (auto& sum : p_hist) {
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
  }

  // should test if faster than atomicAdd
  if (threadIdx.x % 32 == 0)
    for (auto i = 0; i < K; i++) atomicAdd(&s_hist[i - R + offset], p_hist[i]);
  __syncthreads();

  for (auto tix = threadIdx.x; tix < OUTLEN; tix += BLOCK_DIM)
    atomicAdd(out + tix, s_hist[tix]);
}