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

//                 -2 -1  0 +1 +2
//                  |<-R->|<-R->|
// top              |<----K---->|       K=2R+1
// h   |<-----offset------|------offset------>|

/**
 * @deprecated only performance demonstration
 *
 */
template <
    typename T, int CHUNK = 16384, typename FQ = uint32_t, int OUTLEN = 1024,
    int K = 5>
__global__ void histsp_1warp(
    T* in, uint32_t inlen, FQ* out, uint32_t outlen, int offset = 0)
{
  static_assert(K % 2 == 1, "K must be odd.");
  constexpr auto R = (K - 1) / 2;

  // small & big local hist based on presumed access category
  __shared__ FQ h[OUTLEN];
  // cannot scale according to compressiblity becasuse of the register pressure
  // there should be offline optimal
  FQ top[K];
  for (auto& t : top) t = 0;

  auto gidx = [&](auto tix) { return blockIdx.x * CHUNK + tix; };
  constexpr auto BLOCK_DIM = 32;

  // accessor

  for (auto tix = 0; tix < CHUNK; tix += BLOCK_DIM) {
    auto sym = (int)in[gidx(tix)] - offset;

    if (sym != 0) {  // less possible than 0
      if (2 * abs(sym) + 1 <= K)
        top[sym + R] += 1;  // more possible
      else
        atomicAdd(&h[sym + offset], 1);  // less possible
    }
  }

  // fix-up
  for (auto& sum : top) {
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
  }

  // should test if faster than atomicAdd
  if (threadIdx.x == 0)
    for (auto i = 0; i < K; i++) h[i - R + offset] = top[i - R];

  for (auto tix = 0; tix < OUTLEN; tix += BLOCK_DIM)
    atomicAdd(out + tix, h[tix]);
}

template <
    typename T, int NWARP = 4, int CHUNK = 16384, typename FQ = uint32_t,
    int OUTLEN = 1024, int K = 5>
__global__ void histsp_multiwarp(
    T* in, uint32_t inlen, FQ* out, uint32_t outlen, int offset = 0)
{
  static_assert(K % 2 == 1, "K must be odd.");
  constexpr auto R = (K - 1) / 2;

  // small & big local hist based on presumed access category
  __shared__ FQ h[OUTLEN];
  // cannot scale according to compressiblity becasuse of the register pressure
  // there should be offline optimal
  FQ top[K];
  for (auto& t : top) t = 0;

  auto gidx = [&](auto tix) { return blockIdx.x * CHUNK + tix; };
  constexpr auto BLOCK_DIM = 32 * NWARP;

  // accessor

  for (auto tix = 0; tix < CHUNK; tix += BLOCK_DIM) {
    auto sym = (int)in[gidx(tix)] - offset;

    if (sym != 0) {  // less possible than 0
      if (2 * abs(sym) + 1 <= K)
        top[sym + R] += 1;  // more possible
      else
        atomicAdd(&h[sym + offset], 1);  // less possible
    }
  }

  // fix-up
  for (auto& sum : top) {
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
  }

  // should test if faster than atomicAdd
  if (threadIdx.x % 32 == 0)
    for (auto i = 0; i < K; i++) atomicAdd(&h[i - R + offset], top[i - R]);
  __syncthreads();

  for (auto tix = 0; tix < OUTLEN; tix += BLOCK_DIM)
    atomicAdd(out + tix, h[tix]);
}