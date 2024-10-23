/**
 * @file hist_sp.cuhip.inl
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

namespace psz {

//                    -2 -1  0 +1 +2
//                     |<-R->|<-R->|
// p_hist              |<----K---->|       K=2R+1
// s_hist |<-----offset------|------offset------>|
//
// Multiple warps:the large shmem use (relative to #thread).

template <typename T, typename FREQ = uint32_t, int K = 5>
__global__ void KERNEL_CUHIP_histogram_sparse_multiwarp(
    T* in_data, size_t const data_len, FREQ* out_hist, uint16_t const hist_len,
    uint32_t const chunk, int offset = 0)
{
  static_assert(K % 2 == 1, "K must be odd.");
  constexpr auto R = (K - 1) / 2;  // K = 5, R = 2

  // small & big local hist based on presumed access category
  extern __shared__ FREQ s_hist[];
  // cannot scale according to compressiblity becasuse of the register pressure
  // there should be offline optimal
  FREQ p_hist[K] = {0};

  auto global_id = [&](auto i) { return blockIdx.x * chunk + i; };
  auto nworker = [&]() { return blockDim.x; };

  for (auto i = threadIdx.x; i < hist_len; i += blockDim.x) s_hist[i] = 0;
  __syncthreads();

  for (auto i = threadIdx.x; i < chunk; i += blockDim.x) {
    auto gid = global_id(i);
    if (gid < data_len) {
      auto ori = (int)in_data[gid];  // e.g., 512
      auto sym = ori - offset;       // e.g., 512 - 512 = 0

      if (2 * abs(sym) < K) {
        // -2, -1, 0, 1, 2 -> sym + R = 0, 1, 2, 3, 4
        //  4   2  0  2  4 <- 2 * abs(sym)
        p_hist[sym + R] += 1;  // more possible
      }
      else {
        // resume the original input
        atomicAdd(&s_hist[ori], 1);  // less possible
      }
    }
    __syncthreads();
  }
  __syncthreads();

#ifdef __HIP_PLATFORM_AMD__
  for (auto& sum : p_hist) {
    for (auto d = 1; d < 64; d *= 2) {
      auto n = __shfl_up(sum, d, 64);
      if (threadIdx.x % 64 >= d) sum += n;
    }
  }
#else
  for (auto& sum : p_hist) {
    for (auto d = 1; d < 32; d *= 2) {
      auto n = __shfl_up_sync(0xffffffff, sum, d);
      if (threadIdx.x % 32 >= d) sum += n;
    }
  }
#endif

  for (auto i = 0; i < K; i++)
    if (threadIdx.x % 32 == 31)
      atomicAdd(&s_hist[(int)offset + i - R], p_hist[i]);
  __syncthreads();

  for (auto i = threadIdx.x; i < hist_len; i += blockDim.x)
    atomicAdd(out_hist + i, s_hist[i]);
  __syncthreads();
}

}  // namespace psz

#include "utils/timer.hh"

namespace psz::cuhip {

template <typename T, typename FREQ>
int GPU_histogram_sparse(
    T* in_data, size_t const data_len, FREQ* out_hist, uint16_t const hist_len,
    float* milliseconds, cudaStream_t stream)
{
  auto chunk = 32768;
  auto num_chunks = (data_len - 1) / chunk + 1;
  auto num_workers = 256;  // n SIMD-32

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  psz::KERNEL_CUHIP_histogram_sparse_multiwarp<T, FREQ>
      <<<num_chunks, num_workers, sizeof(FREQ) * hist_len, stream>>>(
          in_data, data_len, out_hist, hist_len, chunk, hist_len / 2);
  STOP_GPUEVENT_RECORDING(stream);

  cudaStreamSynchronize(stream);
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;

  return 0;
}

}  // namespace psz::cuhip