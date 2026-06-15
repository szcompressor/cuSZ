#ifndef _PSZ_UTILS_EXTREMA_CUH
#define _PSZ_UTILS_EXTREMA_CUH

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#include "extrema.hh"
#include "mem/atomics.cuh"
#include "mem/cxx_backends.h"
#include "mem/cxx_vtype.cuh"

namespace psz {

using _ptb::atomic_add;
using _ptb::atomic_max;
using _ptb::atomic_min;

template <typename T>
__global__ void KCU_get_extrema(
    T* in, size_t const len, T* minel, T* maxel, T* sum, T const failsafe)
{
  constexpr int N = _ptb::_128b<T>::N;
  __shared__ T s_min, s_max, s_sum;
  T tp_min{failsafe}, tp_max{failsafe}, tp_sum{0};
  if (threadIdx.x == 0) s_min = failsafe, s_max = failsafe, s_sum = 0;
  __syncthreads();

  size_t regular_count = len / N;
  const auto stride = (size_t)gridDim.x * blockDim.x;
  const auto entry = blockIdx.x * blockDim.x;
  for (size_t i = entry + threadIdx.x; i < regular_count; i += stride) {
    auto v = _ptb::ld_128b(in + i * N);
#pragma unroll
    for (int j = 0; j < N; j++) {
      tp_min = min(tp_min, v[j]);
      tp_max = max(tp_max, v[j]);
      tp_sum += v[j];
    }
  }
  // tail/boundary handling
  for (size_t i = regular_count * N + entry + threadIdx.x; i < len; i += stride) {
    T v = in[i];
    tp_min = min(tp_min, v);
    tp_max = max(tp_max, v);
    tp_sum += v;
  }

#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    tp_min = min(tp_min, __shfl_down_sync(0xffffffff, tp_min, o));
    tp_max = max(tp_max, __shfl_down_sync(0xffffffff, tp_max, o));
    tp_sum += __shfl_down_sync(0xffffffff, tp_sum, o);
  }
  if ((threadIdx.x & 31) == 0) {
    atomic_min(&s_min, tp_min);
    atomic_max(&s_max, tp_max);
    atomic_add(&s_sum, tp_sum);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    atomic_min(minel, s_min);
    atomic_max(maxel, s_max);
    atomic_add(sum, s_sum);
  }
}

}  // namespace psz

namespace psz::cuda {

template <typename T>
auto GPU_get_extrema<T>::kernel(T* in, size_t len, void* stream) -> std::tuple<T, T, T, T>
{
  int numSM = 0;
  cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
  int block = 256, grid = numSM * 4;

  auto d_minel = MAKE_UNIQUE_DEVICE(T, 1);
  auto d_maxel = MAKE_UNIQUE_DEVICE(T, 1);
  auto d_sum = MAKE_UNIQUE_DEVICE(T, 1);  // malloc_device zero-inits the sum

  // failsafe/min/max seed from in
  T h_min, h_max, h_sum, failsafe;
  memcpy_allkinds<D2H>(&failsafe, in, 1);
  memcpy_allkinds<D2D>(d_minel.get(), in, 1);
  memcpy_allkinds<D2D>(d_maxel.get(), in, 1);

  psz::KCU_get_extrema<T><<<grid, block, 0, (cudaStream_t)stream>>>(
      in, len, d_minel.get(), d_maxel.get(), d_sum.get(), failsafe);
  sync_by_stream(stream);

  memcpy_allkinds<D2H>(&h_min, d_minel.get(), 1);
  memcpy_allkinds<D2H>(&h_max, d_maxel.get(), 1);
  memcpy_allkinds<D2H>(&h_sum, d_sum.get(), 1);

  return {h_min, h_max, h_sum / len, h_max - h_min};
}

}  // namespace psz::cuda

#endif /* _PSZ_UTILS_EXTREMA_CUH */
