#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#include "atomics.cuhip.inl"
#include "detail/compare.hh"

namespace psz {

template <typename T>
__global__ void KERNEL_CUHIP_extrema(
    T* in, size_t const len, T* minel, T* maxel, T* sum, T const failsafe, int const R)
{
  __shared__ T shared_minv, shared_maxv, shared_sum;
  // failsafe; require external setup
  T tp_minv{failsafe}, tp_maxv{failsafe}, tp_sum{0};

  auto _entry = [&]() { return (blockDim.x * R) * blockIdx.x + threadIdx.x; };
  auto _idx = [&](auto r) { return _entry() + (r * blockDim.x); };

  if (threadIdx.x == 0) shared_minv = failsafe, shared_maxv = failsafe, shared_sum = 0;

  __syncthreads();

  for (auto r = 0; r < R; r++) {
    auto idx = _idx(r);
    if (idx < len) {
      auto val = in[idx];

      tp_minv = min(tp_minv, val);
      tp_maxv = max(tp_maxv, val);
      tp_sum += val;
    }
  }
  __syncthreads();

  constexpr auto is_FP = std::is_floating_point<T>::value;
  constexpr auto is_INT = std::is_integral<T>::value;

  if constexpr (is_FP) {
    atomicMinFp<T>(&shared_minv, tp_minv);
    atomicMaxFp<T>(&shared_maxv, tp_maxv);
    atomicAddFp<T>(&shared_sum, tp_sum);
    __syncthreads();

    if (threadIdx.x == 0) {
      atomicMinFp<T>(minel, shared_minv);
      atomicMaxFp<T>(maxel, shared_maxv);
      atomicAddFp<T>(sum, shared_sum);
    }
  }
  else if constexpr (is_INT) {
    atomicMin(&shared_minv, tp_minv);
    atomicMax(&shared_maxv, tp_maxv);
    atomicAdd(&shared_sum, tp_sum);
    __syncthreads();

    if (threadIdx.x == 0) {
      atomicMin(minel, shared_minv);
      atomicMax(maxel, shared_maxv);
      atomicAdd(sum, shared_sum);
    }
  }
  else
    static_assert(is_FP || is_INT, "T must be either floating point or integer.");
}

}  // namespace psz

namespace psz::module {

template <typename T>
void GPU_extrema(T* in, size_t len, T res[4])
{
  static const int MINVAL = 0;
  static const int MAXVAL = 1;
  static const int AVGVAL = 2;
  static const int RNG = 3;

  // TODO use external stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto chunk = 32768;
  auto nworker = 128;
  auto R = chunk / nworker;

  T h_min, h_max, h_sum, failsafe;
  T *d_minel, *d_maxel, *d_sum;

  cudaMalloc(&d_minel, sizeof(T));
  cudaMalloc(&d_maxel, sizeof(T));
  cudaMalloc(&d_sum, sizeof(T));
  cudaMemset(d_sum, 0, sizeof(T));

  // failsafe init
  cudaMemcpy(&failsafe, in, sizeof(T), cudaMemcpyDeviceToHost);  // transfer the 1st val
  cudaMemcpy(d_minel, in, sizeof(T), cudaMemcpyDeviceToDevice);  // init min el
  cudaMemcpy(d_maxel, in, sizeof(T), cudaMemcpyDeviceToDevice);  // init max el

// launch
#if defined(PSZ_USE_CUDA)
  psz::KERNEL_CUHIP_extrema<T><<<div(len, chunk), nworker, sizeof(T) * 2, stream>>>(
      in, len, d_minel, d_maxel, d_sum, failsafe, R);
#elif defined(PSZ_USE_HIP)
  if constexpr (std::is_same<T, float>::value) {
    psz::extrema_kernel<float><<<div(len, chunk), nworker, sizeof(float) * 2, stream>>>(
        in, len, d_minel, d_maxel, d_sum, failsafe, R);
  }
  else {
    throw std::runtime_error(
        "As of now (5.5.30202), HIP does not support 64-bit integer atomic "
        "operation.");
  }
#endif

  cudaStreamSynchronize(stream);

  // collect results
  cudaMemcpy(&h_min, d_minel, sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_max, d_maxel, sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_sum, d_sum, sizeof(T), cudaMemcpyDeviceToHost);

  res[MINVAL] = h_min;
  res[MAXVAL] = h_max;
  res[AVGVAL] = h_sum / len;
  res[RNG] = h_max - h_min;

  cudaFree(d_minel);
  cudaFree(d_maxel);
  cudaFree(d_sum);

  cudaStreamDestroy(stream);
}

}  // namespace psz::module

#define __INSTANTIATE_CUHIP_EXTREMA(T) \
  template void psz::module::GPU_extrema<T>(T * in, size_t len, T res[4]);
