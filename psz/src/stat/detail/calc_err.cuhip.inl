// 24-06-02 by J. Tian

#include <type_traits>

#include "detail/compare.hh"

namespace psz {

template <typename T>
[[deprecated("same function with the one in extrema.cuhip.inl, keep in sync")]]
__device__ __forceinline__ T atomicAddFp(T* addr, T value)
{
  if constexpr (std::is_same<T, float>::value) { return atomicAdd(addr, value); }
  else if constexpr (std::is_same<T, double>::value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    return atomicAdd(addr, value);
#else
    auto addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr_as_ull, assumed;
    do {
      assumed = old;
      double next = __longlong_as_double(assumed) + value;
      old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(next));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
  }
  else {
    return atomicAdd(addr, value);
  }
}

template <typename T>
__global__ void KERNEL_CUHIP_calculate_errors(
    T* odata, T odata_avg, T* xdata, T xdata_avg, size_t len,  //
    T* sum_corr, T* sum_err_sq, T* sum_var_odata, T* sum_var_xdata, int const R)
{
  __shared__ T s_sum_corr;
  __shared__ T s_sum_err_sq;
  __shared__ T s_sum_var_odata;
  __shared__ T s_sum_var_xdata;

  T p_sum_corr{0}, p_sum_err_sq{0}, p_sum_var_odata{0}, p_sum_var_xdata{0};

  auto _entry = [&]() { return (blockDim.x * R) * blockIdx.x + threadIdx.x; };
  auto _idx = [&](auto r) { return _entry() + (r * blockDim.x); };

  if (threadIdx.x == 0) s_sum_corr = 0, s_sum_err_sq = 0, s_sum_var_odata = 0, s_sum_var_xdata = 0;
  __syncthreads();

  for (auto r = 0; r < R; r++) {
    auto i = _idx(r);
    if (i < len) {
      T o = odata[i];
      T x = xdata[i];
      T odiff = o - odata_avg;
      T xdiff = x - xdata_avg;

      p_sum_corr += odiff * xdiff;
      p_sum_err_sq += (o - x) * (o - x);
      p_sum_var_odata += odiff * odiff;
      p_sum_var_xdata += xdiff * xdiff;
    }
  }
  __syncthreads();

  atomicAddFp(&s_sum_corr, p_sum_corr);
  atomicAddFp(&s_sum_err_sq, p_sum_err_sq);
  atomicAddFp(&s_sum_var_odata, p_sum_var_odata);
  atomicAddFp(&s_sum_var_xdata, p_sum_var_xdata);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAddFp(sum_corr, s_sum_corr);
    atomicAddFp(sum_err_sq, s_sum_err_sq);
    atomicAddFp(sum_var_odata, s_sum_var_odata);
    atomicAddFp(sum_var_xdata, s_sum_var_xdata);
  }
}

}  // namespace psz

template <typename T>
void psz::cuhip::GPU_calculate_errors(
    T* d_odata, T odata_avg, T* d_xdata, T xdata_avg, size_t len, T h_err[4])
{
  constexpr auto SUM_CORR = 0;
  constexpr auto SUM_ERR_SQ = 1;
  constexpr auto SUM_VAR_ODATA = 2;
  constexpr auto SUM_VAR_XDATA = 3;

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  // TODO use external stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  T *d_sum_corr, *d_sum_err_sq, *d_sum_var_odata, *d_sum_var_xdata;

  cudaMalloc(&d_sum_corr, sizeof(T));
  cudaMalloc(&d_sum_err_sq, sizeof(T));
  cudaMalloc(&d_sum_var_odata, sizeof(T));
  cudaMalloc(&d_sum_var_xdata, sizeof(T));

  cudaMemset(d_sum_corr, 0, sizeof(T));
  cudaMemset(d_sum_err_sq, 0, sizeof(T));
  cudaMemset(d_sum_var_odata, 0, sizeof(T));
  cudaMemset(d_sum_var_xdata, 0, sizeof(T));

  auto chunk = 32768;
  auto nworker = 128;
  auto R = chunk / nworker;

  psz::KERNEL_CUHIP_calculate_errors<T><<<div(len, chunk), nworker, 0, stream>>>(
      d_odata, odata_avg, d_xdata, xdata_avg, len, d_sum_corr, d_sum_err_sq, d_sum_var_xdata,
      d_sum_var_odata, R);

  cudaStreamSynchronize(stream);

  cudaMemcpy(&h_err[SUM_CORR], d_sum_corr, sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_err[SUM_ERR_SQ], d_sum_err_sq, sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_err[SUM_VAR_ODATA], d_sum_var_odata, sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_err[SUM_VAR_XDATA], d_sum_var_xdata, sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_sum_corr);
  cudaFree(d_sum_err_sq);
  cudaFree(d_sum_var_odata);
  cudaFree(d_sum_var_xdata);

  cudaStreamDestroy(stream);
}

#define __INSTANTIATE_CUHIP_CALCERRORS(T)            \
  template void psz::cuhip::GPU_calculate_errors<T>( \
      T * d_odata, T odata_avg, T * d_xdata, T xdata_avg, size_t len, T h_err[4]);
