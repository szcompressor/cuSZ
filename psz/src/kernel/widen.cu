#ifndef PSZ_KERNEL_IMPL_WIDEN_CUHIP_INL
#define PSZ_KERNEL_IMPL_WIDEN_CUHIP_INL

#include "kernel.hh"
#include "utils/err.hh"

namespace psz {

template <typename T, typename E>
__global__ void KCU_widen_eq(E const* src, T* dst, size_t const n)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) dst[tid] = static_cast<T>(src[tid]);
}

}  // namespace psz

template <typename T, typename E>
int psz::module::GPU_widen<T, E>::kernel(E const* src, T* dst, size_t const n, void* stream)
{
  auto grid_dim = (n - 1) / 256 + 1;
  psz::KCU_widen_eq<T, E><<<grid_dim, 256, 0, (cudaStream_t)stream>>>(src, dst, n);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  return PSZ_SUCCESS;
}

#endif /* PSZ_KERNEL_IMPL_WIDEN_CUHIP_INL */
