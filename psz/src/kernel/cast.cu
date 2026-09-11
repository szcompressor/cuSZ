#ifndef PSZ_KERNEL_IMPL_CAST_CUHIP_INL
#define PSZ_KERNEL_IMPL_CAST_CUHIP_INL

#include "cusz/type.h"
#include "kernel.hh"
#include "utils/err.hh"

namespace psz {

template <typename Tin, typename Tout>
__global__ void KCU_cast(Tin* in, Tout* out, size_t const n)
{
  auto tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) out[tid] = (Tout)in[tid];
}

}  // namespace psz

template <typename Tin, typename Tout>
int psz::module::GPU_cast<Tin, Tout>::kernel(Tin* in, Tout* out, size_t n, void* stream)
{
  auto grid_dim = (n - 1) / 256 + 1;
  psz::KCU_cast<Tin, Tout><<<grid_dim, 256, 0, (cudaStream_t)stream>>>(in, out, n);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  return PSZ_SUCCESS;
}

#endif /* PSZ_KERNEL_IMPL_CAST_CUHIP_INL */
