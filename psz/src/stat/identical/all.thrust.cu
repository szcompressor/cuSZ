#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include "cusz/type.h"
#include "stat/compare.hh"

bool psz::thrustgpu::GPU_identical(
    void* d1, void* d2, size_t sizeof_T, size_t const len, void* stream)
{
#if defined(PSZ_USE_CUDA)
  thrust::cuda::par.on((cudaStream_t)stream);
#elif defined(PSZ_USE_HIP)
  thrust::hip::par.on((hipStream_t)stream);
#endif
  return thrust::equal(thrust::device, (u1*)d1, (u1*)d1 + sizeof_T * len, (u1*)d2);
}
