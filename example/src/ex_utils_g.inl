// Author: Jiannan Tian

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "cusz/type.h"
#include "ex_utils.hh"

template <typename T>
[[deprecated("-> test")]] u4 count_outlier(T* in, size_t inlen, int radius, void* stream)
{
  using thrust::placeholders::_1;
#if defined(PSZ_USE_CUDA)
  thrust::cuda::par.on((cudaStream_t)stream);
#endif

  return thrust::count_if(thrust::device, in, in + inlen, _1 >= 2 * radius or _1 < 0);
}
