/**
 * @file ex_utils.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-13
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "cusz/type.h"
#include "ex_utils.hh"

template <typename T>
u4 count_outlier(T* in, size_t inlen, int radius, void* stream)
{
  using thrust::placeholders::_1;
#if defined(PSZ_USE_CUDA)
  thrust::cuda::par.on((cudaStream_t)stream);
#elif defined(PSZ_USE_HIP)
  thrust::hip::par.on((hipStream_t)stream);
#endif

  return thrust::count_if(
      thrust::device, in, in + inlen, _1 >= 2 * radius or _1 < 0);
}
