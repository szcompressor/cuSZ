/**
 * @file extreama.thrust.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-08
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "stat/compare.hh"

namespace psz::thrustgpu {

static const int MINVAL = 0;
static const int MAXVAL = 1;
static const int AVGVAL = 2;
static const int RNG = 3;

template <typename T>
void GPU_extrema(T* d_ptr, size_t len, T res[4])
{
  thrust::device_ptr<T> g_ptr = thrust::device_pointer_cast(d_ptr);

  auto minel = thrust::min_element(g_ptr, g_ptr + len) - g_ptr;
  auto maxel = thrust::max_element(g_ptr, g_ptr + len) - g_ptr;
  res[MINVAL] = *(g_ptr + minel);
  res[MAXVAL] = *(g_ptr + maxel);
  res[RNG] = res[MAXVAL] - res[MINVAL];

  auto sum = thrust::reduce(g_ptr, g_ptr + len, (T)0.0, thrust::plus<T>());
  res[AVGVAL] = sum / len;
}

// commented for better build time
/*
template <typename T>
void GPU_extrema(thrust::device_ptr<T> g_ptr, size_t len, T res[4])
{
    auto minel  = thrust::min_element(g_ptr, g_ptr + len) - g_ptr;
    auto maxel  = thrust::max_element(g_ptr, g_ptr + len) - g_ptr;
    res[MINVAL] = *(g_ptr + minel);
    res[MAXVAL] = *(g_ptr + maxel);
    res[RNG]    = res[MAXVAL] - res[MINVAL];

    auto sum    = thrust::reduce(g_ptr, g_ptr + len, (T)0.0,
thrust::plus<T>()); res[AVGVAL] = sum / len;
}
*/

}  // namespace psz::thrustgpu

#define __INSTANTIATE_THRUSTGPU_EXTREMA(T) \
  template void psz::thrustgpu::GPU_extrema(T* d_ptr, size_t len, T res[4]);
