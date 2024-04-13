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

#ifndef C9DB8B27_F5D7_454F_9485_CAD4B2FE4A92
#define C9DB8B27_F5D7_454F_9485_CAD4B2FE4A92

#include <thrust/device_ptr.h>

#include "stat/compare/compare.thrust.hh"
// #include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

namespace psz {
namespace thrustgpu {

static const int MINVAL = 0;
static const int MAXVAL = 1;
static const int AVGVAL = 2;
static const int RNG = 3;

template <typename T>
void thrustgpu_get_extrema_rawptr(T* d_ptr, size_t len, T res[4])
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
void thrustgpu_get_extrema(thrust::device_ptr<T> g_ptr, size_t len, T res[4])
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

}  // namespace thrustgpu
}  // namespace psz

#endif /* C9DB8B27_F5D7_454F_9485_CAD4B2FE4A92 */
