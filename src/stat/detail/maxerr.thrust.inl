/**
 * @file maxerr.thrust.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-08
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef C6875E14_650F_49ED_9DD5_E7F916EE31FF
#define C6875E14_650F_49ED_9DD5_E7F916EE31FF

#include "stat/compare/compare.thrust.hh"

// #include <thrust/count.h>
// #include <thrust/iterator/constant_iterator.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "cusz/type.h"

namespace psz {

template <typename T>
void thrustgpu_get_maxerr(
    T* reconstructed,     // in
    T* original,          // in
    size_t len,           // in
    T& maximum_val,       // out
    size_t& maximum_loc,  // out
    bool destructive)
{
  T* diff;

  if (destructive) {
    diff = original;  // aliasing
  }
  else {
    GpuMalloc(&diff, sizeof(T) * len);
  }

  auto expr = [=] __device__(T rel, T oel) { return rel - oel; };

  // typesafe (also with exec-policy binding)
  thrust::device_ptr<T> r(reconstructed);
  thrust::device_ptr<T> o(original);
  thrust::device_ptr<T> d(diff);

  thrust::transform(r, r + len, o, d, expr);

  auto maximum_ptr = thrust::max_element(d, d + len);
  maximum_val = *maximum_ptr;
  maximum_loc = maximum_ptr - d;

  if (not destructive) { GpuFree(diff); }
}

}  // namespace psz

#endif /* C6875E14_650F_49ED_9DD5_E7F916EE31FF */
