/**
 * @file ebness.thrust.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-08
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef DC032520_A30F_4F2D_A260_CCE0E88CF40C
#define DC032520_A30F_4F2D_A260_CCE0E88CF40C

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>

#include "cusz/type.h"

namespace psz {
namespace thrustgpu {

template <typename T>
bool thrustgpu_error_bounded(
    T* a, T* b, size_t const len, double eb,
    size_t* first_faulty_idx = nullptr)
{
  thrust::device_ptr<T> a_ = thrust::device_pointer_cast(a);
  thrust::device_ptr<T> b_ = thrust::device_pointer_cast(b);
  thrust::constant_iterator<double> eb_(eb);
  using tup = thrust::tuple<T, T, double>;

  auto ab_begin = thrust::make_zip_iterator(thrust::make_tuple(a_, b_, eb_));
  auto ab_end =
      thrust::make_zip_iterator(thrust::make_tuple(a_ + len, b_ + len, eb_));

  // Let compiler figure out the type.
  auto iter =
      thrust::find_if(thrust::device, ab_begin, ab_end, [] __device__(tup t) {
        // debug use
        // if (fabs(thrust::get<1>(t) - thrust::get<0>(t)) > thrust::get<2>(t))
        //     printf("a: %f\tb: %f\teb: %lf\n", (float)thrust::get<1>(t),
        //     (float)thrust::get<0>(t), thrust::get<2>(t));

        return fabs(thrust::get<1>(t) - thrust::get<0>(t)) >
               1.001 * thrust::get<2>(t);
      });

  if (iter == ab_end) { return true; }
  else {
    // *first_faulty_idx = iter - ab_begin;
    return false;
  }
}

}  // namespace thrustgpu
}  // namespace psz

#endif /* DC032520_A30F_4F2D_A260_CCE0E88CF40C */
