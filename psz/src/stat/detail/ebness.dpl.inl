/**
 * @file _compare.cuh
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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>

#include "cusz/type.h"

namespace psz {

template <typename T>
bool thrustgpu_error_bounded(
    T* a, T* b, size_t const len, double eb,
    size_t* first_faulty_idx = nullptr)
{
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();
  dpct::device_pointer<T> a_ = dpct::get_device_pointer(a);
  dpct::device_pointer<T> b_ = dpct::get_device_pointer(b);
  dpct::constant_iterator<double> eb_(eb);
  using tup = std::tuple<T, T, double>;

  auto ab_begin = oneapi::dpl::make_zip_iterator(std::make_tuple(a_, b_, eb_));
  auto ab_end =
      oneapi::dpl::make_zip_iterator(std::make_tuple(a_ + len, b_ + len, eb_));

  // Let compiler figure out the type.
  auto iter = oneapi::dpl::find_if(
      oneapi::dpl::execution::make_device_policy(q_ct1), ab_begin, ab_end,
      [](tup t) {
        // debug use
        // if (fabs(thrust::get<1>(t) - thrust::get<0>(t)) > thrust::get<2>(t))
        //     printf("a: %f\tb: %f\teb: %lf\n", (float)thrust::get<1>(t),
        //     (float)thrust::get<0>(t), thrust::get<2>(t));

        return fabs(std::get<1>(t) - std::get<0>(t)) > 1.001 * std::get<2>(t);
      });

  if (iter == ab_end) { return true; }
  else {
    // *first_faulty_idx = iter - ab_begin;
    return false;
  }
}

}  // namespace psz

#endif /* DC032520_A30F_4F2D_A260_CCE0E88CF40C */
