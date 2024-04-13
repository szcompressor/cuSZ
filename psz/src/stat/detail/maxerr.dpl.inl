#ifndef C6875E14_650F_49ED_9DD5_E7F916EE31FF
#define C6875E14_650F_49ED_9DD5_E7F916EE31FF

#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include "cusz/type.h"
#include "stat/compare/compare.dpl.hh"

namespace psz {

template <typename T>
void dpl_get_maxerr(
    T* reconstructed,     // in
    T* original,          // in
    size_t len,           // in
    T& maximum_val,       // out
    size_t& maximum_loc,  // out
    bool destructive)
{
  dpct::device_ext& dev_ct1 = dpct::get_current_device();
  sycl::queue& q_ct1 = dev_ct1.default_queue();
  T* diff;

  if (destructive) {
    diff = original;  // aliasing
  }
  else {
    diff = (T*)sycl::malloc_device(sizeof(T) * len, q_ct1);
  }

  auto expr = [=](T rel, T oel) { return rel - oel; };

  // typesafe (also with exec-policy binding)
  dpct::device_pointer<T> r(reconstructed);
  dpct::device_pointer<T> o(original);
  dpct::device_pointer<T> d(diff);

  std::transform(
      oneapi::dpl::execution::make_device_policy(q_ct1), r, r + len, o, d,
      expr);

  auto maximum_ptr = std::max_element(oneapi::dpl::execution::seq, d, d + len);
  maximum_val = *maximum_ptr;
  maximum_loc = maximum_ptr - d;

  if (not destructive) { sycl::free(diff, q_ct1); }
}

}  // namespace psz

#endif /* C6875E14_650F_49ED_9DD5_E7F916EE31FF */
