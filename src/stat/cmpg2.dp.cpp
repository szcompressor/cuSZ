#include <dpct/dpct.hpp>
#include <dpct/dpl_utils.hpp>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include "busyheader.hh"
#include "cusz/type.h"
#include "stat/compare/compare.dpl.hh"

bool psz::dpl_identical(void* d1, void* d2, size_t sizeof_T, size_t const len)
{
#warning "DPCT1007:90: Migration of thrust::equal is not supported."
  /*
  DPCT1007:90: Migration of thrust::equal is not supported.
  */
  //   return thrust::equal(
  //       oneapi::dpl::execution::make_device_policy(dpct::get_default_queue()),
  //       (u1*)d1, (u1*)d1 + sizeof_T * len, (u1*)d2);
  throw runtime_error(
      "DPCT1007:90: Migration of thrust::equal is not supported.");
  return false;
}
