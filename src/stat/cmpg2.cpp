#include <thrust/equal.h>

#include "cusz/type.h"
#include "stat/compare_thrust.hh"

bool psz::thrustgpu_identical(
    void* d1, void* d2, size_t sizeof_T, size_t const len)
{
  return thrust::equal(
      thrust::device, (u1*)d1, (u1*)d1 + sizeof_T * len, (u1*)d2);
}
