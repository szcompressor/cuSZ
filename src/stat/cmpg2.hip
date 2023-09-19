#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include "cusz/type.h"
#include "stat/compare.thrust.hh"

bool psz::thrustgpu::thrustgpu_identical(
    void* d1, void* d2, size_t sizeof_T, size_t const len)
{
  return thrust::equal(
      thrust::device, (u1*)d1, (u1*)d1 + sizeof_T * len, (u1*)d2);
}
