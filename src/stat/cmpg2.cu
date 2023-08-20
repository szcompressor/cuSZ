/**
 * @file cmp2g.cu
 * @author Jiannan Tian
 * @brief (split to speed up buid process; part 2)
 * @version 0.3
 * @date 2022-11-03
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "detail/equal_thrust.inl"
#include "stat/compare_gpu.hh"
#include "cusz/type.h"

bool psz::thrustgpu_identical(void* d1, void* d2, size_t sizeof_T, size_t const len)
{
  return thrust::equal(
      thrust::device, (u1*)d1, (u1*)d1 + sizeof_T * len, (u1*)d2);
}

#undef THRUSTGPU_COMPARE_LOSSLESS
