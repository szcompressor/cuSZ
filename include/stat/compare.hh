/**
 * @file compare.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-09
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef CE05A256_23CB_4243_8839_B1FDA9C540D2
#define CE05A256_23CB_4243_8839_B1FDA9C540D2

#include <stdint.h>
#include <stdlib.h>

#include "busyheader.hh"
#include "compare_cpu.hh"
#include "compare_gpu.hh"
#include "cusz/type.h"

namespace psz {

template <pszpolicy P, typename T>
bool identical(T* d1, T* d2, size_t const len)
{
  if (P == CPU)
    cppstd_identical(d1, d2, len);
  else if (P == THRUST)
    thrustgpu_identical(d1, d2, len);
  else {
    throw runtime_error(__FUNCTION__ + ": backend not supported.");
  }
}

template <pszpolicy P, typename T>
void get_extrema(T* d_ptr, size_t len, T res[4])
{
  if (P == THRUST)
    thrustgpu_get_extrema_rawptr(d_ptr, len, res[4]);
  else if (P == CUDA) {
  }
  else
    throw runtime_error(__FUNCTION__ + ": backend not supported.");
}

template <pszpolicy P, typename T>
bool error_bounded(
    T* a, T* b, size_t const len, double const eb,
    size_t* first_faulty_idx = nullptr)
{
  if (P == CPU)
    cppstd_error_bounded(a, b, len, eb, first_faulty_idx);
  else if (P == THRUST)
    thrustgpu_error_bounded(a, b, len, eb, first_faulty_idx);
  else
    throw runtime_error(__FUNCTION__ + ": backend not supported.");
}

template <pszpolicy P, typename T>
void assess_quality(pszsummary* s, T* xdata, T* odata, size_t const len)
{
  if (P == CPU)
    cppstd_assess_quality(s, xdata, odata, len);
  else if (P == THRUST)
    thrustgpu_assess_quality(s, xdata, odata, len);
  else
    throw runtime_error(__FUNCTION__ + ": backend not supported.");
}

}  // namespace psz

#endif /* CE05A256_23CB_4243_8839_B1FDA9C540D2 */
