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
#include "compare/compare.cu_hip.hh"
#include "compare/compare.dp.hh"
#include "compare/compare.dpl.hh"
#include "compare/compare.stl.hh"
#include "compare/compare.thrust.hh"
#include "cusz/type.h"

namespace psz {

template <pszpolicy P, typename T>
bool identical(T* d1, T* d2, size_t const len)
{
  if (P == SEQ)
    psz::cppstl_identical(d1, d2, len);
  else if (P == THRUST)
    thrustgpu_identical(d1, d2, len);
  else {
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");
  }
}

template <pszpolicy P, typename T>
void probe_extrema(T* in, size_t len, T res[4])
{
  if (P == SEQ) psz::cppstl_extrema(in, len, res);
#ifdef REACTIVATE_THRUSTGPU
  else if (P == THRUST)
    thrustgpu::thrustgpu_get_extrema_rawptr(in, len, res);
#endif
  else if (P == CUDA or P == HIP) {
    psz::cu_hip::extrema(in, len, res);
  }
  else if (P == ONEAPI) {
    psz::dpcpp::extrema(in, len, res);
  }
  else
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");
}

template <pszpolicy P, typename T>
bool error_bounded(
    T* a, T* b, size_t const len, double const eb,
    size_t* first_faulty_idx = nullptr)
{
  bool eb_ed = true;
  if (P == SEQ)
    eb_ed = psz::cppstl_error_bounded(a, b, len, eb, first_faulty_idx);
#ifdef REACTIVATE_THRUSTGPU
  else if (P == THRUST)
    eb_ed = psz::thrustgpu::thrustgpu_error_bounded(
        a, b, len, eb, first_faulty_idx);
#endif
  else
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");
  return eb_ed;
}

template <pszpolicy P, typename T>
void assess_quality(pszsummary* s, T* xdata, T* odata, size_t const len)
{
  // [TODO] THRUST is not activated in the frontend
  if (P == SEQ)
    psz::cppstl_assess_quality(s, xdata, odata, len);
  else if (P == THRUST)
    psz::thrustgpu_assess_quality(s, xdata, odata, len);
  else if (P == ONEAPI) {
#if defined(PSZ_USE_1API)
    if constexpr (std::is_same_v<T, f4>) {
      psz::dpl_assess_quality(s, xdata, odata, len);
    }
    else {
      static_assert(
          std::is_same_v<T, f4>, "No f8, fast fail on sycl::aspects::fp64.");
    }
#endif
  }
  else
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");
}

}  // namespace psz

#endif /* CE05A256_23CB_4243_8839_B1FDA9C540D2 */
