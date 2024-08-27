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
#include "cusz/type.h"

// clang-format off
namespace psz::cppstl {
bool CPU_identical(void* d1, void* d2, size_t sizeof_T, size_t const len);
template <typename T> void CPU_assess_quality(psz_summary* s, T* xdata, T* odata, size_t const len);
template <typename T> void CPU_calculate_errors(T* odata, T odata_avg, T* xdata, T xdata_avg, size_t len, T err[4]);
template <typename T> void CPU_extrema(T* ptr, size_t len, T res[4]);
template <typename T> bool CPU_error_bounded(T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx = nullptr);
}  // namespace psz::cu_hip

namespace psz::cuhip {
template <typename T> void GPU_assess_quality(psz_summary* s, T* xdata, T* odata, size_t const len);
template <typename T> void GPU_calculate_errors(T* d_odata, T odata_avg, T* d_xdata, T xdata_avg, size_t len, T h_err[4]);
template <typename T> void GPU_extrema(T* d_ptr, size_t len, T res[4]);
template <typename T> bool GPU_error_bounded(T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx = nullptr);
}  // namespace psz::cu_hip

namespace psz::thrustgpu {
bool GPU_identical(void* d1, void* d2, size_t sizeof_T, size_t const len);
template <typename T> void GPU_assess_quality(psz_summary* s, T* xdata, T* odata, size_t const len);
template <typename T> void GPU_calculate_errors(T* d_odata, T odata_avg, T* d_xdata, T xdata_avg, size_t len, T h_err[4]);
template <typename T> void GPU_extrema(T* d_ptr, size_t len, T res[4]);
template <typename T> bool GPU_error_bounded(T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx);
template <typename T> void GPU_max_error(T* xdata, T* original, size_t len, T& maxval, size_t& maxloc, bool destructive = false);
}  // namespace psz::thrustgpu

namespace psz::dpcpp {
template <typename T> void GPU_assess_quality(psz_summary* s, T* xdata, T* odata, size_t const len);
template <typename T> void GPU_calculate_errors(T* d_odata, T odata_avg, T* d_xdata, T xdata_avg, size_t len, T h_err[4]);
template <typename T> void GPU_extrema(T* d_ptr, size_t len, T res[4]);
}  // namespace psz::dpcpp

namespace psz::dpl {
template <typename T> void GPU_assess_quality(psz_summary* s, T* xdata, T* odata, size_t const len);
template <typename T> void GPU_calculate_errors(T* d_odata, T odata_avg, T* d_xdata, T xdata_avg, size_t len, T h_err[4]);
template <typename T> void GPU_extrema(T* d_ptr, size_t len, T res[4]);
}  // namespace psz::dpl

// clang-format on

namespace psz {

template <psz_policy P, typename T>
bool identical(T* d1, T* d2, size_t const len)
{
  if (P == SEQ)
    cppstl::CPU_identical(d1, d2, sizeof(T), len);
  else if (P == THRUST)
    thrustgpu::GPU_identical(d1, d2, sizeof(T), len);
  else {
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");
  }
}

template <psz_policy P, typename T>
void probe_extrema(T* in, size_t len, T res[4])
{
  if (P == SEQ) cppstl::CPU_extrema(in, len, res);
// #ifdef REACTIVATE_THRUSTGPU
  else if (P == THRUST)
    thrustgpu::GPU_extrema(in, len, res);
// #endif
  else if (P == CUDA or P == HIP) 
    cuhip::GPU_extrema(in, len, res);
  else if (P == ONEAPI) 
    dpcpp::GPU_extrema(in, len, res);
  else
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");
}

template <psz_policy P, typename T>
bool error_bounded(
    T* a, T* b, size_t const len, double const eb,
    size_t* first_faulty_idx = nullptr)
{
  bool eb_ed = true;
  if (P == SEQ)
    eb_ed = cppstl::CPU_error_bounded(a, b, len, eb, first_faulty_idx);
#ifdef REACTIVATE_THRUSTGPU
  else if (P == THRUST)
    eb_ed = thrustgpu::GPU_error_bounded(a, b, len, eb, first_faulty_idx);
#endif
  else
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");
  return eb_ed;
}

template <psz_policy P, typename T>
void assess_quality(psz_summary* s, T* xdata, T* odata, size_t const len)
{
  // [TODO] THRUST is not activated in the frontend
  if constexpr (P == SEQ)
    cppstl::CPU_assess_quality(s, xdata, odata, len);
  else if constexpr (P == CUDA)
    cuhip::GPU_assess_quality<T>(s, xdata, odata, len);
  else if constexpr (P == THRUST)
    thrustgpu::GPU_assess_quality(s, xdata, odata, len);
  else if constexpr (P == ONEAPI) {
    dpl::GPU_assess_quality(s, xdata, odata, len);
  }
  else
    throw runtime_error(string(__FUNCTION__) + ": backend not supported.");
}

}  // namespace psz

#endif /* CE05A256_23CB_4243_8839_B1FDA9C540D2 */
