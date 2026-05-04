#ifndef CE05A256_23CB_4243_8839_B1FDA9C540D2
#define CE05A256_23CB_4243_8839_B1FDA9C540D2

#include <stdint.h>
#include <stdlib.h>

#include <tuple>
#include <type_traits>

#include "c_type.h"
#include "stat.h"

// Type alias for runtime enum
using psz_runtime = _portable_runtime;

// clang-format off
namespace psz::cppstl {
bool CPU_identical(void* d1, void* d2, size_t sizeof_T, size_t const len);
template <typename T> void CPU_assess_quality(psz_stats* s, T* xdata, T* odata, size_t const len);
template <typename T> void CPU_calculate_errors(T* odata, T odata_avg, T* xdata, T xdata_avg, size_t len, T err[4]);
template <typename T> void CPU_extrema(T* ptr, size_t len, T res[4]);
template <typename T> bool CPU_error_bounded(T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx = nullptr);
template <typename T> void CPU_find_max_error(T* a, T* b, size_t const len, T& maxval, size_t& maxloc);
}  // namespace psz::cppstl

namespace psz::module {
bool GPU_identical(void* d1, void* d2, size_t sizeof_T, size_t const len, void* stream = nullptr);
template <typename T> void GPU_extrema(T* d_ptr, size_t len, T res[4]);
template <typename T> void GPU_find_max_error(T* a, T* b, size_t const len, T& maxval, size_t& maxloc, void* stream = nullptr);
}

namespace psz::cuhip {
template <typename T> void GPU_assess_quality(psz_stats* s, T* xdata, T* odata, size_t const len);
template <typename T> void GPU_calculate_errors(T* d_odata, T odata_avg, T* d_xdata, T xdata_avg, size_t len, T h_err[4]);
template <typename T> bool GPU_error_bounded(T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx = nullptr);
}  // namespace psz::cuhip

namespace psz::thrustgpu {
bool GPU_identical(void* d1, void* d2, size_t sizeof_T, size_t const len, void* stream = nullptr);
template <typename T> void GPU_assess_quality(psz_stats* s, T* xdata, T* odata, size_t const len);
template <typename T> void GPU_calculate_errors(T* d_odata, T odata_avg, T* d_xdata, T xdata_avg, size_t len, T h_err[4]);
template <typename T> void GPU_extrema(T* d_ptr, size_t len, T res[4]);
template <typename T> bool GPU_error_bounded(T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx);
template <typename T> void GPU_find_max_error(T* xdata, T* original, size_t len, T& maxval, size_t& maxloc, bool destructive = false);
}  // namespace psz::thrustgpu

namespace psz::dpcpp {
template <typename T> void GPU_assess_quality(psz_stats* s, T* xdata, T* odata, size_t const len);
template <typename T> void GPU_calculate_errors(T* d_odata, T odata_avg, T* d_xdata, T xdata_avg, size_t len, T h_err[4]);
template <typename T> void GPU_extrema(T* d_ptr, size_t len, T res[4]);
}  // namespace psz::dpcpp

namespace psz::dpl {
template <typename T> void GPU_assess_quality(psz_stats* s, T* xdata, T* odata, size_t const len);
template <typename T> void GPU_calculate_errors(T* d_odata, T odata_avg, T* d_xdata, T xdata_avg, size_t len, T h_err[4]);
template <typename T> void GPU_extrema(T* d_ptr, size_t len, T res[4]);
}  // namespace psz::dpl

// clang-format on

namespace psz::analysis {

template <auto...>
inline constexpr bool unsupported_backend_v = false;

template <psz_runtime P, typename T>
bool identical(T* d1, T* d2, size_t const len)
{
  if constexpr (P == SEQ) return cppstl::CPU_identical(d1, d2, sizeof(T), len);
#ifdef REACTIVATE_THRUSTGPU
  else if constexpr (P == THRUST_DPL)
    return thrustgpu::GPU_identical(d1, d2, sizeof(T), len);
#endif
  else
    static_assert(unsupported_backend_v<P>, "identical: backend not supported.");
}

template <typename T1, psz_runtime R = SEQ, typename T2 = T1>
auto CPU_probe_extrema(T1* in, size_t len) -> std::tuple<T2, T2, T2, T2>
{
  T1 res[4];

  if constexpr (R == SEQ)
    cppstl::CPU_extrema(in, len, res);
  else
    static_assert(unsupported_backend_v<R>, "CPU_probe_extrema: backend not supported.");

  return {
      static_cast<T2>(res[0]), static_cast<T2>(res[1]), static_cast<T2>(res[2]),
      static_cast<T2>(res[3])};
}

template <typename T1, psz_runtime R = CUDA, typename T2 = T1>
auto GPU_probe_extrema(T1* in, size_t len) -> std::tuple<T2, T2, T2, T2>
{
  T1 result[4];

  if constexpr (R == CUDA)
    module::GPU_extrema(in, len, result);
  else if constexpr (R == SYCL)
    dpcpp::GPU_extrema(in, len, result);
#ifdef REACTIVATE_THRUSTGPU
  else if constexpr (R == THRUST_DPL)
    thrustgpu::GPU_extrema(in, len, result);
#endif
  else
    static_assert(unsupported_backend_v<R>, "GPU_probe_extrema: backend not supported.");

  return {
      static_cast<T2>(result[0]), static_cast<T2>(result[1]), static_cast<T2>(result[2]),
      static_cast<T2>(result[3])};
}

template <psz_runtime P, typename T>
bool error_bounded(
    T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx = nullptr)
{
  if constexpr (P == SEQ) return cppstl::CPU_error_bounded(a, b, len, eb, first_faulty_idx);
#ifdef REACTIVATE_THRUSTGPU
  else if constexpr (P == THRUST_DPL)
    return thrustgpu::GPU_error_bounded(a, b, len, eb, first_faulty_idx);
#endif
  else
    static_assert(unsupported_backend_v<P>, "error_bounded: backend not supported.");
}

template <psz_runtime P, typename T>
void assess_quality(psz_stats* s, T* xdata, T* odata, size_t const len)
{
  // [TODO] THRUST_DPL is not activated in the frontend
  if constexpr (P == SEQ)
    cppstl::CPU_assess_quality(s, xdata, odata, len);
  else if constexpr (P == CUDA)
    cuhip::GPU_assess_quality<T>(s, xdata, odata, len);
#ifdef REACTIVATE_THRUSTGPU
  else if constexpr (P == THRUST_DPL)
    thrustgpu::GPU_assess_quality(s, xdata, odata, len);
#endif
  else if constexpr (P == SYCL) {
    dpl::GPU_assess_quality(s, xdata, odata, len);
  }
  else
    static_assert(unsupported_backend_v<P>, "assess_quality: backend not supported.");
}

}  // namespace psz::analysis

#endif /* CE05A256_23CB_4243_8839_B1FDA9C540D2 */
