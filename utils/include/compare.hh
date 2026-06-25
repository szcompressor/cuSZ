#ifndef CE05A256_23CB_4243_8839_B1FDA9C540D2
#define CE05A256_23CB_4243_8839_B1FDA9C540D2

#include <stdint.h>
#include <stdlib.h>

#include <tuple>

#include "c_type.h"
#include "stat.h"

// Type alias for runtime enum
using psz_runtime = _ptb_runtime;

// clang-format off
namespace psz::cppstl {
bool CPU_identical(void* d1, void* d2, size_t sizeof_T, size_t const len);
template <typename T> void CPU_assess_quality(psz_stats* s, T* xdata, T* odata, size_t const len);
template <typename T> void CPU_calc_errors(T* odata, T odata_avg, T* xdata, T xdata_avg, size_t len, T err[4]);
template <typename T> void CPU_extrema(T* ptr, size_t len, T res[4]);
template <typename T> bool CPU_error_bounded(T* a, T* b, size_t const len, double const eb, size_t* first_faulty_idx = nullptr);
template <typename T> void CPU_find_max_error(T* a, T* b, size_t const len, T& maxval, size_t& maxloc);
}  // namespace psz::cppstl

namespace psz::cuda {
bool GPU_identical(void* d1, void* d2, size_t sizeof_T, size_t const len, void* stream = nullptr);
template <typename T> void GPU_extrema(T* d_ptr, size_t len, T res[4]);
template <typename T> void GPU_find_max_error(T* a, T* b, size_t const len, T& maxval, size_t& maxloc, void* stream = nullptr);
template <typename T> void GPU_assess_quality(psz_stats* s, T* xdata, T* odata, size_t const len);
template <typename T> void GPU_calc_errors(T* d_odata, T odata_avg, T* d_xdata, T xdata_avg, size_t len, T h_err[4]);
}  // namespace psz::cuda

namespace psz::dpcpp {
template <typename T> void GPU_extrema(T* d_ptr, size_t len, T res[4]);
template <typename T> void GPU_find_max_error(T* a, T* b, size_t len, T& maxval, size_t& maxloc, bool destructive = false);
template <typename T> void GPU_assess_quality(psz_stats* s, T* xdata, T* odata, size_t const len);
template <typename T> void GPU_calculate_errors(T* d_odata, T odata_avg, T* d_xdata, T xdata_avg, size_t len, T h_err[4]);
}  // namespace psz::dpcpp
// clang-format on

namespace psz::analysis {

template <typename T, psz_runtime R = SEQ>
std::tuple<T, T, T, T> CPU_probe_extrema(T* in, size_t len)
{
  static_assert(R == SEQ, "CPU_probe_extrema supports SEQ only.");
  T r[4];
  cppstl::CPU_extrema(in, len, r);
  return {r[0], r[1], r[2], r[3]};
}

template <typename T, psz_runtime R = CUDA>
std::tuple<T, T, T, T> GPU_probe_extrema(T* in, size_t len)
{
  T r[4];
  if constexpr (R == CUDA or R == HIP)
    cuda::GPU_extrema(in, len, r);
  else if constexpr (R == SYCL)
    dpcpp::GPU_extrema(in, len, r);
  else
    static_assert(
        R == CUDA or R == HIP or R == SYCL, "GPU_probe_extrema supports CUDA / HIP / SYCL.");
  return {r[0], r[1], r[2], r[3]};
}

template <psz_runtime P, typename T>
void assess_quality(psz_stats* s, T* xdata, T* odata, size_t const len)
{
  if constexpr (P == SEQ)
    cppstl::CPU_assess_quality(s, xdata, odata, len);
  else if constexpr (P == CUDA or P == HIP)
    cuda::GPU_assess_quality<T>(s, xdata, odata, len);
  else if constexpr (P == SYCL)
    dpcpp::GPU_assess_quality(s, xdata, odata, len);
  else
    static_assert(
        P == SEQ or P == CUDA or P == HIP or P == SYCL, "assess_quality: unsupported backend.");
}

}  // namespace psz::analysis

#endif /* CE05A256_23CB_4243_8839_B1FDA9C540D2 */
