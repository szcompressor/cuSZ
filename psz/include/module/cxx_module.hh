
#ifndef E85FA9DC_9350_4F91_B4D8_5C094661479E
#define E85FA9DC_9350_4F91_B4D8_5C094661479E

#include "kernel/lrz.hh"
#include "mem/array_cxx.h"
#include "mem/compact.hh"

using portable::array3;
using portable::compact_array1;

// TODO add execution policy as template parameter
template <
    typename T, typename E, psz_timing_mode TIMING = SYNC_BY_STREAM,
    bool USE_PROTO = false>
pszerror pszcxx_predict_lorenzo(
    array3<T> in, psz_rc const rc, array3<E> out_errquant,
    compact_array1<T> out_outlier, f4* time_elapsed, void* stream)
try {
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  auto len3 = dim3(in.len3.x, in.len3.y, in.len3.z);
  auto compact = typename CompactDram<CUDA, T>::Compact(out_outlier);
  if constexpr (USE_PROTO)
    psz::cuhip::proto::GPU_c_lorenzo_nd_with_outlier<T, E>(
        in.buf, len3, rc.eb, rc.radius, out_errquant.buf, (void*)&compact,
        time_elapsed, stream);
  else
    psz::cuhip::GPU_c_lorenzo_nd_with_outlier<T, E, TIMING>(
        in.buf, len3, rc.eb, rc.radius, out_errquant.buf, (void*)&compact,
        time_elapsed, stream);
  return CUSZ_SUCCESS;
#elif defined(PSZ_USE_1API)
  // TODO
#endif
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)

template <
    typename T, typename E, psz_timing_mode TIMING = SYNC_BY_STREAM,
    bool USE_PROTO = false>
pszerror pszcxx_reverse_predict_lorenzo(
    array3<E> in_errquant, array3<T> in_scattered_outlier, psz_rc const rc,
    array3<T> out_xdata, f4* time_elapsed, void* stream)
try {
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  auto len3 = dim3(out_xdata.len3.x, out_xdata.len3.y, out_xdata.len3.z);
  if constexpr (USE_PROTO)
    psz::cuhip::proto::GPU_x_lorenzo_nd<T, E>(
        in_errquant.buf, len3, out_xdata.buf, rc.eb, rc.radius, out_xdata.buf,
        time_elapsed, stream);
  else
    psz::cuhip::GPU_x_lorenzo_nd<T, E, TIMING>(
        in_errquant.buf, len3, out_xdata.buf, rc.eb, rc.radius, out_xdata.buf,
        time_elapsed, stream);
  return CUSZ_SUCCESS;
#elif defined(PSZ_USE_1API)
  // TODO
#endif
}
NONEXIT_CATCH(psz::exception_placeholder, CUSZ_NOT_IMPLEMENTED)
NONEXIT_CATCH(psz::exception_incorrect_type, CUSZ_FAIL_UNSUPPORTED_DATATYPE)

/* TODO substitute the two functions in spline.hh
template <
    typename T, typename E, psz_timing_mode TIMING = SYNC_BY_STREAM>
static pszerror pszcxx_predict_spline(
    array3<T> in, psz_rc const rc, array3<E> out_errquant,
    compact_array1<T> out_outlier, array3<T> out_anchor, float* time,
    void* stream);

template <
    typename T, typename E, psz_timing_mode TIMING = SYNC_BY_STREAM>
static pszerror pszcxx_reverse_predict_spline(
    array3<E> in_errquant, array3<T> in_scattered_outlier, array3<T> in_anchor,
    psz_rc const rc, array3<T> out_reconstruct, float* time, void* stream);
*/

template <psz_policy Poilicy, typename T>
pszerror pszcxx_compat_histogram_generic(
    T* in, size_t const inlen, uint32_t* out_hist, int const outlen,
    float* milliseconds, void* stream = nullptr);

template <psz_policy Policy, typename T, typename FQ = uint32_t>
int pszcxx_compat_histogram_cauchy(
    T* in, uint32_t inlen, FQ* out_hist, uint32_t outlen, float* milliseconds,
    void* stream = nullptr);

template <psz_policy Policy, typename T, bool TIMING = true>
pszerror pszcxx_histogram_cauchy(
    array3<T> in, array3<u4> out_hist, float* milliseconds, void* stream);

template <psz_policy P, typename T, bool TIMING = true>
pszerror pszcxx_scatter_naive(
    compact_array1<T> in_outlier, array3<T> out_decomp_space,
    float* milliseconds, void* stream);

template <psz_policy P, typename T, bool TIMING = true>
pszerror pszcxx_gather_make_metadata_host_available(
    compact_array1<T> in_outlier, void* stream);

// see viewer.cuhip.hh
// template <psz_policy P, typename T>
// pszerror pszcxx_evaluate_quality(array3<T> d1, array3<T> d2);

#endif /* E85FA9DC_9350_4F91_B4D8_5C094661479E */
