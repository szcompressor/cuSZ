
#ifndef E85FA9DC_9350_4F91_B4D8_5C094661479E
#define E85FA9DC_9350_4F91_B4D8_5C094661479E

#include "cusz/cxx_array.hh"
#include "cusz/type.h"

namespace _2401 {

template <typename T>
pszerror pszcxx_predict_lorenzo(
    pszarray_cxx<T> in, pszrc2 const rc, pszarray_cxx<u4> out_errquant,
    pszcompact_cxx<T> out_outlier, void* stream);

template <typename T>
pszerror pszcxx_predict_spline(
    pszarray_cxx<T> in, pszrc2 const rc, pszarray_cxx<u4> out_errquant,
    pszcompact_cxx<T> out_outlier, pszarray_cxx<T> out_anchor, float* time,
    void* stream);

template <typename T>
pszerror pszcxx_reverse_predict_lorenzo(
    pszarray_cxx<u4> in_errquant, pszarray_cxx<T> in_scattered_outlier,
    pszrc2 const rc, pszarray_cxx<T> out_reconstruct, void* stream);

template <typename T>
pszerror pszcxx_reverse_predict_spline(
    pszarray_cxx<u4> in_errquant, pszarray_cxx<T> in_scattered_outlier,
    pszarray_cxx<T> in_anchor, pszrc2 const rc,
    pszarray_cxx<T> out_reconstruct, float* time, void* stream);

template <pszpolicy Policy, typename T>
pszerror pszcxx_histogram_cauchy(
    pszarray_cxx<T> in, pszarray_cxx<u4> out_hist, float* milliseconds,
    void* stream);

template <pszpolicy P, typename T>
pszerror pszcxx_scatter_naive(
    pszcompact_cxx<T> in_outlier, pszarray_cxx<T> out_decomp_space,
    float* milliseconds, void* stream);

template <pszpolicy P, typename T>
pszerror pszcxx_gather_make_metadata_host_available(
    pszcompact_cxx<T> in_outlier, void* stream);

// see viewer.cu_hip.hh
// template <pszpolicy P, typename T>
// pszerror pszcxx_evaluate_quality(pszarray_cxx<T> d1, pszarray_cxx<T> d2);

};  // namespace _2401

#endif /* E85FA9DC_9350_4F91_B4D8_5C094661479E */
