
#ifndef E85FA9DC_9350_4F91_B4D8_5C094661479E
#define E85FA9DC_9350_4F91_B4D8_5C094661479E

#include "cusz/type.h"
#include "mem/array_cxx.h"

namespace _2401 {

using namespace portable;

template <typename T, psz_timing_mode TIMING = CPU_BARRIER_AND_TIMING>
class pszpred_lrz {
 public:
  static pszerror pszcxx_predict_lorenzo(
      array3<T> in, pszrc2 const rc, array3<u4> out_errquant,
      compact_array1<T> out_outlier, f4*, void* stream);

  static pszerror pszcxx_reverse_predict_lorenzo(
      array3<u4> in_errquant, array3<T> in_scattered_outlier, pszrc2 const rc,
      array3<T> out_reconstruct, f4*, void* stream);
};

template <typename T, psz_timing_mode TIMING = CPU_BARRIER_AND_TIMING>
class pszpred_spl {
  static pszerror pszcxx_predict_spline(
      array3<T> in, pszrc2 const rc, array3<u4> out_errquant,
      compact_array1<T> out_outlier, array3<T> out_anchor, float* time,
      void* stream);

  static pszerror pszcxx_reverse_predict_spline(
      array3<u4> in_errquant, array3<T> in_scattered_outlier,
      array3<T> in_anchor, pszrc2 const rc, array3<T> out_reconstruct,
      float* time, void* stream);
};

template <pszpolicy Policy, typename T, bool TIMING = true>
pszerror pszcxx_histogram_cauchy(
    array3<T> in, array3<u4> out_hist, float* milliseconds, void* stream);

template <pszpolicy P, typename T, bool TIMING = true>
pszerror pszcxx_scatter_naive(
    compact_array1<T> in_outlier, array3<T> out_decomp_space,
    float* milliseconds, void* stream);

template <pszpolicy P, typename T, bool TIMING = true>
pszerror pszcxx_gather_make_metadata_host_available(
    compact_array1<T> in_outlier, void* stream);

// see viewer.cu_hip.hh
// template <pszpolicy P, typename T>
// pszerror pszcxx_evaluate_quality(array3<T> d1, array3<T> d2);

};  // namespace _2401

#endif /* E85FA9DC_9350_4F91_B4D8_5C094661479E */
