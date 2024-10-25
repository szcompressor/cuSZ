
#ifndef E85FA9DC_9350_4F91_B4D8_5C094661479E
#define E85FA9DC_9350_4F91_B4D8_5C094661479E

#include "kernel/lrz.hh"
#include "mem/array_cxx.h"
#include "mem/compact.hh"

using portable::array3;
using portable::compact_array1;

template <psz_policy Poilicy, typename T>
pszerror pszcxx_compat_histogram_generic(
    T* in, size_t const inlen, uint32_t* out_hist, int const outlen,
    float* milliseconds, void* stream = nullptr);

template <psz_policy Policy, typename T, typename FQ = uint32_t>
int pszcxx_compat_histogram_cauchy(
    T* in, uint32_t inlen, FQ* out_hist, uint32_t outlen, float* milliseconds,
    void* stream = nullptr);

template <psz_policy Policy, typename T, bool TIMING = true>
[[deprecated("array3 will be replaced.")]] pszerror pszcxx_histogram_cauchy(
    array3<T> in, array3<u4> out_hist, float* milliseconds, void* stream);

template <psz_policy P, typename T, bool TIMING = true>
[[deprecated("array3 will be replaced.")]] pszerror pszcxx_scatter_naive(
    compact_array1<T> in_outlier, array3<T> out_decomp_space,
    float* milliseconds, void* stream);

template <psz_policy P, typename T, bool TIMING = true>
[[deprecated("array3 will be replaced.")]] pszerror
pszcxx_gather_make_metadata_host_available(
    compact_array1<T> in_outlier, void* stream);

#endif /* E85FA9DC_9350_4F91_B4D8_5C094661479E */
