#ifndef D2F48D60_CCE7_4049_8A56_2ADDF140192E
#define D2F48D60_CCE7_4049_8A56_2ADDF140192E

#include <cstddef>
#include <cstdint>

#include "cusz/type.h"
#include "module/cxx_module.hh"
#include "port.hh"

template <typename T>
uint32_t count_outlier(T* in, size_t inlen, int radius, void* stream);

template <psz_policy policy, typename T>
void hist(
    bool optim, T* whole_numbers, size_t const len, uint32_t* hist,
    size_t const bklen, float* t, cudaStream_t stream)
{
  if (optim)
    pszcxx_compat_histogram_cauchy<policy, T>(
        whole_numbers, len, hist, bklen, t, stream);
  else
    pszcxx_compat_histogram_generic<policy, T>(
        whole_numbers, len, hist, bklen, t, stream);
}

#endif /* D2F48D60_CCE7_4049_8A56_2ADDF140192E */
