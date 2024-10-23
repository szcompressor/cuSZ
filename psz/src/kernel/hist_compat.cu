// deps
#include "cusz/type.h"
#include "port.hh"
// definitions
#include "detail/hist.cuhip.inl"
#include "module/cxx_module.hh"

#define SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(T)                 \
  template <>                                                    \
  pszerror pszcxx_compat_histogram_generic<psz_policy::CUDA, T>( \
      T * in_data, size_t const data_len, uint32_t* out_hist,    \
      int const hist_len, float* milliseconds, void* stream)     \
  {                                                              \
    return psz::cuhip::GPU_histogram_generic<T>(                 \
        in_data, data_len, out_hist, hist_len, milliseconds,     \
        (cudaStream_t)stream);                                   \
  }

SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(u1);
SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(u2);
SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(u4);
SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(f4);

#undef SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC
