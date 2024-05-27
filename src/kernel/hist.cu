// deps
#include "cusz/type.h"
#include "port.hh"
// definitions
#include "kernel/hist.hh"
#include "detail/hist.cu_hip.inl"

#define SPECIAL(T)                                                      \
  template <>                                                           \
  psz_error_status psz::histogram<pszpolicy::CUDA, T>(                 \
      T * in, size_t const inlen, uint32_t* out_hist, int const nbin,   \
      float* milliseconds, void* stream)                                \
  {                                                                     \
    return psz::cu_hip::hist_default<T>(                       \
        in, inlen, out_hist, nbin, milliseconds, (GpuStreamT)stream); \
  }

SPECIAL(u1);
SPECIAL(u2);
SPECIAL(u4);
SPECIAL(f4);

#undef SPECIAL
