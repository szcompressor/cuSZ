// deps
#include <hip/hip_runtime.h>

#include "cusz/type.h"
#include "port.hh"
// definitions
#include "detail/hist_cu.inl"
#include "module/cxx_module.hh"

#define SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(T)                        \
  template <>                                                           \
  psz_error_status pszcxx_histogram_generic<psz_policy::HIP, T>(        \
      T * in, size_t const inlen, uint32_t* out_hist, int const nbin,   \
      float* milliseconds, void* stream)                                \
  {                                                                     \
    return psz::cu_hip::histogram_generic<T>(                           \
        in, inlen, out_hist, nbin, milliseconds, (cudaStream_t)stream); \
  }

// SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(u1);
// SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(u2);
SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(u4);
// SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(f4);

#undef SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC
