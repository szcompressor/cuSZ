// deps
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "cusz/type.h"
#include "port.hh"
// definitions
#include "detail/hist.dp.inl"
#include "kernel/hist.hh"

#define SPECIAL(T)                                                         \
  template <>                                                              \
  psz_error_status psz::histogram<pszpolicy::CUDA, T>(                    \
      T * in, size_t const inlen, uint32_t* out_hist, int const nbin,      \
      float* milliseconds, void* stream)                                   \
  {                                                                        \
    return psz::cuda_hip_compat::hist_default<T>(                          \
        in, inlen, out_hist, nbin, milliseconds, (dpct::queue_ptr)stream); \
  }

SPECIAL(u1);
SPECIAL(u2);
SPECIAL(u4);
SPECIAL(f4);

#undef SPECIAL
