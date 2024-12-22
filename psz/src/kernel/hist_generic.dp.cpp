// deps
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "cusz/type.h"
#include "port.hh"
// definitions
#include "detail/hist.dp.inl"
#include "module/cxx_module.hh"

#define SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(T)                           \
  template <>                                                              \
  psz_error_status pszcxx_histogram_generic<psz_policy::CUDA, T>(          \
      T * in, size_t const inlen, uint32_t* out_hist, int const nbin,      \
      float* milliseconds, void* stream)                                   \
  {                                                                        \
    return psz::cuda_hip_compat::histogram_generic<T>(                     \
        in, inlen, out_hist, nbin, milliseconds, (dpct::queue_ptr)stream); \
  }

SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(u1);
SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(u2);
SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(u4);
SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC(f4);

#undef SPECIALIZE_PSZCXX_COMPAT_HIST_GENERIC
