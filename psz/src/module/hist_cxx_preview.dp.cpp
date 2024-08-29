#include "module/cxx_module.hh"

#define SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY(T)                           \
  template pszerror _2401::pszcxx_histogram_cauchy<psz_policy::DPCPP, T>( \
      array3<T> in, array3<u4> out_hist, float* milliseconds, void* stream);

SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY(u1)
SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY(u2)
SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY(u4)

#undef SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY
