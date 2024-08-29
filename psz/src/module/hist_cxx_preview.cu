#include "hist_cxx.cuhip.inl"
#include "module/cxx_module.hh"

#define SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY(T, TIMING)                       \
  template pszerror pszcxx_histogram_cauchy<psz_policy::CUDA, T, u4, TIMING>( \
      array3<T> in, array3<u4> out_hist, float* milliseconds, void* stream);

SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY(u1, true)
SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY(u2, true)
SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY(u4, true)

SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY(u1, false)
SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY(u2, false)
SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY(u4, false)

#undef SPECIALIZE_PSZCXX_MODULE_HIST_CAUCHY
