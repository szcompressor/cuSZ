#include "hist_cxx.cu_hip.inl"
#include "module/cxx_module.hh"

#define INS_HIST_CUDA_2401(T, TIMING)                                     \
  template pszerror                                                       \
  _2401::pszcxx_histogram_cauchy<pszpolicy::CUDA, T, u4, TIMING>(         \
      pszarray_cxx<T> in, pszarray_cxx<u4> out_hist, float* milliseconds, \
      void* stream);

INS_HIST_CUDA_2401(u1, true)
INS_HIST_CUDA_2401(u2, true)
INS_HIST_CUDA_2401(u4, true)

INS_HIST_CUDA_2401(u1, false)
INS_HIST_CUDA_2401(u2, false)
INS_HIST_CUDA_2401(u4, false)

#undef INS_HIST_CUDA_2401
