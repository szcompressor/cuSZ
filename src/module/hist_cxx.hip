#include "hist_cxx.cu_hip.inl"
#include "module/cxx_module.hh"

#define INS_HIST_CUDA_2401(T)                                             \
  template pszerror _2401::pszcxx_histogram_cauchy<pszpolicy::HIP, T>(    \
      pszarray_cxx<T> in, pszarray_cxx<u4> out_hist, float* milliseconds, \
      void* stream);

INS_HIST_CUDA_2401(u1)
INS_HIST_CUDA_2401(u2)
INS_HIST_CUDA_2401(u4)

#undef INS_HIST_CUDA_2401
