#include "module/cxx_module.hh"

#define INS_HIST_CUDA_2401(T)                                             \
  template pszerror _2401::pszcxx_histogram_cauchy<psz_policy::DPCPP, T>( \
      array3<T> in, array3<u4> out_hist, float* milliseconds, void* stream);

INS_HIST_CUDA_2401(u1)
INS_HIST_CUDA_2401(u2)
INS_HIST_CUDA_2401(u4)

#undef INS_HIST_CUDA_2401
