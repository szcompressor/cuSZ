
#include "module/cxx_module.hh"
#include "scatter_cxx.cuhip.inl"

#define INS_SCATTER_2401(BACKEND, T, TIMING)                                 \
  template pszerror _2401::pszcxx_scatter_naive<BACKEND, T, TIMING>(         \
      compact_array1<T> in, array3<T> out, f4 * milliseconds, void* stream); \
  template pszerror                                                          \
  _2401::pszcxx_gather_make_metadata_host_available<BACKEND, T, TIMING>(     \
      compact_array1<T> in, void* stream);

INS_SCATTER_2401(CUDA, f4, true)
INS_SCATTER_2401(CUDA, f4, false)
// INS_SCATTER_2401(CUDA, f8)

#undef INS_SCATTER_2401