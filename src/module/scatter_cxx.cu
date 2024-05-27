
#include "module/cxx_module.hh"
#include "scatter_cxx.cu_hip.inl"

#define INS_SCATTER_2401(BACKEND, T, TIMING)                             \
  template pszerror _2401::pszcxx_scatter_naive<BACKEND, T, TIMING>(     \
      pszcompact_cxx<T> in, pszarray_cxx<T> out, f4 * milliseconds,      \
      void* stream);                                                     \
  template pszerror                                                      \
  _2401::pszcxx_gather_make_metadata_host_available<BACKEND, T, TIMING>( \
      pszcompact_cxx<T> in, void* stream);

INS_SCATTER_2401(CUDA, f4, true)
INS_SCATTER_2401(CUDA, f4, false)
// INS_SCATTER_2401(CUDA, f8)

#undef INS_SCATTER_2401