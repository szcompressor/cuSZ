
#include "module/cxx_module.hh"
#include "scatter_cxx.cu_hip.inl"

#define INS_SCATTER_2401(BACKEND, T)                                \
  template pszerror _2401::pszcxx_scatter_naive<BACKEND, T>(        \
      pszcompact_cxx<T> in, pszarray_cxx<T> out, f4 * milliseconds, \
      void* stream);                                                \
  template pszerror                                                 \
  _2401::pszcxx_gather_make_metadata_host_available<BACKEND, T>(    \
      pszcompact_cxx<T> in, void* stream);

INS_SCATTER_2401(CUDA, f4)
// INS_SCATTER_2401(CUDA, f8)

#undef INS_SCATTER_2401