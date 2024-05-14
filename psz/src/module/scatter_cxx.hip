
#include "module/cxx_module.hh"
#include "scatter_c xx.cu_hip.inl"

#define INS_SCATTER_2401(BACKEND, T)                         \
  template pszerror _2401::pszcxx_scatter_naive<BACKEND, T>( \
      compact_array1<T> in, array3<T> out, f4 * milliseconds, void* stream);

INS_SCATTER_2401(HIP, f4)
// INS_SCATTER_2401(HIP, f8)

#undef INS_SCATTER_2401