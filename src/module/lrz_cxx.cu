#include <cuda_runtime.h>

#include "lrz_cxx.cu_hip.inl"

#define INS(T, TIMING)                                                       \
  template pszerror _2401::pszcxx_predict_lorenzo<T, TIMING>(                \
      pszarray_cxx<T>, pszrc2 const, pszarray_cxx<u4>, pszcompact_cxx<T>,    \
      f4*, void*);                                                           \
  template pszerror _2401::pszcxx_reverse_predict_lorenzo<T, TIMING>(        \
      pszarray_cxx<u4>, pszarray_cxx<T>, pszrc2 const, pszarray_cxx<T>, f4*, \
      void*);

INS(f4, true)
INS(f8, true)
INS(f4, false)
INS(f8, false)

#undef INS