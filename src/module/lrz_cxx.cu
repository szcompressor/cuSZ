#include <cuda_runtime.h>

#include "lrz_cxx.cu_hip.inl"

#define INS(T)                                                            \
  template pszerror _2401::pszcxx_predict_lorenzo<T>(                     \
      pszarray_cxx<T>, pszrc2 const, pszarray_cxx<u4>, pszcompact_cxx<T>, \
      void*);                                                             \
  template pszerror _2401::pszcxx_reverse_predict_lorenzo(                \
      pszarray_cxx<u4> in_errquant, pszarray_cxx<T> in_scattered_outlier, \
      pszrc2 const rc, pszarray_cxx<T> out_reconstruct, void* stream);

INS(f4)
INS(f8)

#undef INS