#include "spl_cxx.cu_hip.inl"

#define INS(T)                                                            \
  template pszerror _2401::pszcxx_predict_spline<T>(                      \
      pszarray_cxx<T>, pszrc2 const, pszarray_cxx<u4>, pszcompact_cxx<T>, \
      pszarray_cxx<T>, float* time, void* stream);                        \
  template pszerror _2401::pszcxx_reverse_predict_spline<T>(              \
      pszarray_cxx<u4>, pszarray_cxx<T>, pszarray_cxx<T>, pszrc2 const,   \
      pszarray_cxx<T>, float* time, void* stream);

INS(f4)
INS(f8)

#undef INS
#undef SETUP
