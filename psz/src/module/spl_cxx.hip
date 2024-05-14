#include "spl_cxx.cu_hip.inl"

#define INS(T)                                                                \
  template pszerror _2401::pszcxx_predict_spline<T>(                          \
      array3<T>, pszrc2 const, array3<u4>, compact_array1<T>, array3<T>,      \
      float* time, void* stream);                                             \
  template pszerror _2401::pszcxx_reverse_predict_spline<T>(                  \
      array3<u4>, array3<T>, array3<T>, pszrc2 const, array3<T>, float* time, \
      void* stream);

INS(f4)
INS(f8)

#undef INS
#undef SETUP
