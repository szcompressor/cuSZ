#include <hip/hip_runtime.h>

#include "cxx_lrz.cu_hip.inl"

#define C_INS(T, Eq, ZIGZAG)                                      \
  template pszerror _2401::pszcxx_predict_lorenzo<T, Eq, ZIGZAG>( \
      array3<T>*, pszrc2 const, array3<Eq>*, compact_array1<T>*, void*);

#define X_INS(T, Eq)                                              \
  template pszerror _2401::pszcxx_reverse_predict_lorenzo<T, Eq>( \
      array3<Eq>*, array3<T>*, pszrc2 const, array3<T>*, void*);

C_INS(f4, u4, false)
C_INS(f4, u4, true)
C_INS(f8, u4, false)
C_INS(f8, u4, true)

X_INS(f4, u4);
X_INS(f8, f4);

#undef C_INS
#undef X_INS