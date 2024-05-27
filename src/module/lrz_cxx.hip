#include <hip/hip_runtime.h>

#include "cxx_lrz.cu_hip.inl"

#define C_INS(T, Eq, ZIGZAG)                                                 \
  template pszerror _2401::pszcxx_predict_lorenzo<T, Eq, ZIGZAG>(            \
      pszarray_cxx<T>*, pszrc2 const, pszarray_cxx<Eq>*, pszcompact_cxx<T>*, \
      void*);

#define X_INS(T, Eq)                                                       \
  template pszerror _2401::pszcxx_reverse_predict_lorenzo<T, Eq>(          \
      pszarray_cxx<Eq>*, pszarray_cxx<T>*, pszrc2 const, pszarray_cxx<T>*, \
      void*);

C_INS(f4, u4, false)
C_INS(f4, u4, true)
C_INS(f8, u4, false)
C_INS(f8, u4, true)

X_INS(f4, u4);
X_INS(f8, f4);

#undef C_INS
#undef X_INS