// deps
#include "cusz/type.h"
#include "kernel/lrz/lproto.hh"
// definitions
#include "detail/lproto.inl"

#define CPP_INS(T, Eq)                                                 \
  template pszerror psz_comp_lproto<T, Eq>(                            \
      T* const, dim3 const, double const, int const, Eq* const, void*, \
      float*, void*);                                                  \
                                                                       \
  template pszerror psz_decomp_lproto<T, Eq>(                          \
      Eq*, dim3 const, T*, double const, int const, T*, float*, void*);

// TODO decrease the number of instantiated types
CPP_INS(f4, u1);
CPP_INS(f4, u2);
CPP_INS(f4, u4);
// CPP_INS(f4, f4);
// CPP_INS(f4, i4);

CPP_INS(f8, u1);
CPP_INS(f8, u2);
CPP_INS(f8, u4);
// CPP_INS(f8, f4);
// CPP_INS(f8, int32_t);

#undef CPP_INS
