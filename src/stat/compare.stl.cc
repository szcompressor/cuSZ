/**
 * @file compare.stl.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-09
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "detail/compare.stl.inl"

#include "cusz/type.h"

#define INIT_LOSSLESS(T) \
  template bool psz::cppstl_identical(T* d1, T* d2, size_t const len);

#define INIT_LOSSY(T)                                \
  template bool psz::cppstl_error_bounded(           \
      T* a, T* b, size_t const len, double const eb, \
      size_t* first_faulty_idx);                     \
  template void psz::cppstl_assess_quality(          \
      psz_summary* s, T* xdata, T* odata, size_t const len);

INIT_LOSSLESS(f4)
INIT_LOSSLESS(f8)
INIT_LOSSLESS(u1)
INIT_LOSSLESS(u2)
INIT_LOSSLESS(u4)

INIT_LOSSY(f4)
INIT_LOSSY(f8)

template void psz::cppstl_extrema<f4>(f4* in, szt const len, f4 res[4]);
template void psz::cppstl_extrema<f8>(f8* in, szt const len, f8 res[4]);

#undef INIT_LOSSLESS
#undef INIT_LOSSY
