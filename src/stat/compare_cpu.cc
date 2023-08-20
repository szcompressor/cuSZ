/**
 * @file _compare.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-09
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "detail/compare_cpu.inl"

#define INIT_LOSSLESS(T) \
  template bool psz::cppstd_identical(T* d1, T* d2, size_t const len);

#define INIT_LOSSY(T)                                \
  template bool psz::cppstd_error_bounded(           \
      T* a, T* b, size_t const len, double const eb, \
      size_t* first_faulty_idx);                     \
  template void psz::cppstd_assess_quality(          \
      cusz_stats* s, T* xdata, T* odata, size_t const len);

INIT_LOSSLESS(float)
INIT_LOSSLESS(double)
INIT_LOSSLESS(uint8_t)
INIT_LOSSLESS(uint16_t)
INIT_LOSSLESS(uint32_t)

INIT_LOSSY(float)
INIT_LOSSY(double)

#undef INIT_LOSSLESS
#undef INIT_LOSSY
