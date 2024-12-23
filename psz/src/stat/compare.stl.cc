#include "detail/compare.stl.inl"

#define __INSTANTIATE_CPPSTL_COMPARE(T)                         \
  template bool psz::cppstl::CPU_error_bounded(                 \
      T* a, T* b, size_t const len, double const eb,            \
      size_t* first_faulty_idx);                                \
  template void psz::cppstl::CPU_assess_quality(                \
      psz_statistics* s, T* xdata, T* odata, size_t const len); \
  template void psz::cppstl::CPU_extrema<T>(T * in, szt const len, T res[4]);

__INSTANTIATE_CPPSTL_COMPARE(f4)
__INSTANTIATE_CPPSTL_COMPARE(f8)
