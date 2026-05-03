#include "kernel/lrz_c.dp.cpp"

#define INSTANCIATE_GPU_L23R(T, Eq, ZigZag)                                           \
  template pszerror psz::dpcpp::GPU_c_lorenzo_nd_with_outlier<T, Eq, ZigZag>(         \
      T* const data, sycl::range<3> const len3, PROPER_EB const eb, int const radius, \
      Eq* const eq, void* _outlier, f4* time_elapsed, void* stream);

INSTANCIATE_GPU_L23R(f4, u4, false)

#undef INSTANCIATE_GPU_L23R