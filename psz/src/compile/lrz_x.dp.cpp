#include "kernel/lrz_x.dp.cpp"

#define INSTANTIATE_GPU_L23X_2params(T, Eq)                               \
  template pszerror psz::dpcpp::GPU_x_lorenzo_nd<T, Eq>(                  \
      Eq * eq, sycl::range<3> const len3, T* outlier, PROPER_EB const eb, \
      int const radius, T* xdata, f4* time_elapsed, void* stream);

#define INSTANTIATE_GPU_L23X_1param(T) \
  INSTANTIATE_GPU_L23X_2params(T, u2); \
  INSTANTIATE_GPU_L23X_2params(T, u4); \
  INSTANTIATE_GPU_L23X_2params(T, f4);

INSTANTIATE_GPU_L23X_1param(f4);
// f8 will fail consumer-grade GPUs

#undef CPP_INS
