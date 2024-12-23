#include "utils/viewer.hh"
#include "viewer.cuhip.inl"

#define __INSTANTIATE_CUHIP_VIEWER(T, P)                                   \
  template void pszcxx_evaluate_quality_gpu<T, P>(T*, T*, size_t, size_t); \
  template pszerror pszcxx_evaluate_quality_gpu<T, P>(array3<T>, array3<T>);

__INSTANTIATE_CUHIP_VIEWER(float, THRUST_DPL)
__INSTANTIATE_CUHIP_VIEWER(float, PROPER_RUNTIME)
__INSTANTIATE_CUHIP_VIEWER(double, THRUST_DPL)
__INSTANTIATE_CUHIP_VIEWER(double, PROPER_RUNTIME)