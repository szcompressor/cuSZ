#include "utils/viewer.hh"
#include "viewer.cuhip.inl"

#define __INSTANTIATE_CUHIP_VIEWER(T, P)                                                     \
  template void psz::analysis::GPU_evaluate_quality_and_print<T, P>(T*, T*, size_t, size_t); \
  template void psz::analysis::GPU_evaluate_quality_and_print_concise<T, P>(                 \
      T*, T*, size_t, size_t, psz_header*);

// __INSTANTIATE_CUHIP_VIEWER(float, THRUST_DPL)
__INSTANTIATE_CUHIP_VIEWER(float, CUDA)
// __INSTANTIATE_CUHIP_VIEWER(double, THRUST_DPL)
__INSTANTIATE_CUHIP_VIEWER(double, CUDA)
