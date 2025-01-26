#include "utils/viewer.hh"
#include "viewer.cuhip.inl"

#define __INSTANTIATE_CUHIP_VIEWER(T, P) \
  template void psz::analysis::GPU_evaluate_quality_and_print<T, P>(T*, T*, size_t, size_t);

// __INSTANTIATE_CUHIP_VIEWER(float, THRUST_DPL)
__INSTANTIATE_CUHIP_VIEWER(float, PROPER_RUNTIME)
// __INSTANTIATE_CUHIP_VIEWER(double, THRUST_DPL)
__INSTANTIATE_CUHIP_VIEWER(double, PROPER_RUNTIME)