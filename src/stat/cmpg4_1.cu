#include "detail/compare.thrust.inl"
#include "stat/compare/compare.thrust.hh"

#define THRUSTGPU_ASSESS(Tliteral, T)             \
  template void psz::thrustgpu_assess_quality<T>( \
      psz_summary * s, T * xdata, T * odata, size_t const len);

THRUSTGPU_ASSESS(fp32, float);

#undef THRUSTGPU_ASSESS
