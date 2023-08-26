#include "detail/compare_gpu.inl"
#include "stat/compare_thrust.hh"

#define THRUSTGPU_ASSESS(Tliteral, T) \
    template void psz::thrustgpu_assess_quality<T>(cusz_stats * s, T * xdata, T * odata, size_t const len);

THRUSTGPU_ASSESS(fp64, double);

#undef THRUSTGPU_ASSESS