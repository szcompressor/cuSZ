#include "detail/extrema_thrust.inl"
#include "stat/compare_thrust.hh"

#define THRUSTGPU_DESCRIPTION(Tliteral, T) \
    template void psz::thrustgpu_get_extrema_rawptr(T* d_ptr, size_t len, T res[4]);

THRUSTGPU_DESCRIPTION(fp64, double)

#undef THRUSTGPU_DESCRIPTION