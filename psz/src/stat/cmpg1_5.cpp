#include "detail/extrema.thrust.inl"
#include "stat/compare.thrust.hh"

#define THRUSTGPU_DESCRIPTION(Tliteral, T) \
    template void psz::thrustgpu::thrustgpu_get_extrema_rawptr(T* d_ptr, size_t len, T res[4]);

THRUSTGPU_DESCRIPTION(fp64, double)

#undef THRUSTGPU_DESCRIPTION