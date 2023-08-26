#include "detail/maxerr_thrust.inl"
#include "stat/compare_thrust.hh"

#define THRUSTGPU_ASSESS(Tliteral, T)           \
    template void psz::thrustgpu_get_maxerr<T>( \
        T * reconstructed, T * original, size_t len, T & maximum_val, size_t & maximum_loc, bool destructive);

THRUSTGPU_ASSESS(fp32, float);

#undef THRUSTGPU_ASSESS