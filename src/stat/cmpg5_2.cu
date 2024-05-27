// deps
#include "stat/compare/compare.thrust.hh"
#include "port.hh"
// definitions
#include "detail/maxerr.thrust.inl"

#define THRUSTGPU_ASSESS(Tliteral, T)           \
    template void psz::thrustgpu_get_maxerr<T>( \
        T * reconstructed, T * original, size_t len, T & maximum_val, size_t & maximum_loc, bool destructive);

THRUSTGPU_ASSESS(fp64, double);

#undef THRUSTGPU_ASSESS