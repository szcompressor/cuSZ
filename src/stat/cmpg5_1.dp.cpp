// deps
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include "stat/compare/compare.dpl.hh"
#include "port.hh"
// definitions
#include "detail/maxerr.dpl.inl"

#define DPL_ASSESS(Tliteral, T)           \
    template void psz::dpl_get_maxerr<T>( \
        T * reconstructed, T * original, size_t len, T & maximum_val, size_t & maximum_loc, bool destructive);

DPL_ASSESS(fp32, float);

#undef DPL_ASSESS