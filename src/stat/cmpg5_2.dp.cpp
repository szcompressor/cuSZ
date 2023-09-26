// deps
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include "port.hh"
#include "stat/compare/compare.dpl.hh"
// definitions
#include "detail/maxerr.dpl.inl"

#define DPL_ASSESS(Tliteral, T)                                     \
  template void psz::dpl_get_maxerr<T>(                        \
      T * reconstructed, T * original, size_t len, T & maximum_val, \
      size_t & maximum_loc, bool destructive);

DPL_ASSESS(fp64, double);

#undef DPL_ASSESS