// deps
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include "port.hh"
#include "stat/compare/compare.dpl.hh"
// definitions
#include "detail/maxerr.dpl.inl"

#define DPL_ASSESS(Tliteral, T) \
  template void psz::dpl_get_maxerr<T>(T*, T*, size_t, T&, size_t&, bool);

// [psz::note] cannot pass sycl::aspects::fp64 check
// DPL_ASSESS(fp64, double);

#undef DPL_ASSESS