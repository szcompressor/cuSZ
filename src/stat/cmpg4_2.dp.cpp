#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include "detail/compare.dpl.inl"
#include "stat/compare/compare.dpl.hh"

#define DPL_ASSESS(Tliteral, T) \
    template void psz::dpl_assess_quality<T>(psz_summary * s, T * xdata, T * odata, size_t const len);

// [psz::note] cannot pass sycl::aspects::fp64 check
// DPL_ASSESS(fp64, double);

#undef DPL_ASSESS