#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include "detail/compare.dpl.inl"
#include "stat/compare/compare.dpl.hh"

#define DPL_ASSESS(Tliteral, T) \
    template void psz::dpl_assess_quality<T>(psz_summary * s, T * xdata, T * odata, size_t const len);

DPL_ASSESS(fp32, float);

#undef DPL_ASSESS
