// for lrz and spl
#ifndef PSZ_BLK_OUTLIER_FALLBACK_CUH
#define PSZ_BLK_OUTLIER_FALLBACK_CUH

#include <cuda_runtime.h>

#include "c_type.h"
#include "hfr-pbk.hh"

namespace psz {

using C = psz::HFR_PBK_Constants;  // FIXME: extract non-parameterized part
constexpr auto EncIdShift = (u4)(C::BitsMaxNumUnpred + C::BitsMaxNumBreaks);

template <typename Tuple, typename Val>
__device__ __forceinline__ void fb_overflow_global(
    Tuple dram_tup, u4* dram_cn, size_t max_allowed, Val candidate, u4 to_gid)
{
  if (not(dram_tup and dram_cn)) return;
  auto cur = atomicAdd(dram_cn, 1u);
  if (cur < max_allowed) dram_tup[cur] = {candidate, to_gid};
}

}  // namespace psz

#endif  // PSZ_BLK_OUTLIER_FALLBACK_CUH
