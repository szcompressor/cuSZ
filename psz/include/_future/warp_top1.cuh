// TODO: port back to bleeding-edge (replaces COUNT_LOCAL_STAT macros).
#ifndef PSZ_WARP_TOP1_CUH
#define PSZ_WARP_TOP1_CUH

#include "c_type.h"

namespace psz {

__device__ __forceinline__ void warp_top1_count(bool is_top1, u4& thp_top1_count)
{
  unsigned mask = __ballot_sync(0xffffffff, (unsigned)is_top1);
  if ((threadIdx.x & 31) == 0) thp_top1_count += __popc(mask);
}

}  // namespace psz

#endif
