#ifndef PSZ_INCOMP_RECOMPUTE_CUH
#define PSZ_INCOMP_RECOMPUTE_CUH

#include <cuda_runtime.h>

#include "c_type.h"
#include "hfr-pbk.hh"

namespace psz::incomp_redo {

__device__ __forceinline__ f4 lorenzo_1d(f4 const* in, size_t gid, f4 ebx2_r, u2 radius)
{
  f4 q = round(in[gid] * ebx2_r);
  f4 qw = ((gid % 1024u) > 0) ? round(in[gid - 1] * ebx2_r) : 0.f;
  return (f4)((q - qw) + radius);
}

__device__ __forceinline__ f4 lorenzo_2d(f4 const* in, size_t gid, f4 ebx2_r, u2 radius, u4 leapy)
{
  u4 gix = (u4)(gid % leapy), giy = (u4)(gid / leapy);
  u4 tx = gix % 32u, ty = giy % 32u;
  f4 q = round(in[gid] * ebx2_r);
  f4 qn = (ty > 0) ? round(in[gid - leapy] * ebx2_r) : 0.f;
  f4 dy = q - qn;
  if (tx > 0) {
    f4 qw = round(in[gid - 1] * ebx2_r);
    f4 qnw = (ty > 0) ? round(in[gid - leapy - 1] * ebx2_r) : 0.f;
    dy -= (qw - qnw);
  }
  return (f4)(dy + radius);
}

__device__ __forceinline__ f4
lorenzo_3d(f4 const* in, size_t gid, f4 ebx2_r, u2 radius, u4 leapy, u4 leapz)
{
  u4 gix = (u4)(gid % leapy);
  u4 giz = (u4)(gid / leapz);
  u4 giy = (u4)((gid % leapz) / leapy);
  u4 tx = gix % 8u, ty = giy % 8u, tz = giz % 8u;

  // A(g) for z-difference: q(g) - [tz>0] q(g - leapz)
  auto A = [&](size_t g) -> f4 {
    f4 q = round(in[g] * ebx2_r);
    return q - ((tz > 0) ? round(in[g - leapz] * ebx2_r) : 0.f);
  };
  // x-difference (B), then y-difference (delta)
  f4 B = A(gid);
  if (tx > 0) B -= A(gid - 1);
  f4 delta = B;
  if (ty > 0) {
    f4 By = A(gid - leapy);
    if (tx > 0) By -= A(gid - leapy - 1);
    delta = B - By;
  }
  return (f4)(delta + radius);
}

__device__ __forceinline__ f4 dispatch(IncompRedo const& rc, size_t gid)
{
  switch (rc.kind) {
    case IncompPredKind::Lorenzo1D:
      return lorenzo_1d((f4 const*)rc.in_data, gid, rc.ebx2_r, rc.radius);
    case IncompPredKind::Lorenzo2D:
      return lorenzo_2d((f4 const*)rc.in_data, gid, rc.ebx2_r, rc.radius, rc.leapy);
    case IncompPredKind::Lorenzo3D:
      return lorenzo_3d((f4 const*)rc.in_data, gid, rc.ebx2_r, rc.radius, rc.leapy, rc.leapz);
    default: return (f4)0;
  }
}

}  // namespace psz::incomp_redo

#endif  // PSZ_INCOMP_RECOMPUTE_CUH
