#ifndef PHF_HFR_PBK_VER_HH
#define PHF_HFR_PBK_VER_HH

#include "hfr-pbk.hh"  // root: psz::HFR_PBK_Constants::ReduceTimes

// production reduce-merge / shuffle-merge (v7 only).
enum class RMerge { v7 };
enum class SMerge { v7 };

struct HFR_Opts {
  int reduce_times = (int)psz::HFR_PBK_Constants::ReduceTimes;
  RMerge rm = RMerge::v7;
  SMerge sm = SMerge::v7;
  int magnitude = (int)psz::HFR_PBK_Constants::Magnitude;  // 10 = 1Ki (default), 11 = 2Ki, 12 = 4Ki
  int blockdim = 128;  // 4Ki only: 128 (IterLog=2) or 256 (IterLog=1)
  psz::OutlierCell* block_outliers = nullptr;  // predictor-owned staging (Buf_Comp)
};

#if defined(__CUDACC__)
#define PHF_MERGE_HD __host__ __device__
#else
#define PHF_MERGE_HD
#endif

PHF_MERGE_HD constexpr bool merge_compatible(RMerge r, SMerge s)
{
  return r == RMerge::v7 and s == SMerge::v7;
}

#undef PHF_MERGE_HD

#endif  // PHF_HFR_PBK_VER_HH
