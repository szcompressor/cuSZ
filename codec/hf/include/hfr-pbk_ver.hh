#ifndef PHF_HFR_PBK_VER_HH
#define PHF_HFR_PBK_VER_HH

#include "hfr-pbk.hh"  // root: psz::HFR_PBK_Constants::ReduceTimes

struct HFR_Opts {
  int reduce_times = (int)psz::HFR_PBK_Constants::ReduceTimes;
  int magnitude = (int)psz::HFR_PBK_Constants::Magnitude;  // 10 = 1Ki (default), 11 = 2Ki, 12 = 4Ki
  int blockdim = 128;  // 4Ki only: 128 (IterLog=2) or 256 (IterLog=1)
  psz::OutlierCell* block_outliers = nullptr;  // predictor-owned staging (Buf_Comp)
};

#endif  // PHF_HFR_PBK_VER_HH
