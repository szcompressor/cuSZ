/**
 * @file compact.hh
 * @author Jiannan Tian
 * @brief Data structure for stream compaction.
 * @version 0.4
 * @date 2023-01-23
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef DAB40B13_9236_42A9_8047_49CD896671C9
#define DAB40B13_9236_42A9_8047_49CD896671C9

#include "cusz/type.h"

template <psz_policy Policy, typename T>
struct CompactDram;

#include "compact/compact.seq.hh"

template <>
struct CompactDram<SEQ, f4> {
  using Compact = CompactSerial<f4>;
};
template <>
struct CompactDram<SEQ, f8> {
  using Compact = CompactSerial<f8>;
};

#include "compact/compact.cu.hh"

template <>
struct CompactDram<CUDA, f4> {
  using Compact = psz::CompactGpuDram<f4>;
};
template <>
struct CompactDram<CUDA, f8> {
  using Compact = psz::CompactGpuDram<f8>;
};

#endif /* DAB40B13_9236_42A9_8047_49CD896671C9 */
