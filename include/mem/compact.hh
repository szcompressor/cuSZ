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

template <typename T>
struct CompactSerial;

template <typename T>
struct CompactGpuDram;

#include "compact/compact.seq.hh"

#if defined(PSZ_USE_CUDA)
#include "compact/compact.cu.hh"
#elif defined(PSZ_USE_HIP)
#include "compact/compact.hip.hh"
#elif defined(PSZ_USE_1API)
// #include "compact/compact.1api.hh"
#endif

#endif /* DAB40B13_9236_42A9_8047_49CD896671C9 */
