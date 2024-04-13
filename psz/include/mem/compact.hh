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

template <pszpolicy Policy, typename T>
struct CompactDram;

#include "compact/compact.seq.hh"

// [psz::TODO] pass compilation with host compiler only

// clang-format off
template <> struct CompactDram<SEQ, f4> {
  using Compact = CompactSerial<f4>; };
template <> struct CompactDram<SEQ, f8> {
  using Compact = CompactSerial<f8>; };
// clang-format on

#if defined(PSZ_USE_CUDA)

#include "compact/compact.cu.hh"

// clang-format off
template <> struct CompactDram<CUDA, f4> {
  using Compact = psz::detail::cuda::CompactGpuDram<f4>; };
template <> struct CompactDram<CUDA, f8> {
  using Compact = psz::detail::cuda::CompactGpuDram<f8>; };
// clang-format on

#elif defined(PSZ_USE_HIP)

#include "compact/compact.hip.hh"
// clang-format off
template <> struct CompactDram<HIP, f4> {
  using Compact = psz::detail::hip::CompactGpuDram<f4>; };
template <> struct CompactDram<HIP, f8> {
  using Compact = psz::detail::hip::CompactGpuDram<f8>; };
// clang-format on

#elif defined(PSZ_USE_1API)

#include "compact/compact.dp.hh"
// clang-format off
template <> struct CompactDram<ONEAPI, f4> {
  using Compact = psz::detail::dpcpp::CompactGpuDram<f4>; };
template <> struct CompactDram<ONEAPI, f8> {
  using Compact = psz::detail::dpcpp::CompactGpuDram<f8>; };
// clang-format on

#endif

#endif /* DAB40B13_9236_42A9_8047_49CD896671C9 */
