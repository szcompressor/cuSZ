#ifndef PSZ_DETAIL_COMPOSITE_HH
#define PSZ_DETAIL_COMPOSITE_HH

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#include "cusz/type.h"
#include "cxx_typing.h"
#include "mem/sp_interface.h"
#include "zigzag.hh"

// absorbed from typing.hh
template <typename T>
using cuszCOMPAT = _portable::CudaCompat<T>;

template <bool LARGE>
using LargeInputTrait = _portable::LargeInputTrait<LARGE>;

template <bool FAST>
using FastLowPrecisionTrait = _portable::FastLowPrecisionTrait<FAST>;

template <psz_dtype T>
using Ctype = _portable::Ctype<T>;

template <typename Ctype>
using PszType = _portable::TypeSym<Ctype>;

namespace psz {

enum class Toggle {
  ZigZagEnabled,
  StatLocalEnabled,
  StatGlobalEnabled,
  TopKHistEnabled,
  FutureEIPEnabled,
  QuantGroupingEnabled,
  //
  ZigZagDisabled,
  StatLocalDisabled,
  StatGlobalDisabled,
  TopKHistDisabled,
  FutureEIPDisabled,
  QuantGroupingDisabled,
};

template <
    uint16_t _TileDim, uint8_t _Seq,  // required
    uint8_t _TileDimY = (uint8_t)_TileDim, uint8_t _SeqY = _Seq,
    uint8_t _TileDimZ = (uint8_t)_TileDim, uint8_t _SeqZ = _Seq>
struct PredPerf {
  static const uint16_t TileDim = _TileDim;
  static const uint16_t TiledimX = TileDim;
  static const uint8_t TiledimY = _TileDimY;
  static const uint8_t TiledimZ = _TileDimZ;

  static const uint8_t Seq = _Seq;
  static const uint8_t SeqX = Seq;
  static const uint8_t SeqY = _SeqY;
  static const uint8_t SeqZ = _SeqZ;

  static_assert(Seq < 16, "Sequentiality must be less than 16.");
};

template <
    Toggle _UseZigZag,  //
    Toggle _UseStatLocal = Toggle::StatLocalDisabled,
    Toggle _UseStatGlobal = Toggle::StatGlobalDisabled,
    Toggle _UseQuantGrouping = Toggle::QuantGroupingDisabled,
    Toggle _UseFutureEIP = Toggle::FutureEIPDisabled>
struct PredFunc {
  static constexpr Toggle UseZigZag = _UseZigZag;
  static constexpr Toggle UseStatLocal = _UseStatLocal;
  static constexpr Toggle UseStatGlobal = _UseStatGlobal;
  static constexpr Toggle UseQuantGrouping = _UseQuantGrouping;
  static constexpr Toggle UseFutureEIP = _UseFutureEIP;

  static constexpr bool stat_local_disabled = UseStatLocal == Toggle::StatLocalDisabled;
  static constexpr bool stat_global_enabled = UseStatGlobal == Toggle::StatGlobalEnabled;
  static_assert(
      not(stat_local_disabled and stat_global_enabled),
      "UseLocalStat must be enalbed when UseGlobalStat is enabled.");
};

template <typename BaseT, typename PF, typename _Eq = uint16_t, typename _Fp = BaseT>
struct PredConfig {
  static constexpr Toggle UseZigZag = PF::UseZigZag;
  static constexpr Toggle UseStatLocal = PF::UseStatLocal;
  static constexpr Toggle UseStatGlobal = PF::UseStatGlobal;
  static constexpr Toggle UseQuantGrouping = PF::UseQuantGrouping;
  static constexpr Toggle UseFutureEIP = PF::UseFutureEIP;
#define GradientGrouping QuantGrouping

  /* typing */
  using Eq = _Eq;
  using Fp = _Fp;
  using Metadata = uint32_t;
  using M = Metadata;

  /* sparse parts */
  using CompactVal = BaseT;
  using CompactIdx = uint32_t;
  using CompactNum = uint32_t;
  using CV = CompactVal;
  using CI = CompactIdx;
  using CN = CompactNum;

  using Compact2 = _portable::compact_GPU_DRAM2<CompactVal, M>;
  using C2VI = _portable::compact_cell<CompactVal, M>;

  /* ZigZag setup */
  using ZigZag = psz::ZigZag<Eq>;
  using EqUInt = typename ZigZag::UInt;
  using EqSInt = typename ZigZag::SInt;
};

}  // namespace psz

#endif /* PSZ_DETAIL_COMPOSITE_HH */
