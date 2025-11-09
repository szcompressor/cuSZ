/**
 * @file composite.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-23
 * (create) 2020-09-23, (rev) 2021-09-17
 *
 * @copyright (C) 2020 by Washington State University, Argonne National
 * Laboratory See LICENSE in top-level directory
 *
 */

#ifndef PSZ_DETAIL_COMPOSITE_HH
#define PSZ_DETAIL_COMPOSITE_HH

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#include "mem/sp_interface.h"

// clang-format off
namespace psz {

template <int ByteWidth> struct SInt;
template <> struct SInt<1> { using T =  int8_t; }; 
template <> struct SInt<2> { using T = int16_t; }; 
template <> struct SInt<4> { using T = int32_t; }; 
template <> struct SInt<8> { using T = int64_t; };
template <int ByteWidth> using SInt_t = typename SInt<ByteWidth>::T;

template <int ByteWidth> struct UInt;
template <> struct UInt<1> { using T =  uint8_t; }; 
template <> struct UInt<2> { using T = uint16_t; }; 
template <> struct UInt<4> { using T = uint32_t; }; 
template <> struct UInt<8> { using T = uint64_t; };
template <int ByteWidth> using UInt_t = typename UInt<ByteWidth>::T;

}
// clang-format on

namespace psz {

// ZigZag encoding, reference:
// https://lemire.me/blog/2022/11/25/making-all-your-integers-positive-with-zigzag-encoding/
template <typename T>
struct ZigZag {
 public:
  static constexpr int ByteWidth = sizeof(T);
  using UInt = psz::UInt_t<ByteWidth>;
  using SInt = psz::SInt_t<ByteWidth>;

 private:
  static constexpr int BitWidth = ByteWidth * 8;

 public:
  // force type checking to unsure the bitwidth
  template <typename _SUPPOSED_SINT>
  [[nodiscard]] static constexpr
      typename std::enable_if_t<std::is_same_v<_SUPPOSED_SINT, SInt>, UInt>
      encode(_SUPPOSED_SINT const x)
  {
    static_assert(
        std::is_same_v<_SUPPOSED_SINT, SInt>,
        "[ZigZag] encode() input must be a SIGNED integer, whose bitwidth is "
        "the same as T in ZigZag<T>.");
    return (x << 1) ^ (x >> (BitWidth - 1));
  }

  // force type checking to unsure the bitwidth
  template <typename _SUPPOSED_UINT>
  [[nodiscard]] static constexpr
      typename std::enable_if_t<std::is_same_v<_SUPPOSED_UINT, UInt>, SInt>
      decode(_SUPPOSED_UINT const x)
  {
    static_assert(
        std::is_same_v<_SUPPOSED_UINT, UInt>,
        "[ZigZag] decode() input must be an UNSIGNED integer, whose bitwidth "
        "is the same as T in ZigZag<T>.");
    return (x >> 1) ^ (-(x & 1));
  }
};

}  // namespace psz

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
