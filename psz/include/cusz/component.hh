#ifndef PSZ_CUSZ_COMPONENT_HH
#define PSZ_CUSZ_COMPONENT_HH

#include <cstdint>

#include "kernel/zigzag.hh"
#include "mem/sp_interface.h"

namespace psz {

template <typename _T, typename _E>
struct Buf_Comp;

enum class Toggle { ZigZag_On, ZigZag_Off, H1L_On, H1L_Off, H1G_On, H1G_Off };

template <
    uint16_t _TileDim, uint8_t _Seq,  // required
    uint8_t _TileDimY = (uint8_t)_TileDim, uint8_t _SeqY = _Seq,
    uint8_t _TileDimZ = (uint8_t)_TileDim, uint8_t _SeqZ = _Seq>
struct PredictorTile {
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

template <Toggle _UseZigZag, Toggle _UseH1L = Toggle::H1L_Off, Toggle _UseH1G = Toggle::H1G_Off>
struct PredictorFeature {  // dtype-agnostic
  static constexpr Toggle UseZigZag = _UseZigZag;
  static constexpr Toggle UseH1L = _UseH1L;
  static constexpr Toggle UseH1G = _UseH1G;

  static constexpr bool h1l_off = (UseH1L == Toggle::H1L_Off);
  static constexpr bool h1g_on = (UseH1G == Toggle::H1G_On);
  static_assert(not(h1l_off and h1g_on), "UseH1G-on mandates UseH1L-on.");
};

template <typename BaseT, typename _Eq = uint16_t, typename _Fp = BaseT>
struct PredictorTyping {
  /* typing */
  using T = BaseT;
  using Eq = _Eq;
  using Fp = _Fp;
  using Metadata = uint32_t;
  using M = Metadata;

  /* Buffer */
  using Buf_Comp = psz::Buf_Comp<BaseT, _Eq>;

  /* sparse parts */
  using CompactVal = BaseT;
  using CompactIdx = uint32_t;
  using CompactNum = uint32_t;
  using CV = CompactVal;
  using CI = CompactIdx;
  using CN = CompactNum;

  using Compact2 = _ptb::compact_GPU_DRAM2<CompactVal, M>;
  using CompactValIdx = _ptb::compact_cell<CompactVal, M>;

  /* ZigZag setup */
  using ZigZag = psz::ZigZag<Eq>;
  using EqUInt = typename ZigZag::UInt;
  using EqSInt = typename ZigZag::SInt;
};

}  // namespace psz

#endif /* PSZ_CUSZ_COMPONENT_HH */
