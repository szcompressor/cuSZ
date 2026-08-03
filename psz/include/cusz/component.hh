#ifndef PSZ_CUSZ_COMPONENT_HH
#define PSZ_CUSZ_COMPONENT_HH

#include <cstdint>

#include "kernel/zigzag.hh"
#include "mem/sp_interface.h"

namespace psz {

template <typename _T, typename _E>
struct Buf_Comp;


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

template <int _UseZigZag, int _UseH1GL = 0b00, int _UnpredIncomp = 0b00>
struct PredictorFeature {  // dtype-agnostic
  static constexpr int UseZigZag = _UseZigZag;  // 0b1 = zigzag encode

  // top-1 histogram: bit1 = H1G (global commit), bit0 = H1L (local count)
  // 10: illegal (H1G-on mandates H1L-on)
  static constexpr int UseH1GL = _UseH1GL;
  static_assert(UseH1GL != 0b10, "UseH1G-on mandates UseH1L-on.");

  // 00: Global off, Local off ("compatible" with old design)
  // 01: Global off, Local on  ("quick", future default)
  // 10: Global on,  Local off (impossible/illegal)
  // 11: Global on,  Local on  ("quick" + global spill)
  static constexpr int UnpredIncomp = _UnpredIncomp;
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
  using CVI = CompactValIdx;

  /* ZigZag setup */
  using ZigZag = psz::ZigZag<Eq>;
  using EqUInt = typename ZigZag::UInt;
  using EqSInt = typename ZigZag::SInt;
};

}  // namespace psz

#endif /* PSZ_CUSZ_COMPONENT_HH */
