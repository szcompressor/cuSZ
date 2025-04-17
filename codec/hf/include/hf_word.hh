/**
 * @file hf_word.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-16
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef FDF0862D_B4A6_4E38_A11B_7299C37315A9
#define FDF0862D_B4A6_4E38_A11B_7299C37315A9

#include <cstdint>

#include "c_type.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

template <int WIDTH>
struct HuffmanWord;

using PW4 = HuffmanWord<4>;
using PW8 = HuffmanWord<8>;

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// clang-format off
template <int WIDTH> constexpr int FIELD_CODE();
template <> constexpr int FIELD_CODE<4>() { return 27; }
template <> constexpr int FIELD_CODE<8>() { return 58; }
template <int WIDTH> constexpr int BITWIDTH() { return WIDTH * 8; }
template <int WIDTH> constexpr int FIELD_BITCOUNT() { return BITWIDTH<WIDTH>() - FIELD_CODE<WIDTH>(); }
template <int WIDTH> constexpr int OUTLIER_CUTOFF() { return FIELD_CODE<WIDTH>() + 1; }

namespace {
  template <int W> struct HFtype;
  template <> struct HFtype<4> { typedef u4 type; };
  template <> struct HFtype<8> { typedef u8 type; };
}
// clang-format on

// [psz::caveat] Direct access leads to misaligned GPU addr.
// MSB | log2(32)=5 | max: 27; var-len prefix-code, right aligned |
// MSB | log2(64)=6 | max: 58; var-len prefix-code, right aligned |
template <int WIDTH>
struct HuffmanWord {
  static constexpr int W = WIDTH;
  static constexpr int BITWIDTH = ::BITWIDTH<W>();
  static constexpr int FIELD_CODE = ::FIELD_CODE<W>();
  static constexpr int FIELD_BITCOUNT = ::FIELD_BITCOUNT<W>();
  static constexpr int OUTLIER_CUTOFF = ::OUTLIER_CUTOFF<W>();
  using Hf = typename HFtype<W>::type;

  Hf prefix_code : FIELD_CODE;   // low 27 (58 for u8) bits
  Hf bitcount : FIELD_BITCOUNT;  // high 5 (6 for u8) bits

  HuffmanWord(Hf _prefix_code, Hf _bitcount)
  {
    prefix_code = _prefix_code;
    bitcount = _bitcount;
  }

  Hf to_uint() { return *reinterpret_cast<Hf*>(this); }
};

// MSB | max: 27; var-len prefix-code, left aligned | optional log2(32)=5 |
// MSB | max: 58; var-len prefix-code, left aligned | optional log2(64)=6 |
template <int W>
struct HuffmanWordLeftAlign {
  uint32_t bitcount : HuffmanWord<4>::FIELD_BITCOUNT;  // low 5 bits
  uint32_t prefix_code : HuffmanWord<4>::FIELD_CODE;   // high 27 bits
};

template <int W>
void rightalign_to_leftalign(HuffmanWord<W>& in, HuffmanWordLeftAlign<W>& out)
{
  out.bitcount = in.bitcount;
  out.prefix_code = in.prefix_code << (HuffmanWord<W>::FIELD_CODE - in.bitcount);
}

#endif /* FDF0862D_B4A6_4E38_A11B_7299C37315A9 */
