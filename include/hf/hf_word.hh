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
#include <tuple>

template <int WIDTH>
struct PackedWordByWidth;

#ifndef PSZ_RESEARCH_HUFFBK_CUDA

template <>
struct PackedWordByWidth<4> {
  static const int field_word = 27;
  static const int field_bits = 5;
  uint32_t word : 27;
  uint32_t bits : 5;
};

template <>
struct PackedWordByWidth<8> {
  static const int field_word = 58;
  static const int field_bits = 6;
  uint64_t word : 58;
  uint64_t bits : 6;
};

#else

template <>
struct PackedWordByWidth<4> {
  static const int field_word = 24;
  static const int field_bits = 8;
  uint32_t word : 24;
  uint32_t bits : 8;
};

template <>
struct PackedWordByWidth<8> {
  static const int field_word = 56;
  static const int field_bits = 8;
  uint64_t word : 56;
  uint64_t bits : 8;
};

#endif

#endif /* FDF0862D_B4A6_4E38_A11B_7299C37315A9 */
