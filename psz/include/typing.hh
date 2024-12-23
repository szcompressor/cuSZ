/**
 * @file typing.hh
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

#ifndef PSZ_TYPING_HH
#define PSZ_TYPING_HH

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "cusz/type.h"
#include "cxx_typing.h"

/**
 * @brief CUDA API does not accept u8 (understandable by literal), but instead,
 * `unsigned long long`, which is ambiguous anyway.
 */
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

#endif
