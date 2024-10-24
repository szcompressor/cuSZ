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

template <typename T>
psz_dtype psz_typeof()
{
  if (std::is_same<T, f4>::value)
    return F4;
  else if (std::is_same<T, f8>::value)
    return F8;
  else
    throw std::runtime_error("Type not supported.");
}

// clang-format off

/**
 * @brief CUDA API does not accept u8 (understandable by literal), but instead, 
 * `unsigned long long`, which is ambiguous anyway.
 */
template <typename T> struct cuszCOMPAT;
template <> struct cuszCOMPAT<u4> { using type = u4; };
template <> struct cuszCOMPAT<u8> { using type = ull; };

template <bool LARGE> struct LargeInputTrait;
template <> struct LargeInputTrait<false> { using type = u4; };
template <> struct LargeInputTrait<true>  { using type = u8; };

template <bool FAST> struct FastLowPrecisionTrait;
template <> struct FastLowPrecisionTrait<true>  { typedef f4 type; };
template <> struct FastLowPrecisionTrait<false> { typedef f8 type; };

template <psz_dtype T> struct Ctype;
template <> struct Ctype<F4> { typedef f4 type; static const int width = sizeof(f4); };
template <> struct Ctype<F8> { typedef f8 type; static const int width = sizeof(f8); };
template <> struct Ctype<I1> { typedef i1 type; static const int width = sizeof(i1); };
template <> struct Ctype<I2> { typedef i2 type; static const int width = sizeof(i2); };
template <> struct Ctype<I4> { typedef i4 type; static const int width = sizeof(i4); };
template <> struct Ctype<I8> { typedef i8 type; static const int width = sizeof(i8); };
template <> struct Ctype<U1> { typedef u1 type; static const int width = sizeof(u1); };
template <> struct Ctype<U2> { typedef u2 type; static const int width = sizeof(u2); };
template <> struct Ctype<U4> { typedef u4 type; static const int width = sizeof(u4); };
template <> struct Ctype<U8> { typedef u8 type; static const int width = sizeof(u8); };
template <> struct Ctype<ULL>{ typedef ull type; static const int width = sizeof(ull); };

template <typename Ctype> struct PszType;
template <> struct PszType<f4> { static const psz_dtype type = F4; static const int width = sizeof(f4); };
template <> struct PszType<f8> { static const psz_dtype type = F8; static const int width = sizeof(f8); };
template <> struct PszType<i1> { static const psz_dtype type = I1; static const int width = sizeof(i1); };
template <> struct PszType<i2> { static const psz_dtype type = I2; static const int width = sizeof(i2); };
template <> struct PszType<i4> { static const psz_dtype type = I4; static const int width = sizeof(i4); };
template <> struct PszType<i8> { static const psz_dtype type = I8; static const int width = sizeof(i8); };
template <> struct PszType<u1> { static const psz_dtype type = U1; static const int width = sizeof(u1); };
template <> struct PszType<u2> { static const psz_dtype type = U2; static const int width = sizeof(u2); };
template <> struct PszType<u4> { static const psz_dtype type = U4; static const int width = sizeof(u4); };
template <> struct PszType<u8> { static const psz_dtype type = U8; static const int width = sizeof(u8); };
template <> struct PszType<ull> { static const psz_dtype type = ULL; static const int width = sizeof(ull); };

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
