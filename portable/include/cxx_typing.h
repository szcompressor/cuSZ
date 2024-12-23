/* Modified from typing.hh in cuSZ/pSZ; type traits. */

#ifndef _PORTABLE_CXX_TYPING_H
#define _PORTABLE_CXX_TYPING_H

#include <cstdint>
#include <type_traits>

#include "c_type.h"

namespace _portable {

// clang-format off

/* Compatible with CUDA. */
template <typename T> struct CudaCompat;
template <> struct CudaCompat<u4> { using type = u4; };
template <> struct CudaCompat<u8> { using type = ull; };

template <bool LARGE> struct LargeInputTrait;
template <> struct LargeInputTrait<false> { using type = u4; };
template <> struct LargeInputTrait<true>  { using type = u8; };

template <bool FAST> struct FastLowPrecisionTrait;
template <> struct FastLowPrecisionTrait<true>  { typedef f4 type; };
template <> struct FastLowPrecisionTrait<false> { typedef f8 type; };

template <_portable_dtype T> struct Ctype;
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
template <> struct Ctype<ULL> { typedef ull type; static const int width = sizeof(ull); };

template <typename Ctype> struct TypeSym;
template <> struct TypeSym<f4> { static const _portable_dtype type = F4; static const int width = sizeof(f4); };
template <> struct TypeSym<f8> { static const _portable_dtype type = F8; static const int width = sizeof(f8); };
template <> struct TypeSym<i1> { static const _portable_dtype type = I1; static const int width = sizeof(i1); };
template <> struct TypeSym<i2> { static const _portable_dtype type = I2; static const int width = sizeof(i2); };
template <> struct TypeSym<i4> { static const _portable_dtype type = I4; static const int width = sizeof(i4); };
template <> struct TypeSym<i8> { static const _portable_dtype type = I8; static const int width = sizeof(i8); };
template <> struct TypeSym<u1> { static const _portable_dtype type = U1; static const int width = sizeof(u1); };
template <> struct TypeSym<u2> { static const _portable_dtype type = U2; static const int width = sizeof(u2); };
template <> struct TypeSym<u4> { static const _portable_dtype type = U4; static const int width = sizeof(u4); };
template <> struct TypeSym<u8> { static const _portable_dtype type = U8; static const int width = sizeof(u8); };
template <> struct TypeSym<ull> { static const _portable_dtype type = ULL; static const int width = sizeof(ull); };
// clang-format on

}  // namespace _portable

#endif
