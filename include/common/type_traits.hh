/**
 * @file type_traits.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-23
 * (create) 2020-09-23, (rev) 2021-09-17
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef TYPE_TRAITS_HH
#define TYPE_TRAITS_HH

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "cusz/type.h"
#include "definition.hh"

template <typename T>
cusz_datatype cusz_typeof()
{
    if (std::is_same<T, float>::value)
        return FP32;
    else if (std::is_same<T, double>::value)
        return FP64;
    else
        throw std::runtime_error("Type not supported.");
}

// clang-format off

/**
 * @brief CUDA API does not accept uint64_t (understandable by literal), but instead, 
 * `unsigned long long`, which is ambiguous anyway.
 */
template <typename T> struct cuszCOMPAT;
template <> struct cuszCOMPAT<uint32_t> { using type = uint32_t; };
template <> struct cuszCOMPAT<uint64_t> { using type = unsigned long long; };

template <int WIDTH, bool FP = true> struct DataTrait;
template <> struct DataTrait<4, true>  { typedef float   type; };
template <> struct DataTrait<8, true>  { typedef double  type; };
template <> struct DataTrait<1, false> { typedef int8_t  type; }; // future use
template <> struct DataTrait<2, false> { typedef int16_t type; }; // future use
template <> struct DataTrait<4, false> { typedef int32_t type; }; // future use
template <> struct DataTrait<8, false> { typedef int64_t type; }; // future use

template <int NDIM> struct ChunkingTrait;
template <> struct ChunkingTrait<1>     { static const int BLOCK = 256; static const int SEQ = 8; };
template <> struct ChunkingTrait<0x101> { static const int BLOCK = 128; };
template <> struct ChunkingTrait<0x201> { static const int BLOCK = 64;  };
template <> struct ChunkingTrait<2>     { static const int BLOCK = 16; static const int YSEQ = 8; };
template <> struct ChunkingTrait<3>     { static const int BLOCK = 8;  static const int YSEQ = 8; };

// template <int WIDTH> struct QuantTrait;
// template <> struct QuantTrait<1> { typedef uint8_t type; };
// template <> struct QuantTrait<2> { typedef uint16_t type; };
// template <> struct QuantTrait<4> { typedef uint32_t type; };

template <int WIDTH, bool FP = false> struct ErrCtrlTrait;
template <> struct ErrCtrlTrait<1, false> { typedef uint8_t  type; };
template <> struct ErrCtrlTrait<2, false> { typedef uint16_t type; };
template <> struct ErrCtrlTrait<4, false> { typedef uint32_t type; };
template <> struct ErrCtrlTrait<4, true>  { typedef float    type; };
template <> struct ErrCtrlTrait<8, true>  { typedef double   type; };

template <int WIDTH> struct HuffTrait;
template <> struct HuffTrait<4> { typedef cuszCOMPAT<uint32_t>::type type; };
template <> struct HuffTrait<8> { typedef cuszCOMPAT<uint64_t>::type type; };

template <int WIDTH> struct ReducerTrait;
template <> struct ReducerTrait<4> { typedef uint32_t type; };
template <> struct ReducerTrait<8> { typedef uint64_t type; };

template <int WIDTH> struct MetadataTrait;
template <> struct MetadataTrait<4> { typedef uint32_t type; };
template <> struct MetadataTrait<8> { typedef uint64_t type; }; // size_t is problematic; do not use

template <bool LARGE> struct LargeInputTrait;
template <> struct LargeInputTrait<false> { using type = MetadataTrait<4>::type; };
template <> struct LargeInputTrait<true>  { using type = MetadataTrait<8>::type; };

template <bool FAST> struct FastLowPrecisionTrait;
template <> struct FastLowPrecisionTrait<true>  { typedef float  type; };
template <> struct FastLowPrecisionTrait<false> { typedef double type; };

template <psz_dtype T> struct Ctype;
template <> struct Ctype<F4> { typedef float    type; static const int width = sizeof(float);    };
template <> struct Ctype<F8> { typedef double   type; static const int width = sizeof(double);   };
template <> struct Ctype<I1> { typedef int8_t   type; static const int width = sizeof(int8_t);   };
template <> struct Ctype<I2> { typedef int16_t  type; static const int width = sizeof(int16_t);  };
template <> struct Ctype<I4> { typedef int32_t  type; static const int width = sizeof(int32_t);  };
template <> struct Ctype<I8> { typedef int64_t  type; static const int width = sizeof(int64_t);  };
template <> struct Ctype<U1> { typedef uint8_t  type; static const int width = sizeof(uint8_t);  };
template <> struct Ctype<U2> { typedef uint16_t type; static const int width = sizeof(uint16_t); };
template <> struct Ctype<U4> { typedef uint32_t type; static const int width = sizeof(uint32_t); };
template <> struct Ctype<U8> { typedef uint64_t type; static const int width = sizeof(uint64_t); };
template <> struct Ctype<ULL>{ typedef unsigned long long type; static const int width = sizeof(unsigned long long); };

template <typename Ctype> struct PszType;
template <> struct PszType<float>    { static const psz_dtype type = F4; static const int width = sizeof(float);    };
template <> struct PszType<double>   { static const psz_dtype type = F8; static const int width = sizeof(double);   };
template <> struct PszType<int8_t>   { static const psz_dtype type = I1; static const int width = sizeof(int8_t);   };
template <> struct PszType<int16_t>  { static const psz_dtype type = I2; static const int width = sizeof(int16_t);  };
template <> struct PszType<int32_t>  { static const psz_dtype type = I4; static const int width = sizeof(int32_t);  };
template <> struct PszType<int64_t>  { static const psz_dtype type = I8; static const int width = sizeof(int64_t);  };
template <> struct PszType<uint8_t>  { static const psz_dtype type = U1; static const int width = sizeof(uint8_t);  };
template <> struct PszType<uint16_t> { static const psz_dtype type = U2; static const int width = sizeof(uint16_t); };
template <> struct PszType<uint32_t> { static const psz_dtype type = U4; static const int width = sizeof(uint32_t); };
template <> struct PszType<uint64_t> { static const psz_dtype type = U8; static const int width = sizeof(uint64_t); };
template <> struct PszType<unsigned long long> { static const psz_dtype type = ULL; static const int width = sizeof(unsigned long long); };
// clang-format on

#endif
