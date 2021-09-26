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

#include "definition.hh"
#include "types.hh"

// clang-format off

/********************************************************************************/
/*  CUDA types for compatibility                                                 /
/********************************************************************************/
namespace cusz {

template <typename T> struct CUDA_COMPAT;
template <> struct CUDA_COMPAT<uint32_t> { using type = uint32_t; };
template <> struct CUDA_COMPAT<uint64_t> { using type = unsigned long long; };

}

template <bool FP, int DWIDTH> struct DataTrait;
//template <> struct DataTrait<true, 1> {typedef float  type;}; // placeholder for Cartesian expansion
//template <> struct DataTrait<true, 2> {typedef double type;};
template <> struct DataTrait<true, 4> { typedef float  type; };
template <> struct DataTrait<true, 8> { typedef double type; };
//template <> struct DataTrait<false, 1> {typedef float  type;}; // future use
//template <> struct DataTrait<false, 2> {typedef double type;};
//template <> struct DataTrait<false, 4> {typedef float  type;};
//template <> struct DataTrait<false, 8> {typedef double type;};

template <int ndim> struct MetadataTrait;
template <> struct MetadataTrait<1>     { static const int Block = 256; static const int Sequentiality = 8; };
template <> struct MetadataTrait<0x101> { static const int Block = 128; };
template <> struct MetadataTrait<0x201> { static const int Block = 64;  };
template <> struct MetadataTrait<2>     { static const int Block = 16; static const int YSequentiality = 8;};
template <> struct MetadataTrait<3>     { static const int Block = 8;  static const int YSequentiality = 8;};

template <int QWIDTH> struct QuantTrait;
template <> struct QuantTrait<1> { typedef uint8_t type; };
template <> struct QuantTrait<2> { typedef uint16_t type; };
template <> struct QuantTrait<4> { typedef uint32_t type; };

template <int EWIDTH> struct ErrCtrlTrait;
template <> struct ErrCtrlTrait<1> { typedef uint8_t type; };
template <> struct ErrCtrlTrait<2> { typedef uint16_t type; };
template <> struct ErrCtrlTrait<4> { typedef uint32_t type; };

// TODO where there is atomicOps should be with static<unsigned long long int>(some_uint64_array)

template <int HWIDTH> struct HuffTrait;
template <> struct HuffTrait<4> { typedef cusz::CUDA_COMPAT<uint32_t>::type type; };
template <> struct HuffTrait<8> { typedef cusz::CUDA_COMPAT<uint64_t>::type type; };

template <int RWIDTH> struct ReducerTrait;
template <> struct ReducerTrait<4> { typedef uint32_t type; };
template <> struct ReducerTrait<8> { typedef uint64_t type; };

template <int MWIDTH> struct MetadataTrait;
template <> struct MetadataTrait<4> { typedef uint32_t type; };
template <> struct MetadataTrait<8> { typedef uint64_t type; }; // size_t problem; do not use size_t

template <bool LARGE> struct LargeInputTrait;
template <> struct LargeInputTrait<false> { using type = MetadataTrait<4>::type; };
template <> struct LargeInputTrait<true>  { using type = MetadataTrait<8>::type; };

template <bool FAST> struct FastLowPrecisionTrait;
template <> struct FastLowPrecisionTrait<true>  { typedef float  type; };
template <> struct FastLowPrecisionTrait<false> { typedef double type; };

// sparsity rate is less that 5%
struct SparseMethodSetup { static const int factor = 20; };

// clang-format on

#endif
