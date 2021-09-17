#ifndef TYPE_TRAIT_HH
#define TYPE_TRAIT_HH

/**
 * @file type_trait.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.1.1
 * @date 2020-09-23
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "type_aliasing.hh"
#include "types.hh"

// clang-format off

template <bool If_FP, int DataByte> struct DataTrait;
//template <> struct DataTrait<true, 1> {typedef FP4 Data;}; // placeholder for Cartesian expansion
//template <> struct DataTrait<true, 2> {typedef FP8 Data;};
template <> struct DataTrait<true, 4> { typedef FP4 Data; typedef FP4 type; };
template <> struct DataTrait<true, 8> { typedef FP8 Data; typedef FP8 type; };
//template <> struct DataTrait<false, 1> {typedef FP4 Data;}; // future use
//template <> struct DataTrait<false, 2> {typedef FP8 Data;};
//template <> struct DataTrait<false, 4> {typedef FP4 Data;};
//template <> struct DataTrait<false, 8> {typedef FP8 Data;};

template <int QuantByte> struct QuantTrait;
template <> struct QuantTrait<1> { typedef UI1 Quant; typedef UI1 type; };
template <> struct QuantTrait<2> { typedef UI2 Quant; typedef UI2 type; };
template <> struct QuantTrait<4> { typedef UI4 Quant; typedef UI4 type; };

template <int EWIDTH> struct ErrCtrlTrait;
template <> struct ErrCtrlTrait<1> { typedef UI1 type; };
template <> struct ErrCtrlTrait<2> { typedef UI2 type; };
template <> struct ErrCtrlTrait<4> { typedef UI4 type; };

// obsolete
// template <int SymbolByte> struct CodebookTrait;
// template <> struct CodebookTrait<4> { typedef UI4 Huff; typedef UI4 type; };
// template <> struct CodebookTrait<8> { typedef UI8 Huff; typedef UI8 type; };

// TODO where there is atomicOps should be with static<unsigned long long int>(some_uint64_array)

template <int HuffByte> struct HuffTrait;
template <> struct HuffTrait<4> { typedef UI4 Huff; typedef UI4 type; };
template <> struct HuffTrait<8> { typedef UI8 Huff; typedef UI8 type; };

template <int RWIDTH> struct ReducerTrait;
template <> struct ReducerTrait<4> { typedef UI4 type; };
template <> struct ReducerTrait<8> { typedef UI8 type; };

template <int MWIDTH> struct MetadataTrait;
template <> struct MetadataTrait<4> { typedef uint32_t type; };
template <> struct MetadataTrait<8> { typedef size_t   type; };

// clang-format on

struct HuffConfig {
    static const int Db_encode  = 256;
    static const int Db_deflate = 256;

    static const int enc_sequentiality = 4;  // empirical
    static const int deflate_constant  = 4;  // TODO -> deflate_chunk_constant
};

#endif
