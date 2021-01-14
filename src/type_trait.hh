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
template <> struct DataTrait<true, 4> {typedef FP4 Data;};
template <> struct DataTrait<true, 8> {typedef FP8 Data;};
//template <> struct DataTrait<false, 1> {typedef FP4 Data;}; // future use
//template <> struct DataTrait<false, 2> {typedef FP8 Data;};
//template <> struct DataTrait<false, 4> {typedef FP4 Data;};
//template <> struct DataTrait<false, 8> {typedef FP8 Data;};

template <int ndim> struct Index;
template <> struct Index<1> {typedef UInteger1 idx_t;};
template <> struct Index<2> {typedef UInteger2 idx_t;};
template <> struct Index<3> {typedef UInteger3 idx_t;};
template <> struct Index<4> {typedef UInteger4 idx_t;};


template <int QuantByte> struct QuantTrait;
template <> struct QuantTrait<1> {typedef UI1 Quant;};
template <> struct QuantTrait<2> {typedef UI2 Quant;};

template <int SymbolByte> struct CodebookTrait;
template <> struct CodebookTrait<4> {typedef UI4 Huff;};
template <> struct CodebookTrait<8> {typedef UI8 Huff;};

// TODO where there is atomicOps should be with static<unsigned long long int>(some_uint64_array)
template <int HuffByte> struct HuffTrait;
template <> struct HuffTrait<4> {typedef UI4 Huff;};
template <> struct HuffTrait<8> {typedef UI8 Huff;};
// clang-format on

struct HuffConfig {
    static const int Db_encode  = 256;
    static const int Db_deflate = 128;
};

#endif
