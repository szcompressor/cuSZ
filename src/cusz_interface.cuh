#ifndef CUSZ_INTERFACE_CUH
#define CUSZ_INTERFACE_CUH

/**
 * @file cusz_workflow.cuh
 * @author Jiannan Tian
 * @brief Workflow of cuSZ (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on: 2020-02-12
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <iostream>
#include "argparse.hh"
#include "datapack.hh"
#include "type_trait.hh"

using namespace std;

typedef std::tuple<size_t, size_t, size_t, bool> tuple_3ul_1bool;
namespace cusz {
namespace interface {

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void Compress(
    argpack*,
    struct DataPack<typename DataTrait<If_FP, DataByte>::Data>*,
    int&,
    size_t&,
    size_t&,
    size_t&,
    bool&);

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void Decompress(argpack*, int&, size_t&, size_t&, size_t&, bool);

}  // namespace interface

namespace impl {

template <typename Data, typename Quant>
void PdQ(Data*, Quant*, size_t*, double*);

template <typename Data, typename Quant>
void ReversedPdQ(Data*, Quant*, Data*, size_t*, double);

template <typename Data, typename Quant>
void VerifyHuffman(string const&, size_t, Quant*, int, size_t*, double*);

}  // namespace impl

}  // namespace cusz

#endif
