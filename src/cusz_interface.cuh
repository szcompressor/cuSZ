#ifndef CUSZ_WORKFLOW_CUH
#define CUSZ_WORKFLOW_CUH

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
#include "analysis_utils.hh"
#include "argparse.hh"
#include "type_trait.hh"

using namespace std;

namespace cusz {
namespace interface {

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void Compress(
    argpack*,
    struct AdHocDataPack<typename DataTrait<If_FP, DataByte>::Data>*,
    size_t*,
    double*,
    int&,
    size_t&,
    size_t&,
    size_t&);

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void Decompress(argpack*, size_t*, double*, int&, size_t&, size_t&, size_t&);

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
