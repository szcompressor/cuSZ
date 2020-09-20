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
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "argparse.hh"

namespace cusz {
namespace workflow {

template <typename T, typename Q, typename H>
void Compress(
    std::string& fi,  //
    size_t*      dims_L16,
    double*      ebs_L4,
    int&         nnz_outlier,
    size_t&      n_bits,
    size_t&      n_uInt,
    size_t&      huffman_metadata_size,
    argpack*     ap);

template <typename T, typename Q, typename H>
void Decompress(
    std::string& fi,
    size_t*      dims_L16,
    double*      ebs_L4,
    int&         nnz_outlier,
    size_t&      total_bits,
    size_t&      total_uInt,
    size_t&      huffman_metadata_size,
    argpack*     ap);

}  // namespace workflow

namespace impl {

inline size_t GetEdgeOfReinterpretedSquare(size_t l)
{
    auto sl = static_cast<size_t>(sqrt(l));
    return ((sl - 1) / 2 + 1) * 2;
};

template <typename T, typename Q>
void PdQ(T*, Q*, size_t*, double*);

template <typename T, typename Q>
void ReversedPdQ(T*, Q*, T*, size_t*, double);

template <typename T, typename Q>
void VerifyHuffman(string const&, size_t, Q*, int, size_t*, double*);

}  // namespace impl

}  // namespace cusz

#endif
