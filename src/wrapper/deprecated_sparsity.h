/**
 * @file gather_scatter.h
 * @author Jiannan Tian
 * @brief Gather/scatter method to handle cuSZ prediction outlier (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-09-10
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */
#ifndef OLD_SPARSITY_H
#define OLD_SPARSITY_H

#include <cuda_runtime.h>
#include <cusparse.h>
#include <string>

namespace cusz {
namespace impl {

template <typename DType>
void GatherAsCSR(DType* d_A, size_t lenA, size_t ldA, size_t m, size_t n, int* p_nnz, std::string* fo);

template <typename DType>
void ScatterFromCSR(DType* d_A, size_t lenA, size_t ldA, size_t m, size_t n, const int* p_nnz, std::string* fi);

void PruneGatherAsCSR(
    float*       d_A,  //
    size_t       len,
    const int    lda,
    const int    m,
    const int    n,
    int&         nnzC,
    std::string* fo);

}  // namespace impl
}  // namespace cusz

#endif