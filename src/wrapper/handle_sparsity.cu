/**
 * @file handle_sparsity.cu
 * @author Jiannan Tian
 * @brief A high-level sparsity handling wrapper.
 * @version 0.3
 * @date 2021-06-17
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstddef>
#include <iostream>
#include "../utils/cuda_err.cuh"
#include "handle_sparsity.h"

using handle_t = cusparseHandle_t;
using stream_t = cudaStream_t;
using descr_t  = cusparseMatDescr_t;

/********************************************************************************
 * "S" (for "single-precision") is used; can't generalize
 ********************************************************************************/
void compress_gather_CUDA10(struct OutlierDescriptor<float>* outlier_desc, float* ondev_outlier)
{
    handle_t handle       = nullptr;
    stream_t stream       = nullptr;
    descr_t  mat_desc     = nullptr;
    size_t   lworkInBytes = 0;
    char*    d_work       = nullptr;
    float    threshold    = 0;
    auto     m            = outlier_desc->m;
    auto     n            = outlier_desc->m;
    auto     lda          = outlier_desc->m;

    // clang-format off
    CHECK_CUDA(cudaStreamCreateWithFlags   ( &stream,    cudaStreamNonBlocking        )); // 1. create stream
    CHECK_CUSPARSE(cusparseCreate          ( &handle                                  )); // 2. create handle
    CHECK_CUSPARSE(cusparseSetStream       (  handle,    stream                       )); // 3. bind stream
    CHECK_CUSPARSE(cusparseCreateMatDescr  ( &mat_desc                                )); // 4. create mat_desc
    CHECK_CUSPARSE(cusparseSetMatIndexBase (  mat_desc,  CUSPARSE_INDEX_BASE_ZERO     )); // zero based
    CHECK_CUSPARSE(cusparseSetMatType      (  mat_desc,  CUSPARSE_MATRIX_TYPE_GENERAL )); // type
    // clang-format on

    CHECK_CUDA(cudaMalloc((void**)&outlier_desc->ondev_row_ptr, sizeof(int) * (m + 1)));

    CHECK_CUSPARSE(cusparseSpruneDense2csr_bufferSizeExt(  //
        handle, m, n, ondev_outlier, lda, &threshold, mat_desc, outlier_desc->ondev_csr_val,
        outlier_desc->ondev_row_ptr, outlier_desc->ondev_col_idx, &lworkInBytes));

    if (nullptr != d_work) cudaFree(d_work);
    CHECK_CUDA(cudaMalloc((void**)&d_work, lworkInBytes));  // TODO where to release d_work?

    auto nnz = 0;

    /* step 4: compute row_ptr and nnz */
    CHECK_CUSPARSE(cusparseSpruneDense2csrNnz(  //
        handle, m, n, ondev_outlier, lda, &threshold, mat_desc, outlier_desc->ondev_row_ptr, &nnz, d_work));
    CHECK_CUDA(cudaDeviceSynchronize());

    outlier_desc->configure(nnz);

    if (nnz == 0) {
        std::cout << "nnz == 0, exiting gather.\n";
        return;
    }

    /* step 5: compute col_ind and csr_val */
    CHECK_CUDA(cudaMalloc((void**)&outlier_desc->bytelen_colidx, sizeof(int) * nnz));
    CHECK_CUDA(cudaMalloc((void**)&outlier_desc->bytelen_values, sizeof(float) * nnz));

    CHECK_CUSPARSE(cusparseSpruneDense2csr(  //
        handle, m, n, ondev_outlier, lda, &threshold, mat_desc, outlier_desc->ondev_csr_val,
        outlier_desc->ondev_row_ptr, outlier_desc->ondev_col_idx, d_work));
    CHECK_CUDA(cudaDeviceSynchronize());

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (mat_desc) cusparseDestroyMatDescr(mat_desc);
}

void decompress_scatter_CUDA10(struct OutlierDescriptor<float>* outlier_desc, float* ondev_outlier)
{
    //     throw std::runtime_error("[decompress_scatter] not implemented");
    handle_t handle   = nullptr;
    stream_t stream   = nullptr;
    descr_t  mat_desc = nullptr;
    auto     m        = outlier_desc->m;
    auto     n        = outlier_desc->m;
    auto     lda      = outlier_desc->m;

    // clang-format off
    CHECK_CUDA(cudaStreamCreateWithFlags   ( &stream,   cudaStreamNonBlocking        )); // 1. create stream
    CHECK_CUSPARSE(cusparseCreate          ( &handle                                 )); // 2. create handle
    CHECK_CUSPARSE(cusparseSetStream       (  handle,   stream                       )); // 3. bind stream
    CHECK_CUSPARSE(cusparseCreateMatDescr  ( &mat_desc                               )); // 4. create descr
    CHECK_CUSPARSE(cusparseSetMatIndexBase (  mat_desc, CUSPARSE_INDEX_BASE_ZERO     )); // zero based
    CHECK_CUSPARSE(cusparseSetMatType      (  mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL )); // type
    // clang-format on

    CHECK_CUSPARSE(cusparseScsr2dense(
        handle, m, n, mat_desc, outlier_desc->ondev_csr_val, outlier_desc->ondev_row_ptr, outlier_desc->ondev_col_idx,
        ondev_outlier, lda));
    CHECK_CUDA(cudaDeviceSynchronize());

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (mat_desc) cusparseDestroyMatDescr(mat_desc);
}
