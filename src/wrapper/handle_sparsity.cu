/**
 * @file handle_sparsity.cu
 * @author Jiannan Tian
 * @brief A high-level sparsity handling wrapper. Gather/scatter method to handle cuSZ prediction outlier.
 * @version 0.3
 * @date 2021-07-08
 * (created) 2020-09-10 (rev1) 2021-06-17 (rev2) 2021-07-08
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstddef>
#include <iostream>
#include "../utils/cuda_err.cuh"
#include "../utils/timer.hh"
#include "handle_sparsity.h"

using handle_t = cusparseHandle_t;
using stream_t = cudaStream_t;
using descr_t  = cusparseMatDescr_t;

/********************************************************************************
 * "S" (for "single-precision") is used; can't generalize
 ********************************************************************************/
void compress_gather_CUDA10(struct OutlierDescriptor<float>* csr, float* in_outlier, float& milliseconds)
{
    handle_t handle       = nullptr;
    stream_t stream       = nullptr;
    descr_t  mat_desc     = nullptr;
    size_t   lworkInBytes = 0;
    char*    d_work       = nullptr;
    float    threshold    = 0;
    auto     m            = csr->m;
    auto     n            = csr->m;
    auto     lda          = csr->m;

    // clang-format off
    CHECK_CUDA(cudaStreamCreateWithFlags   ( &stream,    cudaStreamNonBlocking        )); // 1. create stream
    CHECK_CUSPARSE(cusparseCreate          ( &handle                                  )); // 2. create handle
    CHECK_CUSPARSE(cusparseSetStream       (  handle,    stream                       )); // 3. bind stream
    CHECK_CUSPARSE(cusparseCreateMatDescr  ( &mat_desc                                )); // 4. create mat_desc
    CHECK_CUSPARSE(cusparseSetMatIndexBase (  mat_desc,  CUSPARSE_INDEX_BASE_ZERO     )); // zero based
    CHECK_CUSPARSE(cusparseSetMatType      (  mat_desc,  CUSPARSE_MATRIX_TYPE_GENERAL )); // type
    // clang-format on

    {
        auto timer_step3 = new cuda_timer_t;
        timer_step3->timer_start();

        CHECK_CUSPARSE(cusparseSpruneDense2csr_bufferSizeExt(  //
            handle, m, n, in_outlier, lda, &threshold, mat_desc, csr->pool.entry.values, csr->pool.entry.rowptr,
            csr->pool.entry.colidx, &lworkInBytes));

        milliseconds += timer_step3->timer_end_get_elapsed_time();
        delete timer_step3;
    }

    if (nullptr != d_work) cudaFree(d_work);
    CHECK_CUDA(cudaMalloc((void**)&d_work, lworkInBytes));  // TODO where to release d_work?

    auto nnz = 0;

    /* step 4: compute rowptr and nnz */
    {
        auto timer_step4 = new cuda_timer_t;
        timer_step4->timer_start();

        CHECK_CUSPARSE(cusparseSpruneDense2csrNnz(  //
            handle, m, n, in_outlier, lda, &threshold, mat_desc, csr->pool.entry.rowptr, &nnz, d_work));

        milliseconds += timer_step4->timer_end_get_elapsed_time();
        CHECK_CUDA(cudaDeviceSynchronize());
        delete timer_step4;
    }

    csr->compress_configure_with_nnz(nnz);

    if (nnz == 0) {
        std::cout << "nnz == 0, exiting gather.\n";
        return;
    }

    /* step 5: compute col_ind and values */
    {
        auto timer_step5 = new cuda_timer_t;
        timer_step5->timer_start();

        CHECK_CUSPARSE(cusparseSpruneDense2csr(  //
            handle, m, n, in_outlier, lda, &threshold, mat_desc, csr->pool.entry.values, csr->pool.entry.rowptr,
            csr->pool.entry.colidx, d_work));

        milliseconds += timer_step5->timer_end_get_elapsed_time();
        CHECK_CUDA(cudaDeviceSynchronize());
        delete timer_step5;
    }

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (mat_desc) cusparseDestroyMatDescr(mat_desc);
}

void decompress_scatter_CUDA10(struct OutlierDescriptor<float>* csr, float* in_outlier, float& milliseconds)
{
    //     throw std::runtime_error("[decompress_scatter] not implemented");
    handle_t handle   = nullptr;
    stream_t stream   = nullptr;
    descr_t  mat_desc = nullptr;
    auto     m        = csr->m;
    auto     n        = csr->m;
    auto     lda      = csr->m;

    // clang-format off
    CHECK_CUDA(cudaStreamCreateWithFlags   ( &stream,   cudaStreamNonBlocking        )); // 1. create stream
    CHECK_CUSPARSE(cusparseCreate          ( &handle                                 )); // 2. create handle
    CHECK_CUSPARSE(cusparseSetStream       (  handle,   stream                       )); // 3. bind stream
    CHECK_CUSPARSE(cusparseCreateMatDescr  ( &mat_desc                               )); // 4. create descr
    CHECK_CUSPARSE(cusparseSetMatIndexBase (  mat_desc, CUSPARSE_INDEX_BASE_ZERO     )); // zero based
    CHECK_CUSPARSE(cusparseSetMatType      (  mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL )); // type
    // clang-format on

    {
        auto timer_scatter = new cuda_timer_t;
        timer_scatter->timer_start();

        CHECK_CUSPARSE(cusparseScsr2dense(
            handle, m, n, mat_desc, csr->pool.entry.values, csr->pool.entry.rowptr, csr->pool.entry.colidx, in_outlier,
            lda));

        milliseconds += timer_scatter->timer_end_get_elapsed_time();
        CHECK_CUDA(cudaDeviceSynchronize());
        delete timer_scatter;
    }

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (mat_desc) cusparseDestroyMatDescr(mat_desc);
}
