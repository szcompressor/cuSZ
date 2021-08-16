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
 * compression use
 ********************************************************************************/

template <typename Data>
OutlierHandler<Data>::OutlierHandler(unsigned int _len)
{
    this->m = static_cast<size_t>(ceil(sqrt(_len)));

    // TODO merge to configure?
    auto initial_nnz = _len / 10;
    // set up pool
    pool.offset.rowptr = 0;
    pool.offset.colidx = sizeof(int) * (m + 1);
    pool.offset.values = sizeof(int) * (m + 1) + sizeof(int) * initial_nnz;

    ;
}

template <typename Data>
OutlierHandler<Data>& OutlierHandler<Data>::configure(uint8_t* _pool)
{
    if (not _pool) throw std::runtime_error("Memory pool is no allocated.");
    pool.ptr          = _pool;
    pool.entry.rowptr = reinterpret_cast<int*>(pool.ptr + pool.offset.rowptr);
    pool.entry.colidx = reinterpret_cast<int*>(pool.ptr + pool.offset.colidx);
    pool.entry.values = reinterpret_cast<Data*>(pool.ptr + pool.offset.values);

    return *this;
}

template <typename Data>
OutlierHandler<Data>& OutlierHandler<Data>::configure_with_nnz(int nnz)
{
    this->nnz      = nnz;
    bytelen.rowptr = sizeof(int) * (m + 1);
    bytelen.colidx = sizeof(int) * nnz;
    bytelen.values = sizeof(Data) * nnz;
    bytelen.total  = bytelen.rowptr + bytelen.colidx + bytelen.values;

    return *this;
}

/********************************************************************************
 * "S" (for "single-precision") is used; can't generalize
 ********************************************************************************/

template <typename Data>
OutlierHandler<Data>&
OutlierHandler<Data>::gather_CUDA10(float* in_outlier, unsigned int& _dump_poolsize, float& milliseconds)
{
    handle_t handle       = nullptr;
    stream_t stream       = nullptr;
    descr_t  mat_desc     = nullptr;
    size_t   lworkInBytes = 0;
    char*    d_work       = nullptr;
    float    threshold    = 0;
    auto     n            = m;
    auto     lda          = m;

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
            handle, m, n, in_outlier, lda, &threshold, mat_desc, pool.entry.values, pool.entry.rowptr,
            pool.entry.colidx, &lworkInBytes));

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
            handle, m, n, in_outlier, lda, &threshold, mat_desc, pool.entry.rowptr, &nnz, d_work));

        milliseconds += timer_step4->timer_end_get_elapsed_time();
        CHECK_CUDA(cudaDeviceSynchronize());
        delete timer_step4;
    }

    this->configure_with_nnz(nnz);

    if (nnz == 0) {
        std::cout << "nnz == 0, exiting gather.\n";
        return *this;
    }

    /* step 5: compute col_ind and values */
    {
        auto timer_step5 = new cuda_timer_t;
        timer_step5->timer_start();

        CHECK_CUSPARSE(cusparseSpruneDense2csr(  //
            handle, m, n, in_outlier, lda, &threshold, mat_desc, pool.entry.values, pool.entry.rowptr,
            pool.entry.colidx, d_work));

        milliseconds += timer_step5->timer_end_get_elapsed_time();
        CHECK_CUDA(cudaDeviceSynchronize());
        delete timer_step5;
    }

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (mat_desc) cusparseDestroyMatDescr(mat_desc);

    /********************************************************************************/
    dump_nbyte     = query_csr_bytelen();
    _dump_poolsize = dump_nbyte;
    /********************************************************************************/
    return *this;
}

template <typename Data>
OutlierHandler<Data>& OutlierHandler<Data>::archive(uint8_t* archive, int& export_nnz)
{
    export_nnz = this->nnz;

    // clang-format off
    cudaMemcpy(archive + 0,                               pool.entry.rowptr, bytelen.rowptr, cudaMemcpyDeviceToHost);
    cudaMemcpy(archive + bytelen.rowptr,                  pool.entry.colidx, bytelen.colidx, cudaMemcpyDeviceToHost);
    cudaMemcpy(archive + bytelen.rowptr + bytelen.colidx, pool.entry.values, bytelen.values, cudaMemcpyDeviceToHost);
    // clang-format on

    return *this;
}

/********************************************************************************
 * decompression use
 ********************************************************************************/

template <typename Data>
OutlierHandler<Data>::OutlierHandler(unsigned int _len, unsigned int _nnz)
{  //
    this->m   = static_cast<size_t>(ceil(sqrt(_len)));
    this->nnz = _nnz;

    bytelen.rowptr = sizeof(int) * (this->m + 1);
    bytelen.colidx = sizeof(int) * this->nnz;
    bytelen.values = sizeof(Data) * this->nnz;
    bytelen.total  = bytelen.rowptr + bytelen.colidx + bytelen.values;
}

template <typename Data>
OutlierHandler<Data>& OutlierHandler<Data>::extract(uint8_t* _pool)
{
    pool.offset.rowptr = 0;
    pool.offset.colidx = bytelen.rowptr;
    pool.offset.values = bytelen.rowptr + bytelen.colidx;

    pool.ptr          = _pool;
    pool.entry.rowptr = reinterpret_cast<int*>(pool.ptr + pool.offset.rowptr);
    pool.entry.colidx = reinterpret_cast<int*>(pool.ptr + pool.offset.colidx);
    pool.entry.values = reinterpret_cast<Data*>(pool.ptr + pool.offset.values);

    return *this;
};

template <typename Data>
OutlierHandler<Data>& OutlierHandler<Data>::scatter_CUDA10(float* in_outlier, float& milliseconds)
{
    //     throw std::runtime_error("[decompress_scatter] not implemented");
    handle_t handle   = nullptr;
    stream_t stream   = nullptr;
    descr_t  mat_desc = nullptr;
    auto     n        = m;
    auto     lda      = m;

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
            handle, m, n, mat_desc, pool.entry.values, pool.entry.rowptr, pool.entry.colidx, in_outlier, lda));

        milliseconds += timer_scatter->timer_end_get_elapsed_time();
        CHECK_CUDA(cudaDeviceSynchronize());
        delete timer_scatter;
    }

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (mat_desc) cusparseDestroyMatDescr(mat_desc);

    return *this;
}

template class OutlierHandler<float>;
