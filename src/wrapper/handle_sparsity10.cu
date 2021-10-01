/**
 * @file handle_sparsity10.cu
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

#include "../common.hh"
#include "../utils.hh"

#include "handle_sparsity10.cuh"

using handle_t = cusparseHandle_t;
using stream_t = cudaStream_t;
using descr_t  = cusparseMatDescr_t;

/********************************************************************************
 * compression use
 ********************************************************************************/

namespace cusz {

template <typename T>
OutlierHandler10<T>::OutlierHandler10(unsigned int _len, unsigned int* init_workspace_nbyte)
{
    if (init_workspace_nbyte == nullptr)
        throw std::runtime_error("[OutlierHandler10::constructor] init_workspace_nbyte must not be null.");

    m = Reinterpret1DTo2D::get_square_size(_len);

    // TODO merge to configure?
    auto initial_nnz = _len / SparseMethodSetup::factor;
    // set up pool
    offset.rowptr = 0;
    offset.colidx = sizeof(int) * (m + 1);
    offset.values = sizeof(int) * (m + 1) + sizeof(int) * initial_nnz;

    *init_workspace_nbyte = sizeof(int) * (m + 1) +      // rowptr
                            sizeof(int) * initial_nnz +  // colidx
                            sizeof(T) * initial_nnz;     // values
}

template <typename T>
void OutlierHandler10<T>::configure_workspace(uint8_t* _pool)
{
    if (not _pool) throw std::runtime_error("Memory is no allocated.");
    pool_ptr     = _pool;
    entry.rowptr = reinterpret_cast<int*>(pool_ptr + offset.rowptr);
    entry.colidx = reinterpret_cast<int*>(pool_ptr + offset.colidx);
    entry.values = reinterpret_cast<T*>(pool_ptr + offset.values);
}

template <typename T>
void OutlierHandler10<T>::reconfigure_with_precise_nnz(int nnz)
{
    this->nnz    = nnz;
    nbyte.rowptr = sizeof(int) * (m + 1);
    nbyte.colidx = sizeof(int) * nnz;
    nbyte.values = sizeof(T) * nnz;
    nbyte.total  = nbyte.rowptr + nbyte.colidx + nbyte.values;
}

/********************************************************************************
 * "S" (for "single-precision") is used; can't generalize
 ********************************************************************************/

template <typename T>
void OutlierHandler10<T>::gather_CUDA10(T* in_outlier, unsigned int& _dump_poolsize)
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
            handle, m, n, in_outlier, lda, &threshold, mat_desc, entry.values, entry.rowptr, entry.colidx,
            &lworkInBytes));

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
            handle, m, n, in_outlier, lda, &threshold, mat_desc, entry.rowptr, &nnz, d_work));

        milliseconds += timer_step4->timer_end_get_elapsed_time();
        CHECK_CUDA(cudaDeviceSynchronize());
        delete timer_step4;
    }

    reconfigure_with_precise_nnz(nnz);

    if (nnz == 0) {
        std::cout << "nnz == 0, exiting gather.\n";
        // return *this;
        return;
    }

    /* step 5: compute col_ind and values */
    {
        auto timer_step5 = new cuda_timer_t;
        timer_step5->timer_start();

        CHECK_CUSPARSE(cusparseSpruneDense2csr(  //
            handle, m, n, in_outlier, lda, &threshold, mat_desc, entry.values, entry.rowptr, entry.colidx, d_work));

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
}

template <typename T>
void OutlierHandler10<T>::archive(uint8_t* dst, int& export_nnz, cudaMemcpyKind direction)
{
    export_nnz = this->nnz;

    // clang-format off
    cudaMemcpy(dst + 0,                           entry.rowptr, nbyte.rowptr, direction);
    cudaMemcpy(dst + nbyte.rowptr,                entry.colidx, nbyte.colidx, direction);
    cudaMemcpy(dst + nbyte.rowptr + nbyte.colidx, entry.values, nbyte.values, direction);
    // clang-format on

    // TODO not working, alignment issue?
    // cudaMemcpy(dst, entry.rowptr, bytelen.total, direction);
}

/********************************************************************************
 * decompression use
 ********************************************************************************/

template <typename T>
OutlierHandler10<T>::OutlierHandler10(unsigned int _len, unsigned int _nnz)
{  //
    this->m   = Reinterpret1DTo2D::get_square_size(_len);
    this->nnz = _nnz;

    nbyte.rowptr = sizeof(int) * (this->m + 1);
    nbyte.colidx = sizeof(int) * this->nnz;
    nbyte.values = sizeof(T) * this->nnz;
    nbyte.total  = nbyte.rowptr + nbyte.colidx + nbyte.values;
}

template <typename T>
void OutlierHandler10<T>::extract(uint8_t* _pool)
{
    offset.rowptr = 0;
    offset.colidx = nbyte.rowptr;
    offset.values = nbyte.rowptr + nbyte.colidx;

    pool_ptr     = _pool;
    entry.rowptr = reinterpret_cast<int*>(pool_ptr + offset.rowptr);
    entry.colidx = reinterpret_cast<int*>(pool_ptr + offset.colidx);
    entry.values = reinterpret_cast<T*>(pool_ptr + offset.values);
};

template <typename T>
void OutlierHandler10<T>::scatter_CUDA10(T* in_outlier)
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

        CHECK_CUSPARSE(
            cusparseScsr2dense(handle, m, n, mat_desc, entry.values, entry.rowptr, entry.colidx, in_outlier, lda));

        milliseconds += timer_scatter->timer_end_get_elapsed_time();
        CHECK_CUDA(cudaDeviceSynchronize());
        delete timer_scatter;
    }

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (mat_desc) cusparseDestroyMatDescr(mat_desc);
}

//
}  // namespace cusz

template class cusz::OutlierHandler10<float>;
