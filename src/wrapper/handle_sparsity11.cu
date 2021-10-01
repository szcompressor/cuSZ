/**
 * @file handle_sparsity11.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-28
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

#include "handle_sparsity11.cuh"

using handle_t = cusparseHandle_t;
using stream_t = cudaStream_t;
// using descr_t  = cusparseMatDescr_t;

/********************************************************************************
 * compression use
 ********************************************************************************/

namespace cusz {

template <typename T>
OutlierHandler11<T>::OutlierHandler11(unsigned int _len, unsigned int* init_workspace_nbyte)
{
    if (init_workspace_nbyte == nullptr)
        throw std::runtime_error("[OutlierHandler11::constructor] init_workspace_nbyte must not be null.");

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
void OutlierHandler11<T>::configure_workspace(uint8_t* _pool)
{
    if (not _pool) throw std::runtime_error("Memory is no allocated.");
    pool_ptr     = _pool;
    entry.rowptr = reinterpret_cast<int*>(pool_ptr + offset.rowptr);
    entry.colidx = reinterpret_cast<int*>(pool_ptr + offset.colidx);
    entry.values = reinterpret_cast<T*>(pool_ptr + offset.values);
}

template <typename T>
void OutlierHandler11<T>::reconfigure_with_precise_nnz(int nnz)
{
    this->nnz    = nnz;
    nbyte.rowptr = sizeof(int) * (m + 1);
    nbyte.colidx = sizeof(int) * nnz;
    nbyte.values = sizeof(T) * nnz;
    nbyte.total  = nbyte.rowptr + nbyte.colidx + nbyte.values;
}

template <typename T>
void OutlierHandler11<T>::gather_CUDA11(T* in_data, unsigned int& _dump_poolsize)
{
    cusparseHandle_t     handle = nullptr;
    cusparseSpMatDescr_t matB;  // sparse
    cusparseDnMatDescr_t matA;  // dense
    void*                dBuffer    = nullptr;
    size_t               bufferSize = 0;

    auto d_dense = in_data;

    CHECK_CUSPARSE(cusparseCreate(&handle));

    auto num_rows = m;
    auto num_cols = m;
    auto ld       = m;

    // Create dense matrix A
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense, cuszCUSPARSE<T>::type, CUSPARSE_ORDER_ROW));

    // Create sparse matrix B in CSR format
    auto d_csr_offsets = entry.rowptr;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matB, num_rows, num_cols, 0, d_csr_offsets, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, cuszCUSPARSE<T>::type));

    // allocate an external buffer if needed
    {
        auto t = new cuda_timer_t;
        t->timer_start();

        CHECK_CUSPARSE(
            cusparseDenseToSparse_bufferSize(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));

        milliseconds += t->timer_end_get_elapsed_time();
        delete t;

        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    // execute Sparse to Dense conversion
    {
        auto t = new cuda_timer_t;
        t->timer_start();

        CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

        milliseconds += t->timer_end_get_elapsed_time();
        delete t;
    }

    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, __nnz;
    /**  this is all HOST, skip timing **/
    CHECK_CUSPARSE(cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp, &__nnz));

    auto d_csr_columns = entry.colidx;
    auto d_csr_values  = entry.values;

    // allocate CSR column indices and values (skipped in customiztion)

    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE(cusparseCsrSetPointers(matB, d_csr_offsets, d_csr_columns, d_csr_values));

    // execute Sparse to Dense conversion
    {
        auto t = new cuda_timer_t;
        t->timer_start();

        CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

        milliseconds += t->timer_end_get_elapsed_time();
        delete t;
    }

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroyDnMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    /********************************************************************************/
    reconfigure_with_precise_nnz(__nnz);
    dump_nbyte     = query_csr_bytelen();
    _dump_poolsize = dump_nbyte;
}

template <typename T>
void OutlierHandler11<T>::archive(uint8_t* dst, int& export_nnz, cudaMemcpyKind direction)
{
    export_nnz = this->nnz;

    // clang-format off
    cudaMemcpy(dst + 0,                           entry.rowptr, nbyte.rowptr, direction);
    cudaMemcpy(dst + nbyte.rowptr,                entry.colidx, nbyte.colidx, direction);
    cudaMemcpy(dst + nbyte.rowptr + nbyte.colidx, entry.values, nbyte.values, direction);
    // clang-format on
}

/********************************************************************************
 * decompression use
 ********************************************************************************/

template <typename T>
OutlierHandler11<T>::OutlierHandler11(unsigned int _len, unsigned int _nnz)
{  //
    this->m   = Reinterpret1DTo2D::get_square_size(_len);
    this->nnz = _nnz;

    nbyte.rowptr = sizeof(int) * (this->m + 1);
    nbyte.colidx = sizeof(int) * this->nnz;
    nbyte.values = sizeof(T) * this->nnz;
    nbyte.total  = nbyte.rowptr + nbyte.colidx + nbyte.values;
}

template <typename T>
void OutlierHandler11<T>::extract(uint8_t* _pool)
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
void OutlierHandler11<T>::scatter_CUDA11(T* out_dn)
{
    auto d_csr_offsets = entry.rowptr;
    auto d_csr_columns = entry.colidx;
    auto d_csr_values  = entry.values;

    /********************************************************************************/

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;

    auto num_rows = m;
    auto num_cols = m;
    auto ld       = m;

    auto d_dense = out_dn;

    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, num_rows, num_cols, nnz, d_csr_offsets, d_csr_columns, d_csr_values, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuszCUSPARSE<T>::type));
    // Create dense matrix B
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&matB, num_rows, num_cols, ld, d_dense, cuszCUSPARSE<T>::type, CUSPARSE_ORDER_ROW));

    {
        auto t = new cuda_timer_t;
        t->timer_start();

        // allocate an external buffer if needed
        CHECK_CUSPARSE(
            cusparseSparseToDense_bufferSize(handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize));

        milliseconds += t->timer_end_get_elapsed_time();
        delete t;
    }
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute Sparse to Dense conversion
    {
        auto t = new cuda_timer_t;
        t->timer_start();

        CHECK_CUSPARSE(cusparseSparseToDense(handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, dBuffer));

        milliseconds += t->timer_end_get_elapsed_time();
        delete t;
    }

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}

//
}  // namespace cusz

template class cusz::OutlierHandler11<float>;
