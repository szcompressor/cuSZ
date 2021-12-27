/**
 * @file csr11.cu
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

#include "csr11.cuh"

/********************************************************************************
 * compression use
 ********************************************************************************/

namespace cusz {

template <typename T>
void CSR11<T>::reconfigure_with_precise_nnz(int nnz)
{
    this->nnz    = nnz;
    nbyte.rowptr = sizeof(int) * (m + 1);
    nbyte.colidx = sizeof(int) * nnz;
    nbyte.values = sizeof(T) * nnz;
    nbyte.total  = nbyte.rowptr + nbyte.colidx + nbyte.values;
}

#if CUDART_VERSION >= 11020

template <typename T>
void CSR11<T>::gather_CUDA11(T* in_data, unsigned int& _dump_poolsize, cudaStream_t stream)
{
    cusparseHandle_t     handle = nullptr;
    cusparseSpMatDescr_t matB;  // sparse
    cusparseDnMatDescr_t matA;  // dense
    void*                dBuffer    = nullptr;
    size_t               bufferSize = 0;

    auto d_dense = in_data;

    CHECK_CUSPARSE(cusparseCreate(&handle));

    if (stream) CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    auto num_rows = m;
    auto num_cols = m;
    auto ld       = m;

    // Create dense matrix A
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense, cuszCUSPARSE<T>::type, CUSPARSE_ORDER_ROW));

    // Create sparse matrix B in CSR format
    auto d_csr_offsets = rowptr.template get<DEFAULT_LOC>();
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matB, num_rows, num_cols, 0, d_csr_offsets, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, cuszCUSPARSE<T>::type));

    // allocate an external buffer if needed
    {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(
            cusparseDenseToSparse_bufferSize(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();

        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    // execute Sparse to Dense conversion
    {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
    }

    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, __nnz;
    /**  this is all HOST, skip timing **/
    CHECK_CUSPARSE(cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp, &__nnz));

    auto d_csr_columns = colidx.template get<DEFAULT_LOC>();
    auto d_csr_values  = values.template get<DEFAULT_LOC>();

    // allocate CSR column indices and values (skipped in customiztion)

    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE(cusparseCsrSetPointers(matB, d_csr_offsets, d_csr_columns, d_csr_values));

    // execute Sparse to Dense conversion
    {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
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

#elif CUDART_VERSION >= 10000

template <typename T>
void CSR11<T>::gather_CUDA10(T* in_outlier, unsigned int& _dump_poolsize, cudaStream_t ext_stream)
{
    cusparseHandle_t   handle       = nullptr;
    cudaStream_t       stream       = nullptr;
    cusparseMatDescr_t mat_desc     = nullptr;
    size_t             lworkInBytes = 0;
    char*              d_work       = nullptr;
    float              threshold    = 0;
    auto               n            = m;
    auto               lda          = m;

    auto has_ext_stream = false;

    if (ext_stream) {
        has_ext_stream = true;
        stream         = ext_stream;
    }
    else {
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));  // 1. create stream
    }

    // clang-format off
    CHECK_CUSPARSE(cusparseCreate          ( &handle                                  )); // 2. create handle
    CHECK_CUSPARSE(cusparseSetStream       (  handle,    stream                       )); // 3. bind stream
    CHECK_CUSPARSE(cusparseCreateMatDescr  ( &mat_desc                                )); // 4. create mat_desc
    CHECK_CUSPARSE(cusparseSetMatIndexBase (  mat_desc,  CUSPARSE_INDEX_BASE_ZERO     )); // zero based
    CHECK_CUSPARSE(cusparseSetMatType      (  mat_desc,  CUSPARSE_MATRIX_TYPE_GENERAL )); // type
    // clang-format on

    {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(cusparseSpruneDense2csr_bufferSizeExt(  //
            handle, m, n, in_outlier, lda, &threshold, mat_desc, values.template get<DEFAULT_LOC>(),
            rowptr.template get<DEFAULT_LOC>(), colidx.template get<DEFAULT_LOC>(), &lworkInBytes));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
    }

    if (nullptr != d_work) cudaFree(d_work);
    CHECK_CUDA(cudaMalloc((void**)&d_work, lworkInBytes));  // TODO where to release d_work?

    auto nnz = 0;

    /* step 4: compute rowptr and nnz */
    {
        cuda_timer_t t.;
        t.timer_start(stream);

        CHECK_CUSPARSE(cusparseSpruneDense2csrNnz(  //
            handle, m, n, in_outlier, lda, &threshold, mat_desc, rowptr.template get<DEFAULT_LOC>(), &nnz, d_work));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    reconfigure_with_precise_nnz(nnz);

    if (nnz == 0) {
        std::cout << "nnz == 0, exiting gather.\n";
        // return *this;
        return;
    }

    /* step 5: compute col_ind and values */
    {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(cusparseSpruneDense2csr(  //
            handle, m, n, in_outlier, lda, &threshold, mat_desc, values.template get<DEFAULT_LOC>(),
            rowptr.template get<DEFAULT_LOC>(), colidx.template get<DEFAULT_LOC>(), d_work));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
        CHECK_CUDA(cudaStreamSynchronize(stream));
        delete timer_step5;
    }

    if (handle) cusparseDestroy(handle);
    if (mat_desc) cusparseDestroyMatDescr(mat_desc);

    if ((not has_ext_stream) and stream) cudaStreamDestroy(stream);

    /********************************************************************************/
    dump_nbyte     = query_csr_bytelen();
    _dump_poolsize = dump_nbyte;
    /********************************************************************************/
}

#else
#error CUDART_VERSION must be no less than 10.0!
#endif

template <typename T>
template <cusz::LOC FROM, cusz::LOC TO>
CSR11<T>& CSR11<T>::consolidate(uint8_t* dst)
{
    constexpr auto direction = CopyDirection<FROM, TO>::direction;
    // clang-format off
    cudaMemcpy(dst + 0,                           rowptr.template get<DEFAULT_LOC>(), nbyte.rowptr, direction);
    cudaMemcpy(dst + nbyte.rowptr,                colidx.template get<DEFAULT_LOC>(), nbyte.colidx, direction);
    cudaMemcpy(dst + nbyte.rowptr + nbyte.colidx, values.template get<DEFAULT_LOC>(), nbyte.values, direction);
    // clang-format on
    return *this;
}

template <typename T>
CSR11<T>& CSR11<T>::decompress_set_nnz(unsigned int _nnz)
{  //
    this->nnz = _nnz;

    nbyte.rowptr = sizeof(int) * (this->m + 1);
    nbyte.colidx = sizeof(int) * this->nnz;
    nbyte.values = sizeof(T) * this->nnz;
    nbyte.total  = nbyte.rowptr + nbyte.colidx + nbyte.values;

    return *this;
}

template <typename T>
void CSR11<T>::extract(uint8_t* _pool)
{
    offset.rowptr = 0;
    offset.colidx = nbyte.rowptr;
    offset.values = nbyte.rowptr + nbyte.colidx;

    pool_ptr                           = _pool;
    rowptr.template get<DEFAULT_LOC>() = reinterpret_cast<int*>(pool_ptr + offset.rowptr);
    colidx.template get<DEFAULT_LOC>() = reinterpret_cast<int*>(pool_ptr + offset.colidx);
    values.template get<DEFAULT_LOC>() = reinterpret_cast<T*>(pool_ptr + offset.values);
};

#if CUDART_VERSION >= 11020

template <typename T>
void CSR11<T>::scatter_CUDA11(T* out_dn, cudaStream_t stream)
{
    auto d_csr_offsets = rowptr.template get<DEFAULT_LOC>();
    auto d_csr_columns = colidx.template get<DEFAULT_LOC>();
    auto d_csr_values  = values.template get<DEFAULT_LOC>();

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

    if (stream) CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA, num_rows, num_cols, nnz, d_csr_offsets, d_csr_columns, d_csr_values, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, cuszCUSPARSE<T>::type));
    // Create dense matrix B
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&matB, num_rows, num_cols, ld, d_dense, cuszCUSPARSE<T>::type, CUSPARSE_ORDER_ROW));

    {
        cuda_timer_t t;
        t.timer_start(stream);

        // allocate an external buffer if needed
        CHECK_CUSPARSE(
            cusparseSparseToDense_bufferSize(handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
    }
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute Sparse to Dense conversion
    {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(cusparseSparseToDense(handle, matA, matB, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, dBuffer));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
    }

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}

#elif CUDART_VERSION >= 10000

template <typename T>
void CSR11<T>::scatter_CUDA10(T* out_dn, cudaStream_t ext_stream)
{
    cusparseHandle_t   handle   = nullptr;  // TODO move cusparse handle outside
    cudaStream_t       stream   = nullptr;
    cusparseMatDescr_t mat_desc = nullptr;
    auto               n        = m;
    auto               lda      = m;

    auto has_external_stream = false;

    if (ext_stream) {
        has_external_stream = true;
        stream              = ext_stream;
    }
    else {
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));  // 1. create stream
    }

    // clang-format off
    CHECK_CUSPARSE(cusparseCreate          ( &handle                                 )); // 2. create handle
    CHECK_CUSPARSE(cusparseSetStream       (  handle,   stream                       )); // 3. bind stream
    CHECK_CUSPARSE(cusparseCreateMatDescr  ( &mat_desc                               )); // 4. create descr
    CHECK_CUSPARSE(cusparseSetMatIndexBase (  mat_desc, CUSPARSE_INDEX_BASE_ZERO     )); // zero based
    CHECK_CUSPARSE(cusparseSetMatType      (  mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL )); // type
    // clang-format on

    {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(cusparseScsr2dense(
            handle, m, n, mat_desc, values.template get<DEFAULT_LOC>(), rowptr.template get<DEFAULT_LOC>(),
            colidx.template get<DEFAULT_LOC>(), out_dn, lda));

        t.timer_end();
        milliseconds += t.get_time_elapsed();
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // TODO move cusparse handle outside
    if (handle) cusparseDestroy(handle);

    if (mat_desc) cusparseDestroyMatDescr(mat_desc);

    if ((not has_external_stream) and stream) cudaStreamDestroy(stream);
}

#else
#error CUDART_VERSION must be no less than 10.0!
#endif

//
}  // namespace cusz

#define CSR11_TYPE cusz::CSR11<float>

template class CSR11_TYPE;

template CSR11_TYPE& CSR11_TYPE::consolidate<cusz::LOC::HOST, cusz::LOC::HOST>(uint8_t*);
template CSR11_TYPE& CSR11_TYPE::consolidate<cusz::LOC::HOST, cusz::LOC::DEVICE>(uint8_t*);
template CSR11_TYPE& CSR11_TYPE::consolidate<cusz::LOC::DEVICE, cusz::LOC::HOST>(uint8_t*);
template CSR11_TYPE& CSR11_TYPE::consolidate<cusz::LOC::DEVICE, cusz::LOC::DEVICE>(uint8_t*);
