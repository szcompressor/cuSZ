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
void CSR11<T>::gather_CUDA11(T* in_dense, unsigned int& _dump_poolsize, cudaStream_t stream)
{
    cusparseHandle_t     handle = nullptr;
    cusparseSpMatDescr_t spmat;  // sparse
    cusparseDnMatDescr_t dnmat;  // dense
    void*                d_buffer      = nullptr;
    size_t               d_buffer_size = 0;

    CHECK_CUSPARSE(cusparseCreate(&handle));

    if (stream) CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    auto num_rows = m;
    auto num_cols = m;
    auto ld       = m;

    // Create dense matrix A
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&dnmat, num_rows, num_cols, ld, in_dense, cuszCUSPARSE<T>::type, CUSPARSE_ORDER_ROW));

    // Create sparse matrix B in CSR format
    auto d_rowptr = rowptr.template get<DEFAULT_LOC>();
    CHECK_CUSPARSE(cusparseCreateCsr(
        &spmat, num_rows, num_cols, 0, d_rowptr, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, cuszCUSPARSE<T>::type));

    // allocate an external buffer if needed
    {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(
            cusparseDenseToSparse_bufferSize(handle, dnmat, spmat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &d_buffer_size));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();

        CHECK_CUDA(cudaMalloc(&d_buffer, d_buffer_size));
    }

    // execute Sparse to Dense conversion
    {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(
            cusparseDenseToSparse_analysis(handle, dnmat, spmat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buffer));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
    }

    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, __nnz;
    /**  this is all HOST, skip timing **/
    CHECK_CUSPARSE(cusparseSpMatGetSize(spmat, &num_rows_tmp, &num_cols_tmp, &__nnz));

    auto d_colidx = colidx.template get<DEFAULT_LOC>();
    auto d_val    = values.template get<DEFAULT_LOC>();

    // allocate CSR column indices and values (skipped in customiztion)

    // reset offsets, column indices, and values pointers
    CHECK_CUSPARSE(cusparseCsrSetPointers(spmat, d_rowptr, d_colidx, d_val));

    // execute Sparse to Dense conversion
    {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(
            cusparseDenseToSparse_convert(handle, dnmat, spmat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buffer));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
    }

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroyDnMat(dnmat));
    CHECK_CUSPARSE(cusparseDestroySpMat(spmat));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    /********************************************************************************/
    reconfigure_with_precise_nnz(__nnz);
    dump_nbyte     = query_csr_bytelen();
    _dump_poolsize = dump_nbyte;
}

template <typename T>
void CSR11<T>::gather_CUDA11_new(T* in_dense, cudaStream_t stream)
{
    auto num_rows = rte.m;
    auto num_cols = rte.m;
    auto ld       = rte.m;

    auto gather11_init_mat = [&]() {
        // create dense matrix wrapper
        CHECK_CUSPARSE(cusparseCreateDnMat(
            &rte.dnmat, num_rows, num_cols, ld, in_dense, cuszCUSPARSE<T>::type, CUSPARSE_ORDER_ROW));

        // create CSR wrapper
        CHECK_CUSPARSE(cusparseCreateCsr(
            &rte.spmat, num_rows, num_cols, 0, d_rowptr, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, cuszCUSPARSE<T>::type));
    };

    auto gather11_init_buffer = [&]() {
        {  // allocate an external buffer if needed
            cuda_timer_t t;
            t.timer_start(stream);

            CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
                rte.handle, rte.dnmat, rte.spmat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &rte.d_buffer_size));

            t.timer_end(stream);
            milliseconds += t.get_time_elapsed();

            CHECK_CUDA(cudaMalloc(&rte.d_buffer, rte.d_buffer_size));
        }
    };

    auto gather11_analysis = [&]() {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(cusparseDenseToSparse_analysis(
            rte.handle, rte.dnmat, rte.spmat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, rte.d_buffer));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
    };

    int64_t num_rows_tmp, num_cols_tmp;

    auto gather11_get_nnz = [&]() {
        // get number of non-zero elements
        CHECK_CUSPARSE(cusparseSpMatGetSize(rte.spmat, &num_rows_tmp, &num_cols_tmp, &rte.nnz));
    };

    auto gather11_get_rowptr = [&]() {
        // reset offsets, column indices, and values pointers
        CHECK_CUSPARSE(cusparseCsrSetPointers(rte.spmat, d_rowptr, d_colidx, d_val));
    };

    auto gather11_dn2csr = [&]() {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(cusparseDenseToSparse_convert(
            rte.handle, rte.dnmat, rte.spmat, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, rte.d_buffer));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
    };

    /********************************************************************************/
    CHECK_CUSPARSE(cusparseCreate(&rte.handle));
    if (stream) CHECK_CUSPARSE(cusparseSetStream(rte.handle, stream));  // TODO move out

    gather11_init_mat();
    gather11_init_buffer();
    gather11_analysis();
    gather11_get_nnz();
    gather11_get_rowptr();
    gather11_dn2csr();

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroyDnMat(rte.dnmat));
    CHECK_CUSPARSE(cusparseDestroySpMat(rte.spmat));
    CHECK_CUSPARSE(cusparseDestroy(rte.handle));
}

#elif CUDART_VERSION >= 10000

template <typename T>
void CSR11<T>::gather_CUDA10(T* in_dense, unsigned int& _dump_poolsize, cudaStream_t ext_stream)
{
    cusparseHandle_t   handle         = nullptr;
    cudaStream_t       stream         = nullptr;
    cusparseMatDescr_t mat_desc       = nullptr;
    size_t             lwork_in_bytes = 0;
    char*              d_work         = nullptr;
    float              threshold      = 0;
    auto               n              = m;
    auto               ld             = m;

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
            handle, m, n, in_dense, ld, &threshold, mat_desc, values.template get<DEFAULT_LOC>(),
            rowptr.template get<DEFAULT_LOC>(), colidx.template get<DEFAULT_LOC>(), &lwork_in_bytes));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
    }

    if (nullptr != d_work) cudaFree(d_work);
    CHECK_CUDA(cudaMalloc((void**)&d_work, lwork_in_bytes));  // TODO where to release d_work?

    auto nnz = 0;

    /* step 4: compute rowptr and nnz */
    {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(cusparseSpruneDense2csrNnz(  //
            handle, m, n, in_dense, ld, &threshold, mat_desc, rowptr.template get<DEFAULT_LOC>(), &nnz, d_work));

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
            handle, m, n, in_dense, ld, &threshold, mat_desc, values.template get<DEFAULT_LOC>(),
            rowptr.template get<DEFAULT_LOC>(), colidx.template get<DEFAULT_LOC>(), d_work));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    if (handle) cusparseDestroy(handle);
    if (mat_desc) cusparseDestroyMatDescr(mat_desc);

    if ((not has_ext_stream) and stream) cudaStreamDestroy(stream);

    /********************************************************************************/
    dump_nbyte     = query_csr_bytelen();
    _dump_poolsize = dump_nbyte;
    /********************************************************************************/
}

template <typename T>
void CSR11<T>::gather_CUDA10_new(T* in_dense, cudaStream_t stream)
{
    int num_rows, num_cols, ld;
    num_rows = num_cols = ld = rte.m;

    float threshold{0};
    auto  has_ext_stream{false};

    /******************************************************************************/

    auto gather10_init_and_probe = [&]() {
        {  // init

            CHECK_CUSPARSE(cusparseCreateMatDescr(&rte.mat_desc));                            // 4. create rte.mat_desc
            CHECK_CUSPARSE(cusparseSetMatIndexBase(rte.mat_desc, CUSPARSE_INDEX_BASE_ZERO));  // zero based
            CHECK_CUSPARSE(cusparseSetMatType(rte.mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL));   // type
        }

        {  // probe
            cuda_timer_t t;
            t.timer_start(stream);

            CHECK_CUSPARSE(cusparseSpruneDense2csr_bufferSizeExt(
                rte.handle, num_rows, num_cols, in_dense, ld, &threshold, rte.mat_desc, d_val, d_rowptr, d_colidx,
                &rte.lwork_in_bytes));

            t.timer_end(stream);
            milliseconds += t.get_time_elapsed();
        }

        if (nullptr != rte.d_work) cudaFree(rte.d_work);
        CHECK_CUDA(cudaMalloc((void**)&rte.d_work, rte.lwork_in_bytes));  // TODO where to release d_work?
    };

    auto gather10_compute_rowptr_and_nnz = [&]() {  // step 4
        cuda_timer_t t;
        t.timer_start(stream);

        int nnz;  // for compatibility; cuSPARSE of CUDA 11 changed data type

        CHECK_CUSPARSE(cusparseSpruneDense2csrNnz(
            rte.handle, num_rows, num_cols, in_dense, ld, &threshold, rte.mat_desc, d_rowptr, &nnz, rte.d_work));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
        CHECK_CUDA(cudaStreamSynchronize(stream));

        rte.nnz = nnz;
    };

    auto gather10_compute_colidx_and_val = [&]() {  // step 5
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(cusparseSpruneDense2csr(  //
            rte.handle, num_rows, num_cols, in_dense, ld, &threshold, rte.mat_desc, d_val, d_rowptr, d_colidx,
            rte.d_work));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
        CHECK_CUDA(cudaStreamSynchronize(stream));
    };

    /********************************************************************************/
    if (stream)
        has_ext_stream = true;
    else
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));  // 1. create stream
    CHECK_CUSPARSE(cusparseCreate(&rte.handle));                                // 2. create handle
    CHECK_CUSPARSE(cusparseSetStream(rte.handle, stream));                      // 3. bind stream

    gather10_init_and_probe();
    gather10_compute_rowptr_and_nnz();
    if (nnz == 0) { return; }
    gather10_compute_colidx_and_val();

    // TODO no need to destroy?
    if (rte.handle) cusparseDestroy(rte.handle);
    if (rte.mat_desc) cusparseDestroyMatDescr(rte.mat_desc);
    if ((not has_ext_stream) and stream) cudaStreamDestroy(stream);
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
void CSR11<T>::scatter_CUDA11(T* out_dense, cudaStream_t stream)
{
    auto d_rowptr = rowptr.template get<DEFAULT_LOC>();
    auto d_colidx = colidx.template get<DEFAULT_LOC>();
    auto d_val    = values.template get<DEFAULT_LOC>();

    /********************************************************************************/

    cusparseHandle_t     handle{nullptr};
    cusparseSpMatDescr_t spmat;
    cusparseDnMatDescr_t dnmat;
    void*                d_buffer{nullptr};
    size_t               d_buffer_size{0};

    auto num_rows = m;
    auto num_cols = m;
    auto ld       = m;

    CHECK_CUSPARSE(cusparseCreate(&handle));

    if (stream) CHECK_CUSPARSE(cusparseSetStream(handle, stream));

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(
        &spmat, num_rows, num_cols, nnz, d_rowptr, d_colidx, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, cuszCUSPARSE<T>::type));
    // Create dense matrix B
    CHECK_CUSPARSE(
        cusparseCreateDnMat(&dnmat, num_rows, num_cols, ld, out_dense, cuszCUSPARSE<T>::type, CUSPARSE_ORDER_ROW));

    {
        cuda_timer_t t;
        t.timer_start(stream);

        // allocate an external buffer if needed
        CHECK_CUSPARSE(
            cusparseSparseToDense_bufferSize(handle, spmat, dnmat, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &d_buffer_size));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
    }
    CHECK_CUDA(cudaMalloc(&d_buffer, d_buffer_size));

    // execute Sparse to Dense conversion
    {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(cusparseSparseToDense(handle, spmat, dnmat, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, d_buffer));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
    }

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(spmat));
    CHECK_CUSPARSE(cusparseDestroyDnMat(dnmat));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}

template <typename T>
void CSR11<T>::scatter_CUDA11_new(BYTE* in_csr, T* out_dense, cudaStream_t stream, bool header_on_device)
{
    header_t header;
    if (header_on_device) CHECK_CUDA(cudaMemcpyAsync(&header, in_csr, sizeof(header), cudaMemcpyDeviceToHost, stream));

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_csr + header.entry[HEADER::SYM])
    auto d_rowptr = ACCESSOR(ROWPTR, int);
    auto d_colidx = ACCESSOR(COLIDX, int);
    auto d_val    = ACCESSOR(VAL, T);
#undef ACCESSOR

    auto num_rows = header.m;
    auto num_cols = header.m;
    auto ld       = header.m;
    auto nnz      = header.nnz;

    auto scatter11_init_mat = [&]() {
        CHECK_CUSPARSE(cusparseCreateCsr(
            &rte.spmat, num_rows, num_cols, nnz, d_rowptr, d_colidx, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, cuszCUSPARSE<T>::type));

        CHECK_CUSPARSE(cusparseCreateDnMat(
            &rte.dnmat, num_rows, num_cols, ld, out_dense, cuszCUSPARSE<T>::type, CUSPARSE_ORDER_ROW));
    };

    auto scatter11_init_buffer = [&]() {
        cuda_timer_t t;
        t.timer_start(stream);

        // allocate an external buffer if needed
        CHECK_CUSPARSE(cusparseSparseToDense_bufferSize(
            rte.handle, rte.spmat, rte.dnmat, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &rte.d_buffer_size));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();

        CHECK_CUDA(cudaMalloc(&rte.d_buffer, rte.d_buffer_size));
    };

    auto scatter11_csr2dn = [&]() {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(
            cusparseSparseToDense(rte.handle, rte.spmat, rte.dnmat, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, rte.d_buffer));

        t.timer_end(stream);
        milliseconds += t.get_time_elapsed();
    };

    /******************************************************************************/
    CHECK_CUSPARSE(cusparseCreate(&rte.handle));
    if (stream) CHECK_CUSPARSE(cusparseSetStream(rte.handle, stream));

    scatter11_init_mat();
    scatter11_init_buffer();
    scatter11_csr2dn();

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(rte.spmat));
    CHECK_CUSPARSE(cusparseDestroyDnMat(rte.dnmat));
    CHECK_CUSPARSE(cusparseDestroy(rte.handle));
}

#elif CUDART_VERSION >= 10000

template <typename T>
void CSR11<T>::scatter_CUDA10(T* out_dense, cudaStream_t ext_stream)
{
    cusparseHandle_t   handle   = nullptr;  // TODO move cusparse handle outside
    cudaStream_t       stream   = nullptr;
    cusparseMatDescr_t mat_desc = nullptr;
    auto               n        = m;
    auto               ld       = m;

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
            colidx.template get<DEFAULT_LOC>(), out_dense, ld));

        t.timer_end();
        milliseconds += t.get_time_elapsed();
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // TODO move cusparse handle outside
    if (handle) cusparseDestroy(handle);
    if (mat_desc) cusparseDestroyMatDescr(mat_desc);
    if ((not has_external_stream) and stream) cudaStreamDestroy(stream);
}

template <typename T>
void CSR11<T>::scatter_CUDA10_new(BYTE* in_csr, T* out_dense, cudaStream_t stream, bool header_on_device)
{
    header_t header;
    if (header_on_device) CHECK_CUDA(cudaMemcpyAsync(&header, in_csr, sizeof(header), cudaMemcpyDeviceToHost, stream));

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_csr + header.entry[HEADER::SYM])
    auto d_rowptr = ACCESSOR(ROWPTR, int);
    auto d_colidx = ACCESSOR(COLIDX, int);
    auto d_val    = ACCESSOR(VAL, T);
#undef ACCESSOR

    auto num_rows = header.m;
    auto num_cols = header.m;
    auto ld       = header.m;

    auto has_external_stream = false;

    /******************************************************************************/

    auto scatter10_init = [&]() {
        CHECK_CUSPARSE(cusparseCreateMatDescr(&rte.mat_desc));                            // 4. create descr
        CHECK_CUSPARSE(cusparseSetMatIndexBase(rte.mat_desc, CUSPARSE_INDEX_BASE_ZERO));  // zero based
        CHECK_CUSPARSE(cusparseSetMatType(rte.mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL));   // type
    };

    auto scatter10_sparse2dense = [&]() {
        cuda_timer_t t;
        t.timer_start(stream);

        CHECK_CUSPARSE(
            cusparseScsr2dense(rte.handle, num_rows, num_cols, rte.mat_desc, d_val, d_rowptr, d_colidx, out_dense, ld));

        t.timer_end();
        milliseconds += t.get_time_elapsed();
        CHECK_CUDA(cudaStreamSynchronize(stream));
    };

    /******************************************************************************/
    if (stream)
        has_external_stream = true;
    else
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUSPARSE(cusparseCreate(&rte.handle));
    CHECK_CUSPARSE(cusparseSetStream(rte.handle, stream));

    scatter10_init();
    scatter10_sparse2dense();

    if (rte.handle) cusparseDestroy(rte.handle);
    if (rte.mat_desc) cusparseDestroyMatDescr(rte.mat_desc);
    if ((not has_external_stream) and stream) cudaStreamDestroy(stream);
    /******************************************************************************/
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
