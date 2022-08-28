/**
 * @file launch_sparse_method.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-06-13
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef CUSZ_LAUNCH_SPARSE_METHOD_CUH
#define CUSZ_LAUNCH_SPARSE_METHOD_CUH

#include <hip/hip_runtime.h>
#include <hipsparse.h>

#include "../common.hh"
#include "../utils.hh"

// #if CUDART_VERSION >= 11020

template <typename T, typename M>
void launch_cusparse_gather_cuda11200_onward(
    hipsparseHandle_t     handle,
    T*                    in_dense,
    uint32_t const        num_rows,
    uint32_t const        num_cols,
    hipsparseDnMatDescr_t dnmat,
    hipsparseSpMatDescr_t spmat,
    void*                 d_buffer,
    size_t&               d_buffer_size,
    M*                    d_rowptr,
    M*                    d_colidx,
    T*                    d_val,
    int64_t&              nnz,
    float&                milliseconds,
    hipStream_t           stream)
{
    auto ld = num_rows;

    auto gather11_init_mat = [&]() {
        // create dense matrix wrapper
        CHECK_CUSPARSE(
            hipsparseCreateDnMat(&dnmat, num_rows, num_cols, ld, in_dense, cuszCUSPARSE<T>::type, HIPSPARSE_ORDER_ROW));

        // create CSR wrapper
        CHECK_CUSPARSE(hipsparseCreateCsr(
            &spmat, num_rows, num_cols, 0, d_rowptr, nullptr, nullptr, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
            HIPSPARSE_INDEX_BASE_ZERO, cuszCUSPARSE<T>::type));
    };

    auto gather11_init_buffer = [&]() {
        {  // allocate an external buffer if needed
            // cuda_timer_t t;
            // t.timer_start(stream);

            CHECK_CUSPARSE(hipsparseDenseToSparse_bufferSize(
                handle, dnmat, spmat, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT, &d_buffer_size));

            // t.timer_end(stream);
            milliseconds += 0;  // t.get_time_elapsed();

            CHECK_CUDA(hipMalloc(&d_buffer, d_buffer_size));
        }
    };

    auto gather11_analysis = [&]() {
        // cuda_timer_t t;
        // t.timer_start(stream);

        CHECK_CUSPARSE(
            hipsparseDenseToSparse_analysis(handle, dnmat, spmat, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buffer));

        // t.timer_end(stream);
        milliseconds += 0;  // t.get_time_elapsed();
    };

    int64_t num_rows_tmp, num_cols_tmp;

    auto gather11_get_nnz = [&]() {
        // get number of non-zero elements
        CHECK_CUSPARSE(hipsparseSpMatGetSize(spmat, &num_rows_tmp, &num_cols_tmp, &nnz));
    };

    auto gather11_get_rowptr = [&]() {
        // reset offsets, column indices, and values pointers
        CHECK_CUSPARSE(hipsparseCsrSetPointers(spmat, d_rowptr, d_colidx, d_val));
    };

    auto gather11_dn2csr = [&]() {
        // cuda_timer_t t;
        // t.timer_start(stream);

        CHECK_CUSPARSE(
            hipsparseDenseToSparse_convert(handle, dnmat, spmat, HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buffer));

        // t.timer_end(stream);
        milliseconds += 0;  // t.get_time_elapsed();
    };

    /********************************************************************************/
    milliseconds = 0;

    CHECK_CUSPARSE(hipsparseCreate(&handle));
    if (stream) CHECK_CUSPARSE(hipsparseSetStream(handle, stream));  // TODO move out

    gather11_init_mat();
    gather11_init_buffer();
    gather11_analysis();
    gather11_get_nnz();
    gather11_get_rowptr();
    gather11_dn2csr();

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(hipsparseDestroyDnMat(dnmat));
    CHECK_CUSPARSE(hipsparseDestroySpMat(spmat));
    CHECK_CUSPARSE(hipsparseDestroy(handle));
}

// void SpcodecCSR<T, M>::impl::scatter_CUDA_11020(BYTE* in_csr, T* out_dense, hipStream_t stream, bool
// header_on_device)

template <typename T, typename M>
void launch_cusparse_scatter_cuda11200_onward(
    hipsparseHandle_t     handle,
    int*                  d_rowptr,
    int*                  d_colidx,
    T*                    d_val,
    int const             num_rows,
    int const             num_cols,
    int const             nnz,
    hipsparseDnMatDescr_t dnmat,
    hipsparseSpMatDescr_t spmat,
    void*                 d_buffer,
    size_t&               d_buffer_size,
    T*                    out_dense,
    float&                milliseconds,
    hipStream_t           stream)
{
    auto ld = num_rows;

    auto scatter11_init_mat = [&]() {
        CHECK_CUSPARSE(hipsparseCreateCsr(
            &spmat, num_rows, num_cols, nnz, d_rowptr, d_colidx, d_val, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
            HIPSPARSE_INDEX_BASE_ZERO, cuszCUSPARSE<T>::type));

        CHECK_CUSPARSE(hipsparseCreateDnMat(
            &dnmat, num_rows, num_cols, ld, out_dense, cuszCUSPARSE<T>::type, HIPSPARSE_ORDER_ROW));
    };

    auto scatter11_init_buffer = [&]() {
        // cuda_timer_t t;
        // t.timer_start(stream);

        // allocate an external buffer if needed
        CHECK_CUSPARSE(hipsparseSparseToDense_bufferSize(
            handle, spmat, dnmat, HIPSPARSE_SPARSETODENSE_ALG_DEFAULT, &d_buffer_size));

        // t.timer_end(stream);
        milliseconds += 0;  // t.get_time_elapsed();

        CHECK_CUDA(hipMalloc(&d_buffer, d_buffer_size));
    };

    auto scatter11_csr2dn = [&]() {
        // cuda_timer_t t;
        // t.timer_start(stream);

        CHECK_CUSPARSE(hipsparseSparseToDense(handle, spmat, dnmat, HIPSPARSE_SPARSETODENSE_ALG_DEFAULT, d_buffer));

        // t.timer_end(stream);
        milliseconds += 0;  // t.get_time_elapsed();
    };

    /******************************************************************************/
    milliseconds = 0;

    CHECK_CUSPARSE(hipsparseCreate(&handle));
    if (stream) CHECK_CUSPARSE(hipsparseSetStream(handle, stream));

    scatter11_init_mat();
    scatter11_init_buffer();
    scatter11_csr2dn();

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(hipsparseDestroySpMat(spmat));
    CHECK_CUSPARSE(hipsparseDestroyDnMat(dnmat));
    CHECK_CUSPARSE(hipsparseDestroy(handle));
}

// #elif CUDART_VERSION >= 10000

template <typename T, typename M>
void launch_cusparse_gather_before_cuda11200(
    hipsparseHandle_t   handle,
    T*                  in_dense,
    uint32_t const      num_rows,
    uint32_t const      num_cols,
    hipsparseMatDescr_t mat_desc,
    void*               d_work,
    size_t&             lwork_in_bytes,
    M*                  d_rowptr,
    M*                  d_colidx,
    T*                  d_val,
    int&                nnz,  // int is for compatibility; cuSPARSE of CUDA 11 changed data type
    float&              milliseconds,
    hipStream_t         stream)
{
    auto ld = num_rows;

    float threshold{0};
    auto  has_ext_stream{false};

    /******************************************************************************/

    auto gather10_init_and_probe = [&]() {
        {  // init

            CHECK_CUSPARSE(hipsparseCreateMatDescr(&mat_desc));                             // 4. create rte.mat_desc
            CHECK_CUSPARSE(hipsparseSetMatIndexBase(mat_desc, HIPSPARSE_INDEX_BASE_ZERO));  // zero based
            CHECK_CUSPARSE(hipsparseSetMatType(mat_desc, HIPSPARSE_MATRIX_TYPE_GENERAL));   // type
        }

        {  // probe
            // cuda_timer_t t;
            // t.timer_start(stream);

            CHECK_CUSPARSE(hipsparseSpruneDense2csr_bufferSizeExt(
                handle, num_rows, num_cols, in_dense, ld, &threshold, mat_desc, d_val, d_rowptr, d_colidx,
                &lwork_in_bytes));

            // t.timer_end(stream);
            milliseconds += 0;  // t.get_time_elapsed();
        }

        if (nullptr != d_work) hipFree(d_work);
        CHECK_CUDA(hipMalloc((void**)&d_work, lwork_in_bytes));  // TODO where to release d_work?
    };

    auto gather10_compute_rowptr_and_nnz = [&]() {  // step 4
        // cuda_timer_t t;
        // t.timer_start(stream);

        CHECK_CUSPARSE(hipsparseSpruneDense2csrNnz(
            handle, num_rows, num_cols, in_dense, ld, &threshold, mat_desc, d_rowptr, &nnz, d_work));

        // t.timer_end(stream);
        milliseconds += 0;  // t.get_time_elapsed();
        CHECK_CUDA(hipStreamSynchronize(stream));

    };

    auto gather10_compute_colidx_and_val = [&]() {  // step 5
        // cuda_timer_t t;
        // t.timer_start(stream);

        CHECK_CUSPARSE(hipsparseSpruneDense2csr(  //
            handle, num_rows, num_cols, in_dense, ld, &threshold, mat_desc, d_val, d_rowptr, d_colidx, d_work));

        // t.timer_end(stream);
        milliseconds += 0;  // t.get_time_elapsed();
        CHECK_CUDA(hipStreamSynchronize(stream));
    };

    /********************************************************************************/
    milliseconds = 0;

    if (stream)
        has_ext_stream = true;
    else
        CHECK_CUDA(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));  // 1. create stream
    CHECK_CUSPARSE(hipsparseCreate(&handle));                                 // 2. create handle
    CHECK_CUSPARSE(hipsparseSetStream(handle, stream));                       // 3. bind stream

    gather10_init_and_probe();
    gather10_compute_rowptr_and_nnz();
    if (nnz == 0) { return; }
    gather10_compute_colidx_and_val();

    // TODO no need to destroy?
    if (handle) hipsparseDestroy(handle);
    if (mat_desc) hipsparseDestroyMatDescr(mat_desc);
    if ((not has_ext_stream) and stream) hipStreamDestroy(stream);
    /********************************************************************************/
}

// #endif

template <typename T, typename M>
void launch_cusparse_scatter_before_cuda11200(
    hipsparseHandle_t   handle,
    int*                d_rowptr,
    int*                d_colidx,
    T*                  d_val,
    int const           num_rows,
    int const           num_cols,
    int const           nnz,
    hipsparseMatDescr_t mat_desc,
    void*               d_buffer,
    size_t&             d_buffer_size,
    T*                  out_dense,
    float&              milliseconds,
    hipStream_t         stream)
{
    auto ld = num_rows;

    auto has_external_stream = false;

    /******************************************************************************/

    auto scatter10_init = [&]() {
        CHECK_CUSPARSE(hipsparseCreateMatDescr(&mat_desc));                             // 4. create descr
        CHECK_CUSPARSE(hipsparseSetMatIndexBase(mat_desc, HIPSPARSE_INDEX_BASE_ZERO));  // zero based
        CHECK_CUSPARSE(hipsparseSetMatType(mat_desc, HIPSPARSE_MATRIX_TYPE_GENERAL));   // type
    };

    auto scatter10_sparse2dense = [&]() {
        // cuda_timer_t t;
        // t.timer_start(stream);

        CHECK_CUSPARSE(
            hipsparseScsr2dense(handle, num_rows, num_cols, mat_desc, d_val, d_rowptr, d_colidx, out_dense, ld));

        // t.timer_end();
        milliseconds += 0;  // t.get_time_elapsed();
        CHECK_CUDA(hipStreamSynchronize(stream));
    };

    /******************************************************************************/
    if (stream)
        has_external_stream = true;
    else
        CHECK_CUDA(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    CHECK_CUSPARSE(hipsparseCreate(&handle));
    CHECK_CUSPARSE(hipsparseSetStream(handle, stream));

    scatter10_init();
    scatter10_sparse2dense();

    if (handle) hipsparseDestroy(handle);
    if (mat_desc) hipsparseDestroyMatDescr(mat_desc);
    if ((not has_external_stream) and stream) hipStreamDestroy(stream);
    /******************************************************************************/
}

#endif
