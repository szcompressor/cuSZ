// 20-09-10

#include <bits/stdint-uintn.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cassert>
#include <iostream>
using std::cout;
using std::endl;

#include "cuda_error_handling.cuh"
#include "format.hh"
#include "gather_scatter.cuh"
#include "io.hh"

template <typename DType>
void cuSZ::impl::GatherAsCSR(DType* d_A, size_t lenA, size_t ldA, int* nnz, std::string* fo)
{
    // dealing with outlier
    uint8_t* outbin;
    size_t   lrp, lci, lv, ltotal;

    {
        cusparseHandle_t   handle      = nullptr;
        cudaStream_t       stream      = nullptr;
        cusparseMatDescr_t descr       = nullptr;
        const int          m           = ldA;
        const int          n           = ldA;
        int*               d_nnzPerRow = nullptr;
        int*               d_csrRowPtr = nullptr;
        int*               d_csrColInd = nullptr;
        DType*             d_csrVal    = nullptr;

        // clang-format off
        CHECK_CUDA(cudaStreamCreateWithFlags(   &stream, cudaStreamNonBlocking        )); // 1. create stream
        CHECK_CUSPARSE(cusparseCreate(          &handle                               )); // 2. create handle
        CHECK_CUSPARSE(cusparseSetStream(        handle, stream                       )); // 3. bind stream
        CHECK_CUSPARSE(cusparseCreateMatDescr(  &descr                                )); // 4. create descr
        CHECK_CUSPARSE(cusparseSetMatIndexBase(  descr,  CUSPARSE_INDEX_BASE_ZERO     )); // zero based
        CHECK_CUSPARSE(cusparseSetMatType(       descr,  CUSPARSE_MATRIX_TYPE_GENERAL )); // typ
        // clang-format on

        // compute nnz
        CHECK_CUDA(cudaMalloc((void**)&d_nnzPerRow, sizeof(int) * m));

        CHECK_CUSPARSE(cusparseSnnz(
            handle, CUSPARSE_DIRECTION_ROW,  // parsed by row
            m, n, descr, d_A, ldA,           // descrption of d_A
            d_nnzPerRow, nnz)                // output
        );

        cout << "nnz: " << *nnz << endl;

        lrp    = sizeof(int) * (m + 1);
        lci    = sizeof(int) * *nnz;
        lv     = sizeof(DType) * *nnz;
        ltotal = lrp + lci + lv;
        // csrRowPtr   = new int[m + 1];
        // csrColInd   = new int[*nnz];
        // csrVal      = new DType[*nnz];
        outbin = new uint8_t[ltotal];
        CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtr, lrp));
        CHECK_CUDA(cudaMalloc((void**)&d_csrColInd, lci));
        CHECK_CUDA(cudaMalloc((void**)&d_csrVal, lv));

        CHECK_CUSPARSE(cusparseSdense2csr(
            handle,                              //
            m, n, descr, d_A, ldA,               // descritpion of d_A
            d_nnzPerRow,                         // prefileld by nnz() func
            d_csrVal, d_csrRowPtr, d_csrColInd)  // output
        );
        CHECK_CUDA(cudaDeviceSynchronize());

        // clang-format off
        // CHECK_CUDA(cudaMemcpy(csrRowPtr, d_csrRowPtr, lrp, cudaMemcpyDeviceToHost));
        // CHECK_CUDA(cudaMemcpy(csrColInd, d_csrColInd, lci, cudaMemcpyDeviceToHost));
        // CHECK_CUDA(cudaMemcpy(csrVal,    d_csrVal,    lv,  cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(outbin,             d_csrRowPtr, lrp, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(outbin + lrp,       d_csrColInd, lci, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(outbin + lrp + lci, d_csrVal,    lv,  cudaMemcpyDeviceToHost));
        // clang-format on

        auto csrval = reinterpret_cast<DType*>(outbin + lrp + lci);
        int  count  = 0;
        for (auto i = 0; i < *nnz; i++) {
            if (csrval[i] != 0) {
                // cout << i << "\t" << csrVal[i] << endl;
                count++;
            }
        }
        cout << "non zero count again: " << count << endl;

        // clean up
        if (d_csrRowPtr) cudaFree(d_csrRowPtr);
        if (d_csrColInd) cudaFree(d_csrColInd);
        if (d_csrVal) cudaFree(d_csrVal);
        if (d_nnzPerRow) cudaFree(d_nnzPerRow);
        if (handle) cusparseDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
        if (descr) cusparseDestroyMatDescr(descr);
    }

    cout << log_dbg << "outlier_bin byte length:\t" << ltotal << endl;

    io::WriteBinaryFile(outbin, ltotal, fo);
    cout << log_info << "Saved outlier in CSR format." << endl;
    delete[] outbin;
};

template void cuSZ::impl::GatherAsCSR<float>(float* d_A, size_t lenA, size_t ldA, int* nnz, std::string* fo);

template <typename DType>
void cuSZ::impl::ScatterFromCSR(DType* d_A, size_t lenA, size_t ldA, int* nnz, std::string* fi)
{
    // clang-format off
    auto lrp   = sizeof(int)   * (ldA + 1);
    auto lci   = sizeof(int)   * *nnz;
    auto lv      = sizeof(DType) * *nnz;
    auto l_total       = lrp + lci + lv;
    auto outlier_bin   = io::ReadBinaryFile<uint8_t>(*fi, l_total);
    auto csrRowPtr     = reinterpret_cast<int*  >(outlier_bin);
    auto csrColInd     = reinterpret_cast<int*  >(outlier_bin + lrp);
    auto csrVal        = reinterpret_cast<DType*>(outlier_bin + lrp + lci);  // TODO template
    // clang-format on

    int count = 0;
    for (auto i = 0; i < *nnz; i++) {
        if (csrVal[i] != 0) {
            // cout << i << "\t" << csrVal[i] << endl;
            count++;
        }
    }
    cout << "non zero count again (extract): " << count << endl;
    cout << log_dbg << "outlier_bin byte length:\t" << l_total << endl;

    {
        cusparseHandle_t   handle      = nullptr;
        cudaStream_t       stream      = nullptr;
        cusparseMatDescr_t descr       = nullptr;
        const int          m           = ldA;
        const int          n           = m;
        int*               d_csrRowPtr = nullptr;
        int*               d_csrColInd = nullptr;
        DType*             d_csrVal    = nullptr;

        // clang-format off
        CHECK_CUDA(cudaStreamCreateWithFlags(   &stream, cudaStreamNonBlocking        )); // 1. create stream
        CHECK_CUSPARSE(cusparseCreate(          &handle                               )); // 2. create handle
        CHECK_CUSPARSE(cusparseSetStream(        handle, stream                       )); // 3. bind stream
        CHECK_CUSPARSE(cusparseCreateMatDescr(  &descr                                )); // 4. create descr
        CHECK_CUSPARSE(cusparseSetMatIndexBase(  descr,  CUSPARSE_INDEX_BASE_ZERO     )); // zero based
        CHECK_CUSPARSE(cusparseSetMatType(       descr,  CUSPARSE_MATRIX_TYPE_GENERAL )); // type

        CHECK_CUDA(cudaMalloc( (void**)&d_csrRowPtr,   lrp ));
        CHECK_CUDA(cudaMalloc( (void**)&d_csrColInd,   lci ));
        CHECK_CUDA(cudaMalloc( (void**)&d_csrVal,      lv    ));
        CHECK_CUDA(cudaMemcpy( d_csrRowPtr, csrRowPtr, lrp, cudaMemcpyHostToDevice ));
        CHECK_CUDA(cudaMemcpy( d_csrColInd, csrColInd, lci, cudaMemcpyHostToDevice ));
        CHECK_CUDA(cudaMemcpy( d_csrVal,    csrVal,    lv,    cudaMemcpyHostToDevice ));
        // clang-format on

        // fill
        CHECK_CUSPARSE(cusparseScsr2dense(handle, m, n, descr, d_csrVal, d_csrRowPtr, d_csrColInd, d_A, ldA));
        CHECK_CUDA(cudaDeviceSynchronize());

        if (d_csrRowPtr) cudaFree(d_csrRowPtr);
        if (d_csrColInd) cudaFree(d_csrColInd);
        if (d_csrVal) cudaFree(d_csrVal);
        if (handle) cusparseDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
        if (descr) cusparseDestroyMatDescr(descr);
    }

    cout << log_info << "Extracted outlier from CSR format." << endl;

    delete[] outlier_bin;
}

template void cuSZ::impl::ScatterFromCSR<float>(float* d_A, size_t lenA, size_t ldA, int* nnz, std::string* fi);

void cuSZ::impl::PruneGatherAsCSR(
    float*       d_A,  //
    size_t       lenA,
    const int    m,
    int&         nnzC,
    std::string* fo)
{
    cusparseHandle_t   handle = nullptr;
    cudaStream_t       stream = nullptr;
    cusparseMatDescr_t descrC = nullptr;
    const int          lda    = m;
    const int          n      = m;  // square
    // int*               csrRowPtrC   = nullptr;
    // int*               csrColIndC   = nullptr;
    // float*             csrValC      = nullptr;
    int*   d_csrRowPtrC = nullptr;
    int*   d_csrColIndC = nullptr;
    float* d_csrValC    = nullptr;
    size_t lworkInBytes = 0;
    char*  d_work       = nullptr;

    float threshold = 0; /* remove Aij <= 4.1 */

    // clang-format off
    CHECK_CUDA(cudaStreamCreateWithFlags(   &stream, cudaStreamNonBlocking        )); // 1. create stream
    CHECK_CUSPARSE(cusparseCreate(          &handle                               )); // 2. create handle
    CHECK_CUSPARSE(cusparseSetStream(        handle, stream                       )); // 3. bind stream
    CHECK_CUSPARSE(cusparseCreateMatDescr(  &descrC                               )); // 4. create descr
    CHECK_CUSPARSE(cusparseSetMatIndexBase(  descrC, CUSPARSE_INDEX_BASE_ZERO     )); // zero based
    CHECK_CUSPARSE(cusparseSetMatType(       descrC, CUSPARSE_MATRIX_TYPE_GENERAL )); // typ
    // clang-format on

    CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtrC, sizeof(int) * (m + 1)));

    CHECK_CUSPARSE(cusparseSpruneDense2csr_bufferSizeExt(  //
        handle,                                            //
        m,                                                 //
        n,                                                 //
        d_A,                                               //
        lda,                                               //
        &threshold,                                        //
        descrC,                                            //
        d_csrValC,                                         //
        d_csrRowPtrC,                                      //
        d_csrColIndC,                                      //
        &lworkInBytes));

    //    printf("lworkInBytes (prune) = %lld \n", (long long)lworkInBytes);

    if (nullptr != d_work) {
        cudaFree(d_work);
    }
    CHECK_CUDA(cudaMalloc((void**)&d_work, lworkInBytes));

    /* step 4: compute csrRowPtrC and nnzC */
    CHECK_CUSPARSE(cusparseSpruneDense2csrNnz(  //
        handle,                                 //
        m,                                      //
        n,                                      //
        d_A,                                    //
        lda,                                    //
        &threshold,                             //
        descrC,                                 //
        d_csrRowPtrC,                           //
        &nnzC,                                  // host
        d_work));

    CHECK_CUDA(cudaDeviceSynchronize());

    if (0 == nnzC) cout << log_info << "No outlier." << endl;

    /* step 5: compute csrColIndC and csrValC */
    CHECK_CUDA(cudaMalloc((void**)&d_csrColIndC, sizeof(int) * nnzC));
    CHECK_CUDA(cudaMalloc((void**)&d_csrValC, sizeof(float) * nnzC));

    CHECK_CUSPARSE(cusparseSpruneDense2csr(  //
        handle,                              //
        m,                                   //
        n,                                   //
        d_A,                                 //
        lda,                                 //
        &threshold,                          //
        descrC,                              //
        d_csrValC,                           //
        d_csrRowPtrC,                        //
        d_csrColIndC,                        //
        d_work));
    CHECK_CUDA(cudaDeviceSynchronize());

    /* step 6: output C */
    auto lrp    = sizeof(int) * (m + 1);
    auto lci    = sizeof(int) * nnzC;
    auto lv     = sizeof(float) * nnzC;
    auto ltotal = lrp + lci + lv;
    auto outbin = new uint8_t[ltotal];

    // CHECK_CUDA(cudaMemcpy(csrRowPtrC, d_csrRowPtrC, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaMemcpy(csrColIndC, d_csrColIndC, sizeof(int) * nnzC, cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaMemcpy(csrValC, d_csrValC, sizeof(float) * nnzC, cudaMemcpyDeviceToHost));
    // clang-format off
    CHECK_CUDA(cudaMemcpy(outbin,             d_csrRowPtrC, lrp, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(outbin + lrp,       d_csrColIndC, lci, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(outbin + lrp + lci, d_csrValC,    lv,  cudaMemcpyDeviceToHost));
    // clang-format on

    io::WriteBinaryFile(outbin, ltotal, fo);

    // printCsr(m, n, nnzC, descrC, csrValC, csrRowPtrC, csrColIndC, "C");

    /* free resources */
    if (d_A) cudaFree(d_A);
    if (d_csrRowPtrC) cudaFree(d_csrRowPtrC);
    if (d_csrColIndC) cudaFree(d_csrColIndC);
    if (d_csrValC) cudaFree(d_csrValC);

    if (outbin) delete[] outbin;

    //    for (auto i = 0; i < 200; i++) cout << i << "\t" << csrColIndC[i] << "\t" << csrValC[i] << endl;

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (descrC) cusparseDestroyMatDescr(descrC);
}
