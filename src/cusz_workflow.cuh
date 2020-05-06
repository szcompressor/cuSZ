#ifndef CUSZ_WORKFLOW_CUH
#define CUSZ_WORKFLOW_CUH

#include <cuda_runtime.h>
#include <cusparse.h>

#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "__cuda_error_handling.cu"
#include "__io.hh"
#include "cuda_mem.cuh"
#include "cusz_dualquant.cuh"
#include "filter.cuh"
#include "huffman_codec.cuh"
#include "huffman_workflow.cuh"
#include "types.hh"
#include "verify.hh"

template <typename T>
__global__ void CountOutlier(T const* const outlier, int* _d_n_outlier, size_t len) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= len) return;
    if (outlier[gid] != 0) atomicAdd(_d_n_outlier, 1);
    __syncthreads();
}

namespace cuSZ {
namespace FineMassive {

template <typename T, typename Q>
void workflow_PdQ(T* d_data, Q* d_bcode, T* d_outlier, size_t* dims_L16, double* ebs_L4) {
    auto d_dims_L16 = mem::CreateDeviceSpaceAndMemcpyFromHost(dims_L16, 16);
    auto d_ebs_L4   = mem::CreateDeviceSpaceAndMemcpyFromHost(ebs_L4, 4);

    if (dims_L16[nDIM] == 1) {
        dim3 blockNum(dims_L16[nBLK0]);
        dim3 threadNum(B_1d);
        PdQ::c_lorenzo_1d1l<T, Q, B_1d><<<blockNum, threadNum>>>(d_data, d_outlier, d_bcode, d_dims_L16, d_ebs_L4);
    } else if (dims_L16[nDIM] == 2) {
        dim3 blockNum(dims_L16[nBLK0], dims_L16[nBLK1]);
        dim3 threadNum(B_2d, B_2d);
        PdQ::c_lorenzo_2d1l<T, Q, B_2d>                                     //
            <<<blockNum, threadNum, (B_2d + 1) * (B_2d + 1) * sizeof(T)>>>  //
            (d_data, d_outlier, d_bcode, d_dims_L16, d_ebs_L4);
    } else if (dims_L16[nDIM] == 3) {
        dim3 blockNum(dims_L16[nBLK0], dims_L16[nBLK1], dims_L16[nBLK2]);
        dim3 threadNum(B_3d, B_3d, B_3d);
        PdQ::c_lorenzo_3d1l<T, Q, B_3d>                                                  //
            <<<blockNum, threadNum, (B_3d + 1) * (B_3d + 1) * (B_3d + 1) * sizeof(T)>>>  //
            (d_data, d_outlier, d_bcode, d_dims_L16, d_ebs_L4);
    }
    cudaDeviceSynchronize();
}

template <typename T, typename Q>
void workflow_reversePdQ(T* d_xdata, Q* d_bcode, T* d_outlier, size_t* dims_L16, double _2eb) {
    auto d_dims_L16 = mem::CreateDeviceSpaceAndMemcpyFromHost(dims_L16, 16);

    if (dims_L16[nDIM] == 1) {
        const static size_t p = B_1d;

        dim3 thread_num(p);
        dim3 block_num((dims_L16[nBLK0] - 1) / p + 1);
        PdQ::x_lorenzo_1d1l<T, Q, B_1d><<<block_num, thread_num>>>(d_xdata, d_outlier, d_bcode, d_dims_L16, _2eb);
    } else if (dims_L16[nDIM] == 2) {
        const static size_t p = B_2d;

        dim3 thread_num(p, p);
        dim3 block_num((dims_L16[nBLK0] - 1) / p + 1,   //
                       (dims_L16[nBLK1] - 1) / p + 1);  //
        PdQ::x_lorenzo_2d1l<T, Q, B_2d><<<block_num, thread_num>>>(d_xdata, d_outlier, d_bcode, d_dims_L16, _2eb);
    } else if (dims_L16[nDIM] == 3) {
        const static size_t p = B_3d;

        dim3 thread_num(p, p, p);
        dim3 block_num((dims_L16[nBLK0] - 1) / p + 1,   //
                       (dims_L16[nBLK1] - 1) / p + 1,   //
                       (dims_L16[nBLK2] - 1) / p + 1);  //
        PdQ::x_lorenzo_3d1l<T, Q, B_3d><<<block_num, thread_num>>>(d_xdata, d_outlier, d_bcode, d_dims_L16, _2eb);
    } else {
        cerr << log_err << "no 4D" << endl;
    }
    cudaDeviceSynchronize();

    cudaFree(d_dims_L16);
}

template <typename T>
void workflow_DryRun(T* d, T* d_d, string fi, size_t* dims_L16, double* ebs_L4) {
    cout << log_info << "Entering dry-run mode..." << endl;
    auto len        = dims_L16[LEN];
    auto d_dims_L16 = mem::CreateDeviceSpaceAndMemcpyFromHost(dims_L16, 16);
    auto d_ebs_L4   = mem::CreateDeviceSpaceAndMemcpyFromHost(ebs_L4, 4);

    if (dims_L16[nDIM] == 1) {
        dim3 blockNum(dims_L16[nBLK0]);
        dim3 threadNum(B_1d);
        DryRun::lorenzo_1d1l<T><<<blockNum, threadNum>>>(d_d, d_dims_L16, d_ebs_L4);
    } else if (dims_L16[nDIM] == 2) {
        dim3 blockNum(dims_L16[nBLK0], dims_L16[nBLK1]);
        dim3 threadNum(B_2d, B_2d);
        DryRun::lorenzo_2d1l<T><<<blockNum, threadNum>>>(d_d, d_dims_L16, d_ebs_L4);
    } else if (dims_L16[nDIM] == 3) {
        dim3 blockNum(dims_L16[nBLK0], dims_L16[nBLK1], dims_L16[nBLK2]);
        dim3 threadNum(B_3d, B_3d, B_3d);
        DryRun::lorenzo_3d1l<T><<<blockNum, threadNum>>>(d_d, d_dims_L16, d_ebs_L4);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(d, d_d, len * sizeof(T), cudaMemcpyDeviceToHost);

    auto d2 = io::ReadBinaryFile<T>(fi, len);
    Analysis::VerifyData(d, d2, len,  //
                         false,       //
                         ebs_L4[EB],  //
                         0);          // CR is not valid in dry run
    cout << log_info << "Dry-run finished, exit..." << endl;
    delete[] d;
    delete[] d2;
    cudaFree(d_d);
    cudaFree(d_dims_L16);
    cudaFree(d_ebs_L4);
}

template <typename T>
__global__ void Condenser(T* outlier, int* meta, size_t BLK, size_t nBLK) {
    auto id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= nBLK) return;
    int count = 0;
    for (auto i = 0; i < BLK; i++)
        if (outlier[i] != 0) outlier[count++] = outlier[i];

    meta[id] = count;
}

void DeflateOutlierUsingCuSparse(float*  d_A,  //
                                 size_t  len,
                                 int&    nnzC,
                                 int**   csrRowPtrC,
                                 int**   csrColIndC,
                                 float** csrValC) {
    cusparseHandle_t   handle    = nullptr;
    cudaStream_t       stream    = nullptr;
    cusparseMatDescr_t descrC    = nullptr;
    cusparseStatus_t   status    = CUSPARSE_STATUS_SUCCESS;
    cudaError_t        cudaStat1 = cudaSuccess;
    cudaError_t        cudaStat2 = cudaSuccess;
    cudaError_t        cudaStat3 = cudaSuccess;
    //    cudaError_t        cudaStat4 = cudaSuccess;
    //    cudaError_t        cudaStat5 = cudaSuccess;
    const int m   = 1;
    const int n   = len;
    const int lda = m;

    //    int*   csrRowPtrC = nullptr;
    //    int*   csrColIndC = nullptr;
    //    float* csrValC    = nullptr;

    //    float* d_A          = nullptr;
    int*   d_csrRowPtrC = nullptr;
    int*   d_csrColIndC = nullptr;
    float* d_csrValC    = nullptr;

    size_t lworkInBytes = 0;
    char*  d_work       = nullptr;

    //    int nnzC = 0;

    float threshold = 0; /* remove Aij <= 4.1 */

    /* step 1: create cusparse handle, bind a stream */
    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusparseCreate(&handle);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    status = cusparseSetStream(handle, stream);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    /* step 2: configuration of matrix C */
    status = cusparseCreateMatDescr(&descrC);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);

    //    cudaStat1 = cudaMalloc((void**)&d_A, sizeof(float) * lda * n);
    cudaStat2 = cudaMalloc((void**)&d_csrRowPtrC, sizeof(int) * (m + 1));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    /* step 3: query workspace */
    //    cudaStat1 = cudaMemcpy(d_A, A, sizeof(float) * lda * n, cudaMemcpyHostToDevice);
    //    assert(cudaSuccess == cudaStat1);

    status = cusparseSpruneDense2csr_bufferSizeExt(  //
        handle,                                      //
        m,                                           //
        n,                                           //
        d_A,                                         //
        lda,                                         //
        &threshold,                                  //
        descrC,                                      //
        d_csrValC,                                   //
        d_csrRowPtrC,                                //
        d_csrColIndC,                                //
        &lworkInBytes);
    assert(CUSPARSE_STATUS_SUCCESS == status);

    //    printf("lworkInBytes (prune) = %lld \n", (long long)lworkInBytes);

    if (nullptr != d_work) {
        cudaFree(d_work);
    }
    cudaStat1 = cudaMalloc((void**)&d_work, lworkInBytes);
    assert(cudaSuccess == cudaStat1);

    /* step 4: compute csrRowPtrC and nnzC */
    status = cusparseSpruneDense2csrNnz(  //
        handle,                           //
        m,                                //
        n,                                //
        d_A,                              //
        lda,                              //
        &threshold,                       //
        descrC,                           //
        d_csrRowPtrC,                     //
        &nnzC,                            // host
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);

    if (0 == nnzC) cout << log_info << "No outlier." << endl;

    /* step 5: compute csrColIndC and csrValC */
    cudaStat1 = cudaMalloc((void**)&d_csrColIndC, sizeof(int) * nnzC);
    cudaStat2 = cudaMalloc((void**)&d_csrValC, sizeof(float) * nnzC);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    status = cusparseSpruneDense2csr(  //
        handle,                        //
        m,                             //
        n,                             //
        d_A,                           //
        lda,                           //
        &threshold,                    //
        descrC,                        //
        d_csrValC,                     //
        d_csrRowPtrC,                  //
        d_csrColIndC,                  //
        d_work);
    assert(CUSPARSE_STATUS_SUCCESS == status);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);

    /* step 6: output C */
    //    csrRowPtrC = (int*)malloc(sizeof(int) * (m + 1));
    //    csrColIndC = (int*)malloc(sizeof(int) * nnzC);
    //    csrValC    = (float*)malloc(sizeof(float) * nnzC);
    *csrRowPtrC = new int[m + 1];
    *csrColIndC = new int[nnzC];
    *csrValC    = new float[nnzC];
    assert(nullptr != csrRowPtrC);
    assert(nullptr != csrColIndC);
    assert(nullptr != csrValC);

    cudaStat1 = cudaMemcpy(*csrRowPtrC, d_csrRowPtrC, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(*csrColIndC, d_csrColIndC, sizeof(int) * nnzC, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(*csrValC, d_csrValC, sizeof(float) * nnzC, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    //    printCsr(m, n, nnzC, descrC, csrValC, csrRowPtrC, csrColIndC, "C");

    /* free resources */
    if (d_A) cudaFree(d_A);
    if (d_csrRowPtrC) cudaFree(d_csrRowPtrC);
    if (d_csrColIndC) cudaFree(d_csrColIndC);
    if (d_csrValC) cudaFree(d_csrValC);

    //    if (csrRowPtrC) free(csrRowPtrC);
    //    if (csrColIndC) free(csrColIndC);
    //    if (csrValC) free(csrValC);

    //    for (auto i = 0; i < 200; i++) cout << i << "\t" << csrColIndC[i] << "\t" << csrValC[i] << endl;

    if (handle) cusparseDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
    if (descrC) cusparseDestroyMatDescr(descrC);

    //    cudaDeviceReset();
};

template <typename T>
size_t* DeflateOutlier(T* d_outlier, T* outlier, int* meta, size_t len, size_t BLK, size_t nBLK, int blockDim) {
    // get to know num of non-zeros
    int* d_nnz;
    int  nnz = 0;
    cudaMalloc(&d_nnz, sizeof(int));
    cudaMemset(d_nnz, 0, sizeof(int));
    CountOutlier<<<(len - 1) / 256 + 1, 256>>>(d_outlier, d_nnz, len);
    cudaDeviceSynchronize();
    cudaMemcpy(&nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost);
    // deflate
    meta = new int[nBLK]();
    int* d_meta;
    cudaMalloc(&d_meta, nBLK * sizeof(int));
    cudaMemset(d_meta, 0, nBLK = sizeof(int));
    cudaDeviceSynchronize();
    // copy back to host
    outlier = new T[nnz]();

    cudaMemcpy(meta, d_meta, nBLK * sizeof(int), cudaMemcpyDeviceToHost);
    for (auto i = 0, begin = 0; i < nBLK; i++) {
        cudaMemcpy(outlier + begin, d_outlier + i * BLK, meta[i] * sizeof(T), cudaMemcpyDeviceToHost);
        begin += meta[i];
    }

    return outlier;
}

template <typename T, typename Q>
void workflow_verify_huffman(string const& fi, size_t len, Q* xbcode, int chunk_size, size_t* dims_L16, double* ebs_L4) {
    // TODO error handling from invalid read
    cout << log_info << "Redo PdQ just to get quantization dump." << endl;

    auto veri_data   = io::ReadBinaryFile<T>(fi, len);
    T*   veri_d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(veri_data, len);

    auto veri_d_bcode   = mem::CreateCUDASpace<Q>(len);
    auto veri_d_outlier = mem::CreateCUDASpace<T>(len);
    workflow_PdQ(veri_d_data, veri_d_bcode, veri_d_outlier, dims_L16, ebs_L4);

    auto veri_bcode = mem::CreateHostSpaceAndMemcpyFromDevice(veri_d_bcode, len);

    auto count = 0;
    for (auto i = 0; i < len; i++)
        if (xbcode[i] != veri_bcode[i]) count++;
    if (count != 0)
        cerr << log_err << "percentage of not being equal: " << count / (1.0 * len) << "\n";
    else
        cout << log_info << "Decoded correctly." << endl;

    if (count != 0) {
        //        auto chunk_size = ap->huffman_chunk;
        auto n_chunk = (len - 1) / chunk_size + 1;
        for (auto c = 0; c < n_chunk; c++) {
            auto chunk_id_printed   = false;
            auto prev_point_printed = false;
            for (auto i = 0; i < chunk_size; i++) {
                auto idx = i + c * chunk_size;
                if (idx >= len) break;
                if (xbcode[idx] != xbcode[idx]) {
                    if (not chunk_id_printed) {
                        cerr << "chunk id: " << c << "\t";
                        cerr << "start@ " << c * chunk_size << "\tend@ " << (c + 1) * chunk_size - 1 << endl;
                        chunk_id_printed = true;
                    }
                    if (not prev_point_printed) {
                        if (idx != c * chunk_size) {  // not first point
                            cerr << "PREV-idx:" << idx - 1 << "\t" << xbcode[idx - 1] << "\t" << xbcode[idx - 1] << endl;
                        } else {
                            cerr << "wrong at first point!" << endl;
                        }
                        prev_point_printed = true;
                    }
                    cerr << "idx:" << idx << "\tdecoded: " << xbcode[idx] << "\tori: " << xbcode[idx] << endl;
                }
            }
        }
    }

    cudaFree(veri_d_bcode);
    cudaFree(veri_d_data);
    cudaFree(veri_d_outlier);
    delete[] veri_bcode;
    delete[] veri_data;
    // end of if count
}

template <typename T, typename Q = uint16_t, typename H = uint32_t>
void a(std::string&  fi,
       size_t* const dims_L16,
       double* const ebs_L4,
       size_t&       num_outlier,
       size_t&       n_bits,
       size_t&       n_uInt,
       size_t&       huffman_metadata_size,
       argpack*      ap) {
    string fo_bcode, fo_outlier, fo_outlier_new;

    string fo_cdata = fi + ".sza";
    int    bw       = sizeof(Q) * 8;
    fo_bcode        = fi + ".b" + std::to_string(bw);
    fo_outlier_new  = fi + ".b" + std::to_string(bw) + "outlier_new";

    size_t len    = dims_L16[LEN];
    auto   data   = io::ReadBinaryFile<T>(fi, len);
    T*     d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(data, len);

    if (ap->dry_run) {
        cout << "\n" << log_info << "Dry-run commencing..." << endl;
        workflow_DryRun(data, d_data, fi, dims_L16, ebs_L4);
        exit(0);
    }

    cout << "\n" << log_info << "Compression commencing..." << endl;

    auto d_bcode   = mem::CreateCUDASpace<Q>(len);  // quant. code is not needed for dry-run
    auto d_outlier = mem::CreateCUDASpace<T>(len);

    // prediction-quantization
    workflow_PdQ(d_data, d_bcode, d_outlier, dims_L16, ebs_L4);

    // TODO remove file exchange

    // dealing with outlier
#ifdef OLD_OUTLIER_METHOD
    auto outlier = mem::CreateHostSpaceAndMemcpyFromDevice<T>(d_outlier, len);
    for (auto i = 0; i < len; i++)
        if (outlier[i] != 0) num_outlier++;
    cout << log_info << "real num_outlier:\t" << num_outlier << endl;
    io::WriteBinaryFile(outlier, len, &fo_outlier);
#else
    int*   outlier_dummy_csrRowPtrC = nullptr;
    int*   outlier_index_csrColIndC = nullptr;  // column major, real index
    float* outlier_value_csrValC    = nullptr;  // outlier values; TODO template
    int    nnzC                     = 0;

    DeflateOutlierUsingCuSparse(d_outlier, len, nnzC, &outlier_dummy_csrRowPtrC, &outlier_index_csrColIndC, &outlier_value_csrValC);

    num_outlier      = nnzC;  // TODO temporarily nnzC is not archived because num_outlier is available out of this scope
    auto outlier_bin = new uint8_t[nnzC * (sizeof(int) + sizeof(float))];
    memcpy(outlier_bin, (uint8_t*)outlier_index_csrColIndC, nnzC * sizeof(int));
    memcpy(outlier_bin + nnzC * sizeof(int), (uint8_t*)outlier_value_csrValC, nnzC * sizeof(float));
    cout << log_info << "nnz/num.outlier:\t" << num_outlier << "\t(" << (num_outlier / 1.0 / len * 100) << "%)" << endl;
    cout << log_info << "Dumping outlier..." << endl;
    io::WriteBinaryFile(outlier_bin, nnzC * (sizeof(int) + sizeof(float)), &fo_outlier_new);
#endif

    Q* bcode;
    if (ap->skip_huffman) {
        cout << log_info << "Skipping Huffman..." << endl;
        bcode = mem::CreateHostSpaceAndMemcpyFromDevice(d_bcode, len);
        io::WriteBinaryFile(bcode, len, &fo_bcode);
        cout << log_info << "Compression finished.\n" << endl;
        return;
    }

    // huffman encoding
    std::tuple<size_t, size_t, size_t> t = HuffmanEncode<H, Q>(fo_bcode, d_bcode, len, ap->huffman_chunk, dims_L16[CAP]);

    std::tie(n_bits, n_uInt, huffman_metadata_size) = t;

    cout << log_info << "Compression finished.\n" << endl;

    // clean up
    delete[] data;
#ifdef OLD_OUTLIER_METHOD
    delete[] outlier;
#else
    delete[] outlier_index_csrColIndC;
    delete[] outlier_value_csrValC;
    delete[] outlier_dummy_csrRowPtrC;
    delete[] outlier_bin;
#endif

    cudaFree(d_data);
    cudaFree(d_outlier);
}

template <typename T, typename Q = uint16_t, typename H = uint32_t>
void x(std::string const& fi,  //
       size_t*            dims_L16,
       double*            ebs_L4,
       size_t&            num_outlier,
       size_t&            total_bits,
       size_t&            total_uInt,
       size_t&            huffman_metadata_size,
       argpack*           ap) {
    //    string f_archive = fi + ".sza"; // TODO
    string f_extract = ap->alt_xout_name.empty() ? fi + ".szx" : ap->alt_xout_name;
    string fi_bcode_base, fi_bcode_after_huffman, fi_outlier, fi_outlier_new;

    fi_bcode_base  = fi + ".b" + std::to_string(sizeof(Q) * 8);
    fi_outlier_new = fi_bcode_base + "outlier_new";
    //    fi_bcode_after_huffman = fi_bcode_base + ".x";

    auto dict_size = dims_L16[CAP];
    auto len       = dims_L16[LEN];

    cout << log_info << "Decompression commencing..." << endl;

    Q* xbcode;
    // step 1: read from filesystem or do Huffman decoding to get quant code
    if (ap->skip_huffman) {
        cout << log_info << "Huffman skipped, reading quant. code from filesystem..." << endl;
        xbcode = io::ReadBinaryFile<Q>(fi_bcode_base, len);
    } else {
        cout << log_info << "Getting quant. code from Huffman decoding..." << endl;
        xbcode = HuffmanDecode<H, Q>(fi_bcode_base, len, ap->huffman_chunk, total_uInt, dict_size);
        if (ap->verify_huffman) {
            cout << log_info << "Verifying Huffman codec..." << endl;
            workflow_verify_huffman<T, Q>(fi, len, xbcode, ap->huffman_chunk, dims_L16, ebs_L4);
        }
    }
    auto d_bcode = mem::CreateDeviceSpaceAndMemcpyFromHost(xbcode, len);

#ifdef OLD_OUTLIER_METHOD
    auto outlier   = io::ReadBinaryFile<T>(fi_outlier, len);
    auto d_outlier = mem::CreateDeviceSpaceAndMemcpyFromHost(outlier, len);
#else
    auto outlier_bin = io::ReadBinaryFile<uint8_t>(fi_outlier_new, num_outlier * (sizeof(int) + sizeof(float)));
    auto outlier_idx = reinterpret_cast<int*>(outlier_bin);
    auto outlier_val = reinterpret_cast<float*>(outlier_bin + num_outlier * sizeof(int));  // TODO template
    auto outlier     = new T[len]();
    for (auto i = 0; i < num_outlier; i++) outlier[outlier_idx[i]] = outlier_val[i];
    auto d_outlier = mem::CreateDeviceSpaceAndMemcpyFromHost(outlier, len);
#endif

    auto d_xdata = mem::CreateCUDASpace<T>(len);
    workflow_reversePdQ(d_xdata, d_bcode, d_outlier, dims_L16, ebs_L4[EBx2]);
    auto xdata = mem::CreateHostSpaceAndMemcpyFromDevice(d_xdata, len);

    cout << log_info << "Decompression finished." << endl;

    // TODO move CR out of VerifyData
    auto   odata        = io::ReadBinaryFile<T>(fi, len);
    size_t archive_size = 0;
    // TODO huffman chunking metadata
    if (not ap->skip_huffman)
        archive_size += total_uInt * sizeof(H)    // Huffman coded
                        + huffman_metadata_size;  // chunking metadata and reverse codebook
    else
        archive_size += len * sizeof(Q);
    archive_size += num_outlier * (sizeof(T) + sizeof(int));

    if (ap->pre_binning) cout << log_info << "Because of binning (2x2), we have a 4x CR as the normal case." << endl;
    if (not ap->skip_huffman) {
        cout << log_info << "Huffman metadata of chunking and reverse codebook size (in bytes): " << huffman_metadata_size << endl;
        cout << log_info << "Huffman coded output size: " << total_uInt * sizeof(H) << endl;
    }

    Analysis::VerifyData(xdata, odata, len,  //
                         false,              //
                         ebs_L4[EB],         //
                         archive_size,
                         ap->pre_binning ? 4 : 1);  // suppose binning is 2x2

    if (!ap->skip_writex) {
        if (!ap->alt_xout_name.empty())
            cout << log_info << "Default decompressed data is renamed from " << string(fi + ".szx") << " to " << f_extract << endl;
        io::WriteBinaryFile(xdata, len, &f_extract);
    }

    // clean up
    delete[] odata;
    delete[] xdata;
    delete[] outlier;
    delete[] xbcode;
    cudaFree(d_xdata);
    cudaFree(d_outlier);
    cudaFree(d_bcode);
}

}  // namespace FineMassive

}  // namespace cuSZ

#endif
