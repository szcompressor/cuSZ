//#ifndef CUSZ_WORKFLOW_CUH
//#define CUSZ_WORKFLOW_CUH

#include <cuda_runtime.h>

#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <typeinfo>

#include "__cuda_error_handling.cu"
#include "__io.hh"
#include "cusz_dualquant.cuh"
#include "filter.cuh"
#include "huffman_codec.cuh"
#include "psz_dualquant.hh"
#include "types.hh"
#include "verify.hh"

namespace PdQ = cuSZ::PredictionDualQuantization;

namespace cuSZ {
namespace FineMassive {

template <typename T, typename Q, int B>
void demo_c(std::string& fi, size_t* dims_L16, double const* const ebs_L4, size_t& num_outlier) {
    size_t len = dims_L16[LEN];

    string fo_bincode = fi + ".bcode";
    string fo_outlier = fi + ".outlier";

    T*      _d_data           = nullptr;
    T*      _d_downsampled    = nullptr;
    T*      _d_outlier        = nullptr;
    Q*      _d_bincode        = nullptr;
    size_t* _d_dims_L16     = nullptr;
    double* _d_ebs_L4 = nullptr;

    size_t dim1     = dims_L16[DIM1];
    size_t dim0     = dims_L16[DIM0];
    size_t new_dim1 = (dim1 - 1) / 2 + 1;
    size_t new_dim0 = (dim0 - 1) / 2 + 1;
    size_t len_ds   = new_dim1 * new_dim0;

    auto data        = io::ReadBinaryFile<T>(fi, len);
    auto downsampled = new T[len_ds]();

    cudaMalloc(&_d_data, len * sizeof(T));
    cudaMemset(_d_data, 0, len * sizeof(T));
    cudaMemcpy(_d_data, data, len * sizeof(T), cudaMemcpyHostToDevice);

    cudaMalloc(&_d_downsampled, len_ds * sizeof(T));
    cudaMemset(_d_downsampled, 0, len_ds * sizeof(T));

    cudaMalloc(&_d_bincode, len_ds * sizeof(Q));
    cudaMemset(_d_bincode, 0, len_ds * sizeof(Q));

    cudaMalloc(&_d_outlier, len_ds * sizeof(T));
    cudaMemset(_d_outlier, 0, len_ds * sizeof(T));

    cudaMalloc(&_d_ebs_L4, 4 * sizeof(double));
    cudaMemcpy(_d_ebs_L4, ebs_L4, 4 * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&_d_dims_L16, 16 * sizeof(size_t));
    //////// copy metadata for the first time
    cudaMemcpy(_d_dims_L16, dims_L16, 16 * sizeof(size_t), cudaMemcpyHostToDevice);

    ////////////////////////////////////////////////////////////////////////////////
    // binning
    ////////////////////////////////////////////////////////////////////////////////
    dim3 ds_nThread(B, B);
    dim3 ds_nBlock((new_dim0 - 1) / B + 1, (new_dim1 - 1) / B + 1);
    //    cout << ((new_dim1 - 1) / B + 1);
    //    cout << "\t" << ((new_dim0 - 1) / B + 1) << endl;

    if (dim1 % 2 == 0 and dim0 % 2 == 0) {
        //        cout << "even y, even x" << endl;
        Prototype::binning2d_2x2_eveneven  //
            <<<ds_nBlock, ds_nThread>>>    //
            (_d_data, _d_downsampled, dim0, dim1, new_dim0, new_dim1);
    }
    // HANDLE_ERROR(cudaDeviceSynchronize());
    cudaDeviceSynchronize();

    cudaError_t error0 = cudaGetLastError();
    if (error0 != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(error0)), exit(-1);
    cudaMemcpy(downsampled, _d_downsampled, len_ds * sizeof(T), cudaMemcpyDeviceToHost);

    //////// print downsampled to verify
    /*
        size_t _i = 0;
        for_each(downsampled, downsampled + 2, [&](T& d) { cout << (_i++) << ": " << d << endl; });
        for_each(downsampled + new_dim0, downsampled + new_dim0 + 2, [&](T& d) { cout << (_i++) << ": " << d << endl; });
        for_each(downsampled + 2 * new_dim0, downsampled + 2 * new_dim0 + 2, [&](T& d) { cout << (_i++) << ": " << d << endl; });
    */

    //////// update dims
    dims_L16[DIM1]  = new_dim1;
    dims_L16[DIM0]  = new_dim0;
    dims_L16[LEN]   = len_ds;
    dims_L16[nBLK0] = (dims_L16[DIM0] - 1) / B + 1;
    dims_L16[nBLK1] = (dims_L16[DIM1] - 1) / B + 1;

    auto outlier = new T[len_ds]();
    auto bincode = new Q[len_ds]();

    //////// copy metadata of downsampled
    cudaMemcpy(_d_dims_L16, dims_L16, 16 * sizeof(size_t), cudaMemcpyHostToDevice);

    ////////////////////////////////////////////////////////////////////////////////
    // prediction-quantization
    ////////////////////////////////////////////////////////////////////////////////

    PdQ::c_lorenzo_2d1l<T, Q, B>                                          //
        <<<dim3(dims_L16[nBLK0], dims_L16[nBLK1]), dim3(B, B)>>>  //
        (_d_downsampled, _d_outlier, _d_bincode, _d_dims_L16, _d_ebs_L4);
    HANDLE_ERROR(cudaDeviceSynchronize());

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // send back bincode to CPU
    cudaMemcpy(bincode, _d_bincode, len_ds * sizeof(Q), cudaMemcpyDeviceToHost);
    io::WriteBinaryFile(bincode, len_ds, &fo_bincode);

    // send back outlier to CPU
    cudaMemcpy(outlier, _d_outlier, len_ds * sizeof(T), cudaMemcpyDeviceToHost);
    io::WriteBinaryFile(outlier, len_ds, &fo_outlier);

    // count number of outlier points
    for_each(outlier, outlier + len_ds, [&](T& n) { num_outlier += n == 0 ? 0 : 1; });
    cout << "\e[46mnum.outlier:\t" << num_outlier << "\e[0m" << endl;
    cout << "GPU compression done." << endl << endl;

    ////////////////////////////////////////////////////////////////////////////////
    // clean up
    ////////////////////////////////////////////////////////////////////////////////
    delete[] bincode;
    delete[] data;
    delete[] outlier;

    cudaFree(_d_data);
    cudaFree(_d_bincode);
    cudaFree(_d_outlier);
    cudaFree(_d_dims_L16);
    cudaFree(_d_ebs_L4);
}

template <typename T, typename Q, int B>
void demo_x(std::string const& fi, size_t* dims_L16, double* ebs_L4, size_t& num_outlier) {
    string fi_cdata   = fi + ".gpu.sz";  // TODO
    string fi_bincode = fi + ".bcode";
    string fi_outlier = fi + ".outlier";
    string fo_xdata   = fi + ".szx";

    size_t len = dims_L16[LEN];

    auto code     = io::ReadBinaryFile<Q>(fi_bincode, len);
    auto outlier  = io::ReadBinaryFile<T>(fi_outlier, len);
    auto data_ori = io::ReadBinaryFile<T>(fi, len * 4);
    auto xdata    = new T[len]();

    T*   _d_downsampled = nullptr;
    T*   _d_data_ori    = nullptr;
    auto downsampled    = new T[len];

    cudaMalloc(&_d_downsampled, len * sizeof(T));
    cudaMemset(_d_downsampled, 0, len * sizeof(T));

    cudaMalloc(&_d_data_ori, len * 4 * sizeof(T));
    cudaMemset(_d_data_ori, 0, len * 4 * sizeof(T));
    cudaMemcpy(_d_data_ori, data_ori, len * 4 * sizeof(T), cudaMemcpyHostToDevice);

    auto dim0     = dims_L16[DIM0] * 2;
    auto dim1     = dims_L16[DIM1] * 2;
    auto new_dim0 = dims_L16[DIM0];
    auto new_dim1 = dims_L16[DIM1];

    dim3 ds_nThread(B, B);
    dim3 ds_nBlock((new_dim0 - 1) / B + 1, (new_dim1 - 1) / B + 1);
    //    cout << "even y, even x" << endl;
    Prototype::binning2d_2x2_eveneven  //
        <<<ds_nBlock, ds_nThread>>>    //
        (_d_data_ori, _d_downsampled, dim0, dim1, new_dim0, new_dim1);

    cudaMemcpy(downsampled, _d_downsampled, len * sizeof(T), cudaMemcpyDeviceToHost);

    //    cout << "eb: " << ebs_L4[0] << endl;

    //////// CPU de-dual-quant
    for (size_t b1 = 0; b1 < (dims_L16[DIM1] - 1) / B + 1; b1++) {
        for (size_t b0 = 0; b0 < (dims_L16[DIM0] - 1) / B + 1; b0++) {
            pSZ::PredictionDualQuantization::x_lorenzo_2d1l<T, Q, B>(xdata, outlier, code, dims_L16, ebs_L4[EBx2], b0, b1);
        }
    }

    cout << "CPU decompression done." << endl << endl;
    cout << "Analyzing compression quality..." << endl;

    Analysis::VerifyData(xdata, downsampled, len, true, ebs_L4[EB], len * sizeof(Q) + num_outlier * sizeof(T));

    size_t n_overbound = 0;
    for (size_t i = 0; i < len; i++) {
        auto a = downsampled[i] - xdata[i];
        if (fabs(a) > ebs_L4[EB]) {
            //            cout << i << "\toriginal ds'ed: " << downsampled[i] << ",\tdecomp'ed: " << xdata[i] << endl;
            n_overbound++;
        }
    }
    cout << "#overbound:\t" << n_overbound << endl;

    ////////////////////////////////////////////////////////////////////////////////
    // clean up
    ////////////////////////////////////////////////////////////////////////////////
    delete[] xdata;
    delete[] data_ori;
    delete[] outlier;
    delete[] code;
}

}  // namespace FineMassive

}  // namespace cuSZ

//#endif
