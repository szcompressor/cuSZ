/**
 * @file cusz_workflow.cu
 * @author Jiannan Tian
 * @brief Workflow of cuSZ.
 * @version 0.1
 * @date 2020-09-20
 * Created on: 2020-02-12
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cxxabi.h>
#include <bitset>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <typeinfo>

// #if __cplusplus >= 201103L

#include <type_traits>
#include "ad_hoc_types.hh"
#include "analysis_utils.hh"
#include "argparse.hh"
#include "autotune.h"
#include "constants.hh"
#include "cuda_error_handling.cuh"
#include "cuda_mem.cuh"
#include "cusz_dryrun.cuh"
#include "cusz_dualquant.cuh"
#include "cusz_workflow.cuh"
#include "filter.cuh"
#include "format.hh"
#include "gather_scatter.cuh"
#include "huffman_workflow.cuh"
#include "io.hh"
#include "timer.hh"
#include "verify.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using uint8__t = uint8_t;

const int gpu_B_1d = 256;
const int gpu_B_2d = 16;
const int gpu_B_3d = 8;

typedef std::tuple<size_t, size_t, size_t> tuple3ul;

template <typename Data, typename Quant>
void cusz::impl::PdQ(Data* d_d, Quant* d_q, size_t* dims, double* eb_variants)
{
    auto  d_dims        = mem::CreateDeviceSpaceAndMemcpyFromHost(dims, 16);
    auto  d_eb_variants = mem::CreateDeviceSpaceAndMemcpyFromHost(eb_variants, 4);
    void* args[]        = {&d_d, &d_q, &d_dims, &d_eb_variants};

    if (dims[nDIM] == 1) {
        dim3 block_num(dims[nBLK0]);
        dim3 thread_num(gpu_B_1d);
        cudaLaunchKernel(
            (void*)cusz::predictor_quantizer::c_lorenzo_1d1l<Data, Quant, gpu_B_1d>,  //
            block_num, thread_num, args, 0, nullptr);
    }
    else if (dims[nDIM] == 2) {
        dim3 block_num(dims[nBLK0], dims[nBLK1]);
        dim3 thread_num(gpu_B_2d, gpu_B_2d);
        cudaLaunchKernel(
            (void*)cusz::predictor_quantizer::c_lorenzo_2d1l<Data, Quant, gpu_B_2d>,  //
            block_num, thread_num, args, (gpu_B_2d + 1) * (gpu_B_2d + 1) * sizeof(Data), nullptr);
    }
    else if (dims[nDIM] == 3) {
        dim3 block_num(dims[nBLK0], dims[nBLK1], dims[nBLK2]);
        dim3 thread_num(gpu_B_3d, gpu_B_3d, gpu_B_3d);
        cudaLaunchKernel(
            (void*)cusz::predictor_quantizer::c_lorenzo_3d1l<Data, Quant, gpu_B_3d>,  //
            block_num, thread_num, args, (gpu_B_3d + 1) * (gpu_B_3d + 1) * (gpu_B_3d + 1) * sizeof(Data), nullptr);
    }
    HANDLE_ERROR(cudaDeviceSynchronize());
}

template <typename Data, typename Quant>
void cusz::impl::ReversedPdQ(Data* d_xd, Quant* d_q, Data* d_outlier, size_t* dims, double _2eb)
{
    auto  d_dims = mem::CreateDeviceSpaceAndMemcpyFromHost(dims, 16);
    void* args[] = {&d_xd, &d_outlier, &d_q, &d_dims, &_2eb};

    if (dims[nDIM] == 1) {
        const static size_t p = gpu_B_1d;

        dim3 thread_num(p);
        dim3 block_num((dims[nBLK0] - 1) / p + 1);
        cudaLaunchKernel(
            (void*)cusz::predictor_quantizer::x_lorenzo_1d1l<Data, Quant, gpu_B_1d>, block_num, thread_num, args, 0,
            nullptr);
    }
    else if (dims[nDIM] == 2) {
        const static size_t p = gpu_B_2d;

        dim3 thread_num(p, p);
        dim3 block_num(
            (dims[nBLK0] - 1) / p + 1,   //
            (dims[nBLK1] - 1) / p + 1);  //
        cudaLaunchKernel(
            (void*)cusz::predictor_quantizer::x_lorenzo_2d1l<Data, Quant, gpu_B_2d>, block_num, thread_num, args, 0,
            nullptr);
    }
    else if (dims[nDIM] == 3) {
        const static size_t p = gpu_B_3d;

        dim3 thread_num(p, p, p);
        dim3 block_num(
            (dims[nBLK0] - 1) / p + 1,   //
            (dims[nBLK1] - 1) / p + 1,   //
            (dims[nBLK2] - 1) / p + 1);  //
        cudaLaunchKernel(
            (void*)cusz::predictor_quantizer::x_lorenzo_3d1l<Data, Quant, gpu_B_3d>, block_num, thread_num, args, 0,
            nullptr);
    }
    else {
        cerr << log_err << "no 4D" << endl;
    }
    cudaDeviceSynchronize();

    cudaFree(d_dims);
}

template <typename Data, typename Quant>
void cusz::impl::VerifyHuffman(
    string const& fi,
    size_t        len,
    Quant*        xq,
    int           chunk_size,
    size_t*       dims,
    double*       eb_variants)
{
    // TODO error handling from invalid read
    cout << log_info << "Redo PdQ just to get quantization dump." << endl;

    auto  veri_data   = io::ReadBinaryFile<Data>(fi, len);
    Data* veri_d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(veri_data, len);
    auto  veri_d_q    = mem::CreateCUDASpace<Quant>(len);
    PdQ(veri_d_data, veri_d_q, dims, eb_variants);

    auto veri_q = mem::CreateHostSpaceAndMemcpyFromDevice(veri_d_q, len);

    auto count = 0;
    for (auto i = 0; i < len; i++)
        if (xq[i] != veri_q[i]) count++;
    if (count != 0)
        cerr << log_err << "percentage of not being equal: " << count / (1.0 * len) << "\n";
    else
        cout << log_info << "Decoded correctly." << endl;

    if (count != 0) {
        // auto chunk_size = ap->huffman_chunk;
        auto n_chunk = (len - 1) / chunk_size + 1;
        for (auto c = 0; c < n_chunk; c++) {
            auto chunk_id_printed   = false;
            auto prev_point_printed = false;
            for (auto i = 0; i < chunk_size; i++) {
                auto idx = i + c * chunk_size;
                if (idx >= len) break;
                if (xq[idx] != xq[idx]) {
                    if (not chunk_id_printed) {
                        cerr << "chunk id: " << c << "\t";
                        cerr << "start@ " << c * chunk_size << "\tend@ " << (c + 1) * chunk_size - 1 << endl;
                        chunk_id_printed = true;
                    }
                    if (not prev_point_printed) {
                        if (idx != c * chunk_size) {  // not first point
                            cerr << "PREV-idx:" << idx - 1 << "\t" << xq[idx - 1] << "\t" << xq[idx - 1] << endl;
                        }
                        else {
                            cerr << "wrong at first point!" << endl;
                        }
                        prev_point_printed = true;
                    }
                    cerr << "idx:" << idx << "\tdecoded: " << xq[idx] << "\tori: " << xq[idx] << endl;
                }
            }
        }
    }

    cudaFree(veri_d_q);
    cudaFree(veri_d_data);
    delete[] veri_q;
    delete[] veri_data;
    // end of if count
}

template <typename Data, typename Quant, typename Huff>
void cusz::interface::Compress(
    argpack*                    ap,
    struct AdHocDataPack<Data>* adp,
    size_t*                     dims,
    double*                     eb_variants,
    int&                        nnz_outlier,
    size_t&                     n_bits,
    size_t&                     n_uInt,
    size_t&                     huffman_metadata_size)
{
    // TODO to use a struct
    // TODO already calculated outside in main()
    size_t len = dims[LEN];

    auto data   = adp->data;
    auto d_data = adp->d_data;
    auto m      = adp->m;
    auto mxm    = adp->mxm;

    if (ap->to_dryrun) {
        logall(log_info, "invoke dry-run");
        DryRun(data, d_data, ap->cx_path2file, dims, eb_variants);
        cudaFreeHost(data);
        cudaFree(d_data);
        exit(0);
    }
    logall(log_info, "invoke zipping");

    auto d_q = mem::CreateCUDASpace<Quant>(len);  // quant. code is not needed for dry-run

    // prediction-quantization
    ::cusz::impl::PdQ(d_data, d_q, dims, eb_variants);
    ::cusz::impl::PruneGatherAsCSR(d_data, mxm, m /*lda*/, m /*m*/, m /*n*/, nnz_outlier, &ap->c_fo_outlier);

    auto fmt_nnz = "(" + std::to_string(nnz_outlier / 1.0 / len * 100) + "%)";
    logall(log_info, "nnz/#outlier:", nnz_outlier, fmt_nnz, "saved");
    cudaFree(d_data);  // ad-hoc, release memory for large dataset

    Quant* q;
    if (ap->skip_huffman) {
        q = mem::CreateHostSpaceAndMemcpyFromDevice(d_q, len);
        io::WriteArrayToBinary(ap->c_fo_q, q, len);

        logall(log_info, "to store quant.code directly (Huffman enc skipped)");

        return;
    }

    // autotuning Huffman chunksize
    // subject to change, current `8*` is close to but may note deterministically optimal
    if (ap->autotune_huffman_chunk) {  //
        auto optimal_chunksize = 1;
        auto cuda_core_num     = cusz::tune::GetCUDACoreNum();
        auto cuda_thread_num   = 8 * cuda_core_num;  // empirical value

        while (optimal_chunksize * cuda_thread_num < len) optimal_chunksize *= 2;
        ap->huffman_chunk = optimal_chunksize;
    }

    std::tie(n_bits, n_uInt, huffman_metadata_size) =
        lossless::interface::HuffmanEncode<Quant, Huff>(ap->c_huff_base, d_q, len, ap->huffman_chunk, dims[CAP]);

    logall(log_dbg, "to store Huffman encoded quant.code (default)");

    cudaFree(d_q);
}

template <typename Data, typename Quant, typename Huff>
void cusz::interface::Decompress(
    argpack* ap,
    size_t*  dims,
    double*  eb_variants,
    int&     nnz_outlier,
    size_t&  total_bits,
    size_t&  total_uInt,
    size_t&  huffman_metadata_size)
{
    auto dict_size = dims[CAP];
    auto len       = dims[LEN];
    auto m         = ::cusz::impl::GetEdgeOfReinterpretedSquare(len);
    auto mxm       = m * m;

    logall(log_info, "invoke unzip");

    Quant* xq;
    // step 1: read from filesystem or do Huffman decoding to get quant code
    if (ap->skip_huffman) {
        logall(log_info, "load quant.code from filesystem");
        xq = io::ReadBinaryFile<Quant>(ap->x_fi_q, len);
    }
    else {
        logall(log_info, "Huffman decode -> quant.code");
        xq = lossless::interface::HuffmanDecode<Quant, Huff>(
            ap->cx_path2file, len, ap->huffman_chunk, total_uInt, dict_size);
        if (ap->verify_huffman) {
            // TODO check in argpack
            if (ap->x_fi_origin == "") {
                cerr << log_err << "use \"--origin /path/to/origin_data\" to specify the original datum." << endl;
                exit(-1);
            }
            cout << log_info << "Verifying Huffman codec..." << endl;
            ::cusz::impl::VerifyHuffman<Data, Quant>(ap->x_fi_origin, len, xq, ap->huffman_chunk, dims, eb_variants);
        }
    }
    auto d_xq = mem::CreateDeviceSpaceAndMemcpyFromHost(xq, len);

    auto d_outlier = mem::CreateCUDASpace<Data>(mxm);
    ::cusz::impl::ScatterFromCSR<Data>(d_outlier, mxm, m /*lda*/, m /*m*/, m /*n*/, &nnz_outlier, &ap->x_fi_outlier);

    // TODO merge d_outlier and d_data
    auto d_xdata = mem::CreateCUDASpace<Data>(len);
    ::cusz::impl::ReversedPdQ(d_xdata, d_xq, d_outlier, dims, eb_variants[EBx2]);
    auto xdata = mem::CreateHostSpaceAndMemcpyFromDevice(d_xdata, len);

    logall(log_info, "reconstruct error-bounded datum");

    size_t archive_size = 0;
    // TODO huffman chunking metadata
    if (not ap->skip_huffman)
        archive_size += total_uInt * sizeof(Huff)  // Huffman coded
                        + huffman_metadata_size;   // chunking metadata and reverse codebook
    else
        archive_size += len * sizeof(Quant);
    archive_size += nnz_outlier * (sizeof(Data) + sizeof(int)) + (m + 1) * sizeof(int);

    // TODO g++ and clang++ use mangled type_id name, add macro
    // https://stackoverflow.com/a/4541470/8740097
    auto demangle = [](const char* name) {
        int   status = -4;
        char* res    = abi::__cxa_demangle(name, nullptr, nullptr, &status);

        const char* const demangled_name = (status == 0) ? res : name;
        string            ret_val(demangled_name);
        free(res);
        return ret_val;
    };

    if (ap->skip_huffman) {
        cout << log_info << "dtype is \""         //
             << demangle(typeid(Data).name())     // demangle
             << "\", and quant. code type is \""  //
             << demangle(typeid(Quant).name())    // demangle
             << "\"; a CR of no greater than "    //
             << (sizeof(Data) / sizeof(Quant)) << " is expected when Huffman codec is skipped." << endl;
    }

    if (ap->pre_binning) cout << log_info << "Because of 2x2->1 binning, extra 4x CR is added." << endl;

    // TODO move CR out of VerifyData
    if (ap->x_fi_origin != "") {
        logall(log_info, "load the original datum for comparison");

        auto odata = io::ReadBinaryFile<Data>(ap->x_fi_origin, len);
        analysis::VerifyData(
            xdata, odata,
            len,              //
            false,            //
            eb_variants[EB],  //
            archive_size,
            ap->pre_binning ? 4 : 1);  // TODO use template rather than 2x2
        delete[] odata;
    }
    logall(log_info, "output:", ap->cx_path2file + ".szx");

    if (!ap->skip_writex)
        io::WriteArrayToBinary(ap->x_fo_xd, xdata, len);
    else {
        logall(log_dbg, "skipped writing unzipped to filesystem");
    }

    // clean up
    delete[] xdata;
    delete[] xq;
    cudaFree(d_xdata);
    cudaFree(d_outlier);
    cudaFree(d_xq);
}

typedef struct AdHocDataPack<float> adp_f32_t;
namespace szin = cusz::interface;

template void szin::Compress<FP4, UI1, UI4>(argpack*, adp_f32_t*, size_t*, FP8*, int&, size_t&, size_t&, size_t&);
template void szin::Compress<FP4, UI1, UI8>(argpack*, adp_f32_t*, size_t*, FP8*, int&, size_t&, size_t&, size_t&);
template void szin::Compress<FP4, UI1, UI8_2>(argpack*, adp_f32_t*, size_t*, FP8*, int&, size_t&, size_t&, size_t&);
template void szin::Compress<FP4, UI2, UI4>(argpack*, adp_f32_t*, size_t*, FP8*, int&, size_t&, size_t&, size_t&);
template void szin::Compress<FP4, UI2, UI8>(argpack*, adp_f32_t*, size_t*, FP8*, int&, size_t&, size_t&, size_t&);
template void szin::Compress<FP4, UI2, UI8_2>(argpack*, adp_f32_t*, size_t*, FP8*, int&, size_t&, size_t&, size_t&);

template void szin::Decompress<FP4, UI1, UI4>(argpack*, size_t*, FP8*, int&, size_t&, size_t&, size_t&);
template void szin::Decompress<FP4, UI1, UI8>(argpack*, size_t*, FP8*, int&, size_t&, size_t&, size_t&);
template void szin::Decompress<FP4, UI1, UI8_2>(argpack*, size_t*, FP8*, int&, size_t&, size_t&, size_t&);
template void szin::Decompress<FP4, UI2, UI4>(argpack*, size_t*, FP8*, int&, size_t&, size_t&, size_t&);
template void szin::Decompress<FP4, UI2, UI8>(argpack*, size_t*, FP8*, int&, size_t&, size_t&, size_t&);
template void szin::Decompress<FP4, UI2, UI8_2>(argpack*, size_t*, FP8*, int&, size_t&, size_t&, size_t&);

// #endif