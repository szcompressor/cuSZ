#include <cuda_runtime.h>
#include <cusparse.h>

#include <cxxabi.h>
#include <bitset>
#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "argparse.hh"
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
#include "verify.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

const int gpu_B_1d = 32;
const int gpu_B_2d = 16;
const int gpu_B_3d = 8;

// moved to const_device.cuh
__constant__ int    symb_dims[16];
__constant__ double symb_ebs[4];

typedef std::tuple<size_t, size_t, size_t> tuple3ul;

template <typename T, typename Q>
void cuSZ::impl::PdQ(T* d_data, Q* d_bcode, size_t* dims_L16, double* ebs_L4)
{
    auto  d_dims_L16 = mem::CreateDeviceSpaceAndMemcpyFromHost(dims_L16, 16);
    auto  d_ebs_L4   = mem::CreateDeviceSpaceAndMemcpyFromHost(ebs_L4, 4);
    void* args[]     = {&d_data, &d_bcode, &d_dims_L16, &d_ebs_L4};

    // testing constant memory
    auto dims_inttype = new int[16];
    for (auto i = 0; i < 16; i++) dims_inttype[i] = dims_L16[i];
    cudaMemcpyToSymbol(symb_dims, dims_inttype, 16 * sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(symb_ebs, ebs_L4, 4 * sizeof(double), 0, cudaMemcpyHostToDevice);
    // void* args2[] = {&d_data, &d_bcode}; unreferenced

    if (dims_L16[nDIM] == 1) {
        dim3 blockNum(dims_L16[nBLK0]);
        dim3 threadNum(gpu_B_1d);
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_1d1l<T, Q, gpu_B_1d>,  //
            blockNum, threadNum, args, 0, nullptr);
        /*
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_1d1l<T, Q, gpu_B_1d>,  //
            blockNum, threadNum, args2, gpu_B_1d * sizeof(T), nullptr);
        */
    }
    else if (dims_L16[nDIM] == 2) {
        dim3 blockNum(dims_L16[nBLK0], dims_L16[nBLK1]);
        dim3 threadNum(gpu_B_2d, gpu_B_2d);
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_2d1l<T, Q, gpu_B_2d>,  //
            blockNum, threadNum, args, (gpu_B_2d + 1) * (gpu_B_2d + 1) * sizeof(T), nullptr);
        /*
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_2d1l<T, Q, gpu_B_2d>,  //
            blockNum, threadNum, args2, (gpu_B_2d) * (gpu_B_2d) * sizeof(T), nullptr);
        */
    }
    else if (dims_L16[nDIM] == 3) {
        dim3 blockNum(dims_L16[nBLK0], dims_L16[nBLK1], dims_L16[nBLK2]);
        dim3 threadNum(gpu_B_3d, gpu_B_3d, gpu_B_3d);
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_3d1l<T, Q, gpu_B_3d>,  //
            blockNum, threadNum, args, (gpu_B_3d + 1) * (gpu_B_3d + 1) * (gpu_B_3d + 1) * sizeof(T), nullptr);
        /*
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_3d1l_new<T, Q, gpu_B_3d>,  //
            blockNum, threadNum, args2, (gpu_B_3d + 1) * (gpu_B_3d + 1) * (gpu_B_3d + 1) * sizeof(T), nullptr);
        cudaLaunchKernel(
            (void*)cuSZ::PdQ::c_lorenzo_3d1l<T, Q, gpu_B_3d>,  //
            blockNum, threadNum, args2, (gpu_B_3d) * (gpu_B_3d) * (gpu_B_3d) * sizeof(T), nullptr);
        */
    }
    HANDLE_ERROR(cudaDeviceSynchronize());
}

template void cuSZ::impl::PdQ<float, uint8_t>(float* d_data, uint8_t* d_bcode, size_t* dims_L16, double* ebs_L4);
template void cuSZ::impl::PdQ<float, uint16_t>(float* d_data, uint16_t* d_bcode, size_t* dims_L16, double* ebs_L4);
template void cuSZ::impl::PdQ<float, uint32_t>(float* d_data, uint32_t* d_bcode, size_t* dims_L16, double* ebs_L4);
// template void cuSZ::impl::PdQ<double, uint8_t>(double* d_data, uint8_t* d_bcode, size_t* dims_L16, double* ebs_L4);
// template void cuSZ::impl::PdQ<double, uint16_t>(double* d_data, uint16_t* d_bcode, size_t* dims_L16, double* ebs_L4);
// template void cuSZ::impl::PdQ<double, uint32_t>(double* d_data, uint32_t* d_bcode, size_t* dims_L16, double* ebs_L4);

template <typename T, typename Q>
void cuSZ::impl::ReversedPdQ(T* d_xdata, Q* d_bcode, T* d_outlier, size_t* dims_L16, double _2eb)
{
    auto  d_dims_L16 = mem::CreateDeviceSpaceAndMemcpyFromHost(dims_L16, 16);
    void* args[]     = {&d_xdata, &d_outlier, &d_bcode, &d_dims_L16, &_2eb};

    if (dims_L16[nDIM] == 1) {
        const static size_t p = gpu_B_1d;

        dim3 thread_num(p);
        dim3 block_num((dims_L16[nBLK0] - 1) / p + 1);
        cudaLaunchKernel((void*)PdQ::x_lorenzo_1d1l<T, Q, gpu_B_1d>, block_num, thread_num, args, 0, nullptr);
    }
    else if (dims_L16[nDIM] == 2) {
        const static size_t p = gpu_B_2d;

        dim3 thread_num(p, p);
        dim3 block_num(
            (dims_L16[nBLK0] - 1) / p + 1,   //
            (dims_L16[nBLK1] - 1) / p + 1);  //
        cudaLaunchKernel((void*)PdQ::x_lorenzo_2d1l<T, Q, gpu_B_2d>, block_num, thread_num, args, 0, nullptr);
    }
    else if (dims_L16[nDIM] == 3) {
        const static size_t p = gpu_B_3d;

        dim3 thread_num(p, p, p);
        dim3 block_num(
            (dims_L16[nBLK0] - 1) / p + 1,   //
            (dims_L16[nBLK1] - 1) / p + 1,   //
            (dims_L16[nBLK2] - 1) / p + 1);  //
        cudaLaunchKernel((void*)PdQ::x_lorenzo_3d1l<T, Q, gpu_B_3d>, block_num, thread_num, args, 0, nullptr);
        // PdQ::x_lorenzo_3d1l<T, Q, gpu_B_3d><<<block_num, thread_num>>>(d_xdata, d_outlier, d_bcode, d_dims_L16, _2eb);
    }
    else {
        cerr << log_err << "no 4D" << endl;
    }
    cudaDeviceSynchronize();

    cudaFree(d_dims_L16);
}

template <typename T, typename Q>
void cuSZ::impl::VerifyHuffman(string const& fi, size_t len, Q* xbcode, int chunk_size, size_t* dims_L16, double* ebs_L4)
{
    // TODO error handling from invalid read
    cout << log_info << "Redo PdQ just to get quantization dump." << endl;

    auto veri_data    = io::ReadBinaryFile<T>(fi, len);
    T*   veri_d_data  = mem::CreateDeviceSpaceAndMemcpyFromHost(veri_data, len);
    auto veri_d_bcode = mem::CreateCUDASpace<Q>(len);
    PdQ(veri_d_data, veri_d_bcode, dims_L16, ebs_L4);

    auto veri_bcode = mem::CreateHostSpaceAndMemcpyFromDevice(veri_d_bcode, len);

    auto count = 0;
    for (auto i = 0; i < len; i++)
        if (xbcode[i] != veri_bcode[i]) count++;
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
                if (xbcode[idx] != xbcode[idx]) {
                    if (not chunk_id_printed) {
                        cerr << "chunk id: " << c << "\t";
                        cerr << "start@ " << c * chunk_size << "\tend@ " << (c + 1) * chunk_size - 1 << endl;
                        chunk_id_printed = true;
                    }
                    if (not prev_point_printed) {
                        if (idx != c * chunk_size) {  // not first point
                            cerr << "PREV-idx:" << idx - 1 << "\t" << xbcode[idx - 1] << "\t" << xbcode[idx - 1] << endl;
                        }
                        else {
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
    delete[] veri_bcode;
    delete[] veri_data;
    // end of if count
}

template <typename T, typename Q, typename H>
void cuSZ::workflow::Compress(
    std::string& fi,
    size_t*      dims_L16,
    double*      ebs_L4,
    int&         nnz_outlier,
    size_t&      n_bits,
    size_t&      n_uInt,
    size_t&      huffman_metadata_size,
    argpack*     ap)
{
    int    bw         = sizeof(Q) * 8;
    string fo_cdata   = fi + ".sza";
    string fo_bcode   = fi + ".b" + std::to_string(bw);
    string fo_outlier = fi + ".b" + std::to_string(bw) + "outlier_new";

    // TODO to use a struct
    size_t len         = dims_L16[LEN];
    auto   padded_edge = cuSZ::impl::GetEdgeOfReinterpretedSquare(len);
    auto   padded_len  = padded_edge * padded_edge;

    cout << log_info << "padded edge:\t" << padded_edge << "\tpadded_len:\t" << padded_len << endl;

    auto data = new T[padded_len]();
    io::ReadBinaryFile<T>(fi, data, len);
    T* d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(data, padded_len);

    if (ap->dry_run) {
        cout << "\n" << log_info << "Commencing dry-run..." << endl;
        DryRun(data, d_data, fi, dims_L16, ebs_L4);
        exit(0);
    }
    cout << "\n" << log_info << "Commencing compression..." << endl;

    auto d_bcode = mem::CreateCUDASpace<Q>(len);  // quant. code is not needed for dry-run

    // prediction-quantization
    ::cuSZ::impl::PdQ(d_data, d_bcode, dims_L16, ebs_L4);
    ::cuSZ::impl::GatherAsCSR(d_data, (size_t)padded_len, padded_edge, &nnz_outlier, &fo_outlier);
    // ::cuSZ::impl::GatherOutlierUsingCusparse(d_data, (size_t)padded_len, padded_edge, nnz_outlier, &fo_outlier);

    cout << log_info << "nnz.outlier:\t" << nnz_outlier << "\t(" << (nnz_outlier / 1.0 / len * 100) << "%)" << endl;

    Q* bcode;
    if (ap->skip_huffman) {
        bcode = mem::CreateHostSpaceAndMemcpyFromDevice(d_bcode, len);
        io::WriteBinaryFile(bcode, len, &fo_bcode);
        cout << log_info << "Compression finished, saved quant.code (Huffman skipped).\n" << endl;
        return;
    }

    std::tie(n_bits, n_uInt, huffman_metadata_size) = HuffmanEncode<Q, H>(fo_bcode, d_bcode, len, ap->huffman_chunk, dims_L16[CAP]);

    cout << log_info << "Compression finished, saved Huffman encoded quant.code.\n" << endl;

    delete[] data;
    cudaFree(d_data);
}

template <typename T, typename Q, typename H>
void cuSZ::workflow::Decompress(
    std::string& fi,  //
    size_t*      dims_L16,
    double*      ebs_L4,
    int&         nnz_outlier,
    size_t&      total_bits,
    size_t&      total_uInt,
    size_t&      huffman_metadata_size,
    argpack*     ap)
{
    //    string f_archive = fi + ".sza"; // TODO
    string f_extract = ap->alt_xout_name.empty() ? fi + ".szx" : ap->alt_xout_name;
    string fi_bcode_base, fi_bcode_after_huffman, fi_outlier, fi_outlier_as_cuspm;

    fi_bcode_base       = fi + ".b" + std::to_string(sizeof(Q) * 8);
    fi_outlier_as_cuspm = fi_bcode_base + "outlier_new";

    auto dict_size   = dims_L16[CAP];
    auto len         = dims_L16[LEN];
    auto padded_edge = ::cuSZ::impl::GetEdgeOfReinterpretedSquare(len);
    auto padded_len  = padded_edge * padded_edge;

    cout << log_info << "Commencing decompression..." << endl;

    Q* xbcode;
    // step 1: read from filesystem or do Huffman decoding to get quant code
    if (ap->skip_huffman) {
        cout << log_info << "Getting quant.code from filesystem... (Huffman encoding was skipped.)" << endl;
        xbcode = io::ReadBinaryFile<Q>(fi_bcode_base, len);
    }
    else {
        cout << log_info << "Getting quant.code from Huffman decoding..." << endl;
        xbcode = HuffmanDecode<Q, H>(fi_bcode_base, len, ap->huffman_chunk, total_uInt, dict_size);
        if (ap->verify_huffman) {
            cout << log_info << "Verifying Huffman codec..." << endl;
            ::cuSZ::impl::VerifyHuffman<T, Q>(fi, len, xbcode, ap->huffman_chunk, dims_L16, ebs_L4);
        }
    }
    auto d_bcode = mem::CreateDeviceSpaceAndMemcpyFromHost(xbcode, len);

    auto d_outlier = mem::CreateCUDASpace<T>(padded_len);
    ::cuSZ::impl::ScatterFromCSR<T>(d_outlier, padded_len, padded_edge, &nnz_outlier, &fi_outlier_as_cuspm);

    // TODO merge d_outlier and d_data
    auto d_xdata = mem::CreateCUDASpace<T>(len);
    ::cuSZ::impl::ReversedPdQ(d_xdata, d_bcode, d_outlier, dims_L16, ebs_L4[EBx2]);
    auto xdata = mem::CreateHostSpaceAndMemcpyFromDevice(d_xdata, len);

    cout << log_info << "Decompression finished.\n\n";

    // TODO move CR out of VerifyData
    auto   odata        = io::ReadBinaryFile<T>(fi, len);
    size_t archive_size = 0;
    // TODO huffman chunking metadata
    if (not ap->skip_huffman)
        archive_size += total_uInt * sizeof(H)    // Huffman coded
                        + huffman_metadata_size;  // chunking metadata and reverse codebook
    else
        archive_size += len * sizeof(Q);
    archive_size += nnz_outlier * (sizeof(T) + sizeof(int));

    // TODO g++ and clang++ use mangled type_id name, add macro
    // https://stackoverflow.com/a/4541470/8740097
    auto demangle = [](const char* name) {
        int               status         = -4;
        char*             res            = abi::__cxa_demangle(name, nullptr, nullptr, &status);
        const char* const demangled_name = (status == 0) ? res : name;
        string            ret_val(demangled_name);
        free(res);
        return ret_val;
    };

    if (ap->skip_huffman) {
        cout << log_info << "dtype is \""         //
             << demangle(typeid(T).name())        // demangle
             << "\", and quant. code type is \""  //
             << demangle(typeid(Q).name())        // demangle
             << "\"; a CR of no greater than "    //
             << (sizeof(T) / sizeof(Q)) << " is expected when Huffman codec is skipped." << endl;
    }

    if (ap->pre_binning) cout << log_info << "Because of 2x2->1 binning, extra 4x CR is added." << endl;
    if (not ap->skip_huffman) {
        cout << log_info << "Huffman metadata of chunking and reverse codebook size (in bytes): " << huffman_metadata_size << endl;
        cout << log_info << "Huffman coded output size: " << total_uInt * sizeof(H) << endl;
    }

    analysis::VerifyData(
        xdata, odata, len,  //
        false,              //
        ebs_L4[EB],         //
        archive_size,
        ap->pre_binning ? 4 : 1);  // suppose binning is 2x2

    if (!ap->skip_writex) {
        if (!ap->alt_xout_name.empty()) cout << log_info << "Default decompressed data is renamed from " << string(fi + ".szx") << " to " << f_extract << endl;
        io::WriteBinaryFile(xdata, len, &f_extract);
    }

    // clean up
    delete[] odata;
    delete[] xdata;
    delete[] xbcode;
    cudaFree(d_xdata);
    cudaFree(d_outlier);
    cudaFree(d_bcode);
}

template void cuSZ::workflow::Compress<float, uint8_t, uint32_t>(string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Compress<float, uint8_t, uint64_t>(string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Compress<float, uint16_t, uint32_t>(string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Compress<float, uint16_t, uint64_t>(string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Compress<float, uint32_t, uint32_t>(string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Compress<float, uint32_t, uint64_t>(string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);

template void cuSZ::workflow::Decompress<float, uint8_t, uint32_t>(string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Decompress<float, uint8_t, uint64_t>(string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Decompress<float, uint16_t, uint32_t>(string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Decompress<float, uint16_t, uint64_t>(string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Decompress<float, uint32_t, uint32_t>(string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
template void cuSZ::workflow::Decompress<float, uint32_t, uint64_t>(string&, size_t*, double*, int&, size_t&, size_t&, size_t&, argpack*);
