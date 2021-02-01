/**
 * @file cusz_workflow.cu
 * @author Jiannan Tian
 * @brief Workflow of cuSZ.
 * @version 0.2
 * @date 2021-01-16
 * (create) 2020-02-12; (release) 2020-09-20; (rev1) 2021-01-16
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

//#include "analysis_utils.hh"
#include "argparse.hh"
#include "autotune.hh"
#include "cusz_interface.cuh"
#include "dryrun.cuh"
#include "dualquant.cuh"
#include "gather_scatter.cuh"
#include "huff_interface.cuh"
#include "lorenzo_trait.cuh"
#include "metadata.hh"
#include "type_trait.hh"
#include "utils/cuda_err.cuh"
#include "utils/cuda_mem.cuh"
#include "utils/format.hh"
#include "utils/io.hh"
#include "utils/verify.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

namespace fm = cusz::predictor_quantizer;
namespace dr = cusz::dryrun;

/*
template <typename Data, typename Quant>
void cusz::impl::VerifyHuffman(
    string const& fi,
    size_t        len,
    Quant*        xq,
    int           chunk_size,
    size_t*       dims,
    double*       eb_variants)
{
    LogAll(log_info, "Redo PdQ just to get quant data.");

    auto  veri_data   = io::ReadBinaryToNewArray<Data>(fi, len);
    Data* veri_d_data = mem::CreateDeviceSpaceAndMemcpyFromHost(veri_data, len);
    auto  veri_d_q    = mem::CreateCUDASpace<Quant>(len);

    PdQ(veri_d_data, veri_d_q, dims, eb_variants);

    auto veri_q = mem::CreateHostSpaceAndMemcpyFromDevice(veri_d_q, len);

    auto count = 0;
    for (auto i = 0; i < len; i++)
        if (xq[i] != veri_q[i]) count++;
    if (count != 0)
        LogAll(log_err, "percentage of not being equal:", count / (1.0 * len));
    else
        LogAll(log_info, "Decoded correctly.");

    if (count != 0) {
        auto n_chunk = (len - 1) / chunk_size + 1;
        for (auto c = 0; c < n_chunk; c++) {
            auto chunk_id_printed = false, prev_point_printed = false;
            for (auto i = 0; i < chunk_size; i++) {
                auto idx = i + c * chunk_size;
                if (idx >= len) break;
                if (xq[idx] != xq[idx]) {
                    if (not chunk_id_printed) {
                        cerr << "chunk id: " << c << "\t"
                             << "start@ " << c * chunk_size << "\tend@ " << (c + 1) * chunk_size - 1 << endl;
                        chunk_id_printed = true;
                    }
                    if (not prev_point_printed) {
                        if (idx != c * chunk_size)  // not first point
                            cerr << "PREV-idx:" << idx - 1 << "\t" << xq[idx - 1] << "\t" << xq[idx - 1] << endl;
                        else
                            cerr << "wrong at first point!" << endl;
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
}
 */

template <typename T>
auto CopyToBuffer_3D(
    T* __restrict buffer_dst,
    T* __restrict origin_src,
    size_t          portal,
    Index<3>::idx_t part_dims,
    Index<3>::idx_t block_stride,
    Index<3>::idx_t global_stride)
{
    for (auto k = 0; k < part_dims._2; k++)
        for (auto j = 0; j < part_dims._1; j++)
            for (auto i = 0; i < part_dims._0; i++)
                buffer_dst[i + j * block_stride._1 + k * block_stride._2] =
                    origin_src[portal + (i + j * global_stride._1 + k * global_stride._2)];
}

template <typename T, int N = 3>
auto PrintBuffer(T* data, size_t start, Integer3 strides)
{
    cout << "printing buffer\n";
    for (auto k = 0; k < N; k++) {
        for (auto j = 0; j < N; j++) {
            for (auto i = 0; i < N; i++) {  //
                cout << data[start + (i + j * strides._1 + k * strides._2)] << " ";
            }
            cout << "\n";
        }
    }
    cout << endl;
};

// clang-format off
template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz::interface::Compress(
    argpack* ap,
    struct DataPack<typename DataTrait<If_FP, DataByte>::Data>* adp,
    int&     nnz_outlier,
    size_t&  num_bits,
    size_t&  num_uints,
    size_t&  huff_meta_size,
    bool&    nvcomp_in_use)
{
    // clang-format on
    using Data  = typename DataTrait<If_FP, DataByte>::Data;
    using Quant = typename QuantTrait<QuantByte>::Quant;
    using Huff  = typename HuffTrait<HuffByte>::Huff;

    size_t len = ap->len;

    auto data   = adp->data;
    auto d_data = adp->d_data;
    auto m      = adp->m;
    auto mxm    = adp->mxm;

    auto& wf       = ap->szwf;
    auto& subfiles = ap->subfiles;

    if (wf.lossy_dryrun) {
        LogAll(log_info, "invoke dry-run");

        if (ap->ndim == 1) {
            LorenzoNdConfig<1, Data, workflow::zip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            dr::lorenzo_1d1l<Data><<<lc.cfg.Dg, lc.cfg.Db, lc.cfg.Ns, lc.cfg.S>>>(lc.r_ctx, d_data);
        }
        else if (ap->ndim == 2) {
            LorenzoNdConfig<2, Data, workflow::zip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            dr::lorenzo_2d1l<Data><<<lc.cfg.Dg, lc.cfg.Db, lc.cfg.Ns, lc.cfg.S>>>(lc.r_ctx, d_data);
        }
        else if (ap->ndim == 3) {
            LorenzoNdConfig<3, Data, workflow::zip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            dr::lorenzo_3d1l<Data><<<lc.cfg.Dg, lc.cfg.Db, lc.cfg.Ns, lc.cfg.S>>>(lc.r_ctx, d_data);
        }
        HANDLE_ERROR(cudaDeviceSynchronize());

        auto data_lossy = new Data[len]();
        cudaMemcpy(data_lossy, d_data, len * sizeof(Data), cudaMemcpyDeviceToHost);

        analysis::VerifyData<Data>(&ap->stat, data_lossy, data, len);
        analysis::PrintMetrics<Data>(&ap->stat, false, ap->eb, 0);

        cudaFreeHost(data);
        cudaFree(d_data);
        exit(0);
    }
    LogAll(log_info, "invoke zipping");

    auto d_q = mem::CreateCUDASpace<Quant>(len);  // quant. code is not needed for dry-run

    // prediction-quantization
    {
        if (ap->ndim == 1) {
            LorenzoNdConfig<1, Data, workflow::zip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            fm::c_lorenzo_1d1l<Data, Quant><<<lc.cfg.Dg, lc.cfg.Db, lc.cfg.Ns, lc.cfg.S>>>(lc.z_ctx, d_data, d_q);
        }
        else if (ap->ndim == 2) {
            LorenzoNdConfig<2, Data, workflow::zip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            fm::c_lorenzo_2d1l<Data, Quant><<<lc.cfg.Dg, lc.cfg.Db, lc.cfg.Ns, lc.cfg.S>>>(lc.z_ctx, d_data, d_q);
        }
        else if (ap->ndim == 3) {
            LorenzoNdConfig<3, Data, workflow::zip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            fm::c_lorenzo_3d1l<Data, Quant><<<lc.cfg.Dg, lc.cfg.Db, lc.cfg.Ns, lc.cfg.S>>>(lc.z_ctx, d_data, d_q);
        }
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    ::cusz::impl::PruneGatherAsCSR(d_data, mxm, m /*lda*/, m /*m*/, m /*n*/, nnz_outlier, &subfiles.c_fo_outlier);

    auto fmt_nnz = "(" + std::to_string(nnz_outlier / 1.0 / len * 100) + "%)";
    LogAll(log_info, "nnz/#outlier:", nnz_outlier, fmt_nnz, "saved");
    cudaFree(d_data);  // ad-hoc, release memory for large dataset

    Quant* q;
    if (wf.skip_huffman_enc) {
        q = mem::CreateHostSpaceAndMemcpyFromDevice(d_q, len);
        io::WriteArrayToBinary(subfiles.c_fo_q, q, len);

        LogAll(log_info, "to store quant.code directly (Huffman enc skipped)");

        return;
    }

    // autotuning Huffman chunksize
    // subject to change, current `8*` is close to but may note deterministically optimal
    if (wf.autotune_huffman_chunk) {  //
        auto optimal_chunksize = 1;
        auto cuda_core_num     = cusz::tune::GetCUDACoreNum();
        auto cuda_thread_num   = 8 * cuda_core_num;  // empirical value

        while (optimal_chunksize * cuda_thread_num < len) optimal_chunksize *= 2;
        ap->huffman_chunk = optimal_chunksize;
    }

    if (wf.exp_partitioning_imbalance) {
        // 3D only
        auto part0     = ap->part4._0;
        auto part1     = ap->part4._1;
        auto part2     = ap->part4._2;
        auto num_part0 = (ap->dim4._0 - 1) / part0 + 1;
        auto num_part1 = (ap->dim4._1 - 1) / part1 + 1;
        auto num_part2 = (ap->dim4._2 - 1) / part2 + 1;

        LogAll(log_dbg, "p0:", ap->part4._0, " p1:", ap->part4._1, " p2:", ap->part4._2);
        LogAll(log_dbg, "num_part0:", num_part0, " num_part1:", num_part1, " num_part2:", num_part2);

        size_t block_stride1 = ap->part4._0, block_stride2 = block_stride1 * ap->part4._0;

        LogAll(log_dbg, "stride1:", ap->stride4._1, " stride2:", ap->stride4._2);
        LogAll(log_dbg, "blockstride1:", block_stride1, " blockstride2:", block_stride2);

        auto buffer_size = part0 * part1 * part2;
        LogAll(log_dbg, "buffer size:", buffer_size);
        auto quant_buffer = new Quant[buffer_size]();

        cudaFree(d_data);
        cudaFreeHost(data);

        q = mem::CreateHostSpaceAndMemcpyFromDevice(d_q, len);
        cudaFree(d_q);

        Index<3>::idx_t part_dims{part0, part1, part2};
        Index<3>::idx_t block_strides{1, (int)block_stride1, (int)block_stride2};
        Index<3>::idx_t global_strides{1, (int)ap->stride4._1, (int)ap->stride4._2};

        for (auto pk = 0; pk < num_part2; pk++) {
            for (auto pj = 0; pj < num_part1; pj++) {
                for (auto pi = 0; pi < num_part0; pi++) {
                    auto start = pk * part2 * ap->stride4._2 + pj * part1 * ap->stride4._1 + pi * part0;
                    CopyToBuffer_3D(quant_buffer, q, start, part_dims, block_strides, global_strides);
                    lossless::interface::HuffmanEncodeWithTree_3D<Quant, Huff>(
                        Index<3>::idx_t{pi, pj, pk}, subfiles.c_huff_base, quant_buffer, buffer_size, ap->dict_size);
                }
            }
        }

        delete[] quant_buffer;
        delete[] q;

        exit(0);
    }

    std::tie(num_bits, num_uints, huff_meta_size, nvcomp_in_use) = lossless::interface::HuffmanEncode<Quant, Huff>(
        subfiles.c_huff_base, d_q, len, ap->huffman_chunk, wf.lossless_nvcomp_cascade, ap->dict_size,
        wf.exp_export_codebook);

    LogAll(log_dbg, "to store Huffman encoded quant.code (default)");

    cudaFree(d_q);
}

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz::interface::Decompress(
    argpack* ap,
    int&     nnz_outlier,
    size_t&  total_bits,
    size_t&  total_uint,
    size_t&  huffman_metadata_size,
    bool     nvcomp_in_use)
{
    using Data  = typename DataTrait<If_FP, DataByte>::Data;
    using Quant = typename QuantTrait<QuantByte>::Quant;
    using Huff  = typename HuffTrait<HuffByte>::Huff;

    auto& wf       = ap->szwf;
    auto& subfiles = ap->subfiles;

    auto m   = ::cusz::impl::GetEdgeOfReinterpretedSquare(ap->len);
    auto mxm = m * m;

    LogAll(log_info, "invoke unzip");

    Quant* xq;
    // step 1: read from filesystem or do Huffman decoding to get quant code
    if (wf.skip_huffman_enc) {
        LogAll(log_info, "load quant.code from filesystem");
        xq = io::ReadBinaryToNewArray<Quant>(subfiles.x_fi_q, ap->len);
    }
    else {
        LogAll(log_info, "Huffman decode -> quant.code");
        xq = lossless::interface::HuffmanDecode<Quant, Huff>(
            subfiles.cx_path2file, ap->len, ap->huffman_chunk, total_uint, nvcomp_in_use, ap->dict_size);
        if (wf.verify_huffman) {
            LogAll(log_warn, "Verifying Huffman is temporarily disabled in this version (2021 Week 3");
            /*
            // TODO check in argpack
            if (subfiles.x_fi_origin == "") {
                cerr << log_err << "use \"--origin /path/to/origin_data\" to specify the original datum." << endl;
                exit(-1);
            }
            cout << log_info << "Verifying Huffman codec..." << endl;
            ::cusz::impl::VerifyHuffman<Data, Quant>(subfiles.x_fi_origin, len, xq, ap->huffman_chunk, dims,
            eb_variants);
             */
        }
    }
    auto d_xq = mem::CreateDeviceSpaceAndMemcpyFromHost(xq, ap->len);

    auto d_outlier = mem::CreateCUDASpace<Data>(mxm);
    ::cusz::impl::ScatterFromCSR<Data>(
        d_outlier, mxm, m /*lda*/, m /*m*/, m /*n*/, &nnz_outlier, &subfiles.x_fi_outlier);

    // TODO merge d_outlier and d_data
    auto d_xdata = mem::CreateCUDASpace<Data>(ap->len);

    {
        // temporary
        if (ap->ndim == 1) {
            LorenzoNdConfig<1, Data, workflow::unzip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            fm::x_lorenzo_1d1l<Data, Quant>
                <<<lc.cfg.Dg, lc.cfg.Db, lc.cfg.Ns, lc.cfg.S>>>(lc.x_ctx, d_xdata, d_outlier, d_xq);
        }
        else if (ap->ndim == 2) {
            LorenzoNdConfig<2, Data, workflow::unzip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            fm::x_lorenzo_2d1l<Data, Quant>
                <<<lc.cfg.Dg, lc.cfg.Db, lc.cfg.Ns, lc.cfg.S>>>(lc.x_ctx, d_xdata, d_outlier, d_xq);
        }
        else if (ap->ndim == 3) {
            LorenzoNdConfig<3, Data, workflow::unzip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            fm::x_lorenzo_3d1l<Data, Quant>
                <<<lc.cfg.Dg, lc.cfg.Db, lc.cfg.Ns, lc.cfg.S>>>(lc.x_ctx, d_xdata, d_outlier, d_xq);
        }
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    auto xdata = mem::CreateHostSpaceAndMemcpyFromDevice(d_xdata, ap->len);

    LogAll(log_info, "reconstruct error-bounded datum");

    size_t archive_bytes = 0;
    // TODO huffman chunking metadata
    if (not wf.skip_huffman_enc)
        archive_bytes += total_uint * sizeof(Huff)  // Huffman coded
                         + huffman_metadata_size;   // chunking metadata and reverse codebook
    else
        archive_bytes += ap->len * sizeof(Quant);
    archive_bytes += nnz_outlier * (sizeof(Data) + sizeof(int)) + (m + 1) * sizeof(int);

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

    if (wf.skip_huffman_enc) {
        cout << log_info << "dtype is \""         //
             << demangle(typeid(Data).name())     // demangle
             << "\", and quant. code type is \""  //
             << demangle(typeid(Quant).name())    // demangle
             << "\"; a CR of no greater than "    //
             << (sizeof(Data) / sizeof(Quant)) << " is expected when Huffman codec is skipped." << endl;
    }

    if (wf.pre_binning) cout << log_info << "Because of 2x2->1 binning, extra 4x CR is added." << endl;

    // TODO move CR out of VerifyData
    if (subfiles.x_fi_origin != "") {
        LogAll(log_info, "load the original datum for comparison");

        auto odata = io::ReadBinaryToNewArray<Data>(subfiles.x_fi_origin, ap->len);
        analysis::VerifyData(&ap->stat, xdata, odata, ap->len);
        analysis::PrintMetrics<Data>(&ap->stat, false, ap->eb, archive_bytes, wf.pre_binning ? 4 : 1);

        delete[] odata;
    }
    LogAll(log_info, "output:", subfiles.cx_path2file + ".szx");

    if (wf.skip_write_output)
        io::WriteArrayToBinary(subfiles.x_fo_xd, xdata, ap->len);
    else {
        LogAll(log_dbg, "skipped writing unzipped to filesystem");
    }

    // clean up
    delete[] xdata;
    delete[] xq;
    cudaFree(d_xdata);
    cudaFree(d_outlier);
    cudaFree(d_xq);
}

typedef struct DataPack<float> adp_f32_t;
namespace szin = cusz::interface;

// TODO top-level instantiation really reduce compilation time?
// clang-format off
template void szin::Compress<true, 4, 1, 4>(argpack*, adp_f32_t*, int&, size_t&, size_t&, size_t&, bool&);
template void szin::Compress<true, 4, 1, 8>(argpack*, adp_f32_t*, int&, size_t&, size_t&, size_t&, bool&);
template void szin::Compress<true, 4, 2, 4>(argpack*, adp_f32_t*, int&, size_t&, size_t&, size_t&, bool&);
template void szin::Compress<true, 4, 2, 8>(argpack*, adp_f32_t*, int&, size_t&, size_t&, size_t&, bool&);

template void szin::Decompress<true, 4, 1, 4>(argpack*, int&, size_t&, size_t&, size_t&, bool);
template void szin::Decompress<true, 4, 1, 8>(argpack*, int&, size_t&, size_t&, size_t&, bool);
template void szin::Decompress<true, 4, 2, 4>(argpack*, int&, size_t&, size_t&, size_t&, bool);
template void szin::Decompress<true, 4, 2, 8>(argpack*, int&, size_t&, size_t&, size_t&, bool);
