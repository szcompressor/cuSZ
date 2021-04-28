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

#include <bitset>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "analysis/analyzer.hh"
#include "argparse.hh"
#include "cusz_interface.cuh"
#include "dryrun.cuh"
#include "gather_scatter.cuh"
#include "huff_interface.cuh"
#include "kernel/lorenzo.cuh"
#include "lorenzo_trait.cuh"
#include "metadata.hh"
#include "par_huffman.cuh"
#include "snippets.hh"
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

// namespace fm = cusz::predictor_quantizer;
namespace dr = cusz::dryrun;

namespace draft {
template <typename Huff>
void ExportCodebook(Huff* d_canon_cb, const string& basename, size_t dict_size)
{
    auto              cb_dump = mem::CreateHostSpaceAndMemcpyFromDevice(d_canon_cb, dict_size);
    std::stringstream s;
    s << basename + "-" << dict_size << "-ui" << sizeof(Huff) << ".lean_cb";
    LogAll(log_dbg, "export \"lean\" codebook (of dict_size) as", s.str());
    io::WriteArrayToBinary(s.str(), cb_dump, dict_size);
    delete[] cb_dump;
    cb_dump = nullptr;
}
}  // namespace draft

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
    DataPack<typename DataTrait<If_FP, DataByte>::Data>* datapack,
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

    auto h_data = datapack->hptr();
    auto d_data = datapack->dptr();
    auto m      = datapack->sqrt_ceil;

    auto& wf       = ap->szwf;
    auto& subfiles = ap->subfiles;

    // --------------------------------------------------------------------------------
    // dryrun
    // --------------------------------------------------------------------------------

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

        analysis::VerifyData<Data>(&ap->stat, data_lossy, h_data, len);
        analysis::PrintMetrics<Data>(&ap->stat, false, ap->eb, 0);

        cudaFreeHost(h_data);
        cudaFree(d_data);
        exit(0);
    }
    LogAll(log_info, "invoke lossy-construction");

    // --------------------------------------------------------------------------------
    // constructing quant code
    // --------------------------------------------------------------------------------

    Quant* quant;
    // TODO add hoc padding
    auto d_quant = mem::CreateCUDASpace<Quant>(len + HuffConfig::Db_encode);  // quant. code is not needed for dry-run

    // prediction-quantization
    {
        if (ap->ndim == 1) {
            LorenzoNdConfig<1, Data, workflow::zip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            // seq = 4 for A100
            kernel::c_lorenzo_1d1l_v2<Data, Quant, 4><<<lc.cfg.Dg, lc.cfg.Db.x / 4>>>(lc.z_ctx, d_data, d_quant);
        }
        else if (ap->ndim == 2) {
            LorenzoNdConfig<2, Data, workflow::zip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            kernel::c_lorenzo_2d1l_v1_16x16data_mapto_16x2<Data, Quant>
                <<<lc.cfg.Dg, dim3(16, 2, 1)>>>(lc.z_ctx, d_data, d_quant);
        }
        else if (ap->ndim == 3) {
            LorenzoNdConfig<3, Data, workflow::zip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            kernel::c_lorenzo_3d1l_v1_32x8x8data_mapto_32x1x8<Data, Quant>
                <<<dim3((ap->dim4._0 + 32) / 32, (ap->dim4._1 + 8) / 8, (ap->dim4._2 + 8) / 8), dim3(32, 1, 8)>>>  //
                (lc.z_ctx, d_data, d_quant);
        }
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    // --------------------------------------------------------------------------------
    // gather outlier
    // --------------------------------------------------------------------------------

    // CUDA 10 or earlier //
    {
        auto mxm = datapack->pseudo_matrix_size;
        ::cusz::impl::PruneGatherAsCSR(
            d_data, mxm, m /*lda*/, m /*m*/, m /*n*/, nnz_outlier, &subfiles.compress.out_outlier);
    }

// CUDA 11 onward (hopefully) //
#ifdef TO_REPLACE
    {
        DataPack<Data> sp_csr_val("csr vals");
        DataPack<int>  sp_csr_cols("csr cols");
        DataPack<int>  sp_csr_offsets("csr offsets");

        struct CompressedSparseRow<Data> csr(m, m);  // squarified
        struct DenseMatrix<Data>         mat(m, m);

        sp_csr_offsets.SetLen(csr.num_offsets()).AllocDeviceSpace();  // set sp_csr_offsets size after creating `csr`
        sp_csr_cols.Note(placeholder::length_unknown).Note(placeholder::alloc_in_called_func);
        sp_csr_val.Note(placeholder::length_unknown).Note(placeholder::alloc_in_called_func);

        // set csr and mat afterward
        csr.offsets = sp_csr_offsets.dptr();
        mat.mat     = datapack->dptr();

        SparseOps<Data> op(&mat, &csr);
        op.template Gather<cuSPARSEver::cuda11_onward>();
        auto total_bytelen = op.get_total_bytelen();
        auto outbin        = new u_int8_t[total_bytelen]();
        op.ExportCSR(outbin);
        io::WriteArrayToBinary(subfiles.compress.out_outlier, outbin, total_bytelen);
        delete[] outbin;
        nnz_outlier = csr.sp_size.nnz;
    }
#endif

    auto fmt_nnz = "(" + std::to_string(nnz_outlier / 1.0 / len * 100) + "%)";
    LogAll(log_info, "nnz/#outlier:", nnz_outlier, fmt_nnz, "saved");
    cudaFree(d_data);  // ad-hoc, release memory for large dataset

    // autotuning Huffman chunksize
    int current_dev = 0;
    cudaSetDevice(current_dev);
    cudaDeviceProp dev_prop{};
    cudaGetDeviceProperties(&dev_prop, current_dev);

    auto nSM = dev_prop.multiProcessorCount;
    // auto allowed_thread_per_SM    = dev_prop.maxThreadsPerMultiProcessor;
    auto allowed_thread_per_block = dev_prop.maxThreadsPerBlock;
    // allowed_thread_per_SM * nSM / (HuffConfig::deflate_constant * allowed_thread_per_SM / allowed_thread_per_block);
    auto deflate_nthread    = allowed_thread_per_block * nSM / HuffConfig::deflate_constant;
    auto optimal_chunk_size = (ap->len + deflate_nthread - 1) / deflate_nthread;
    optimal_chunk_size      = ((optimal_chunk_size - 1) / HuffConfig::Db_deflate + 1) * HuffConfig::Db_deflate;
    if (wf.autotune_huffman_chunk) ap->huffman_chunk = optimal_chunk_size;
    LogAll(log_dbg, "Huffman chunk size:", ap->huffman_chunk, "thread num:", (ap->len - 1) / ap->huffman_chunk + 1);

    if (wf.exp_partitioning_imbalance) {
        // 3D only
        unsigned int part0     = ap->part4._0;
        unsigned int part1     = ap->part4._1;
        unsigned int part2     = ap->part4._2;
        unsigned int num_part0 = (ap->dim4._0 - 1) / part0 + 1;
        unsigned int num_part1 = (ap->dim4._1 - 1) / part1 + 1;
        unsigned int num_part2 = (ap->dim4._2 - 1) / part2 + 1;

        LogAll(log_dbg, "p0:", ap->part4._0, " p1:", ap->part4._1, " p2:", ap->part4._2);
        LogAll(log_dbg, "num_part0:", num_part0, " num_part1:", num_part1, " num_part2:", num_part2);

        unsigned int block_stride1 = ap->part4._0, block_stride2 = block_stride1 * ap->part4._0;

        LogAll(log_dbg, "stride1:", ap->stride4._1, " stride2:", ap->stride4._2);
        LogAll(log_dbg, "blockstride1:", block_stride1, " blockstride2:", block_stride2);

        auto buffer_size = part0 * part1 * part2;
        LogAll(log_dbg, "buffer size:", buffer_size);
        auto quant_buffer = new Quant[buffer_size]();

        cudaFree(d_data);
        cudaFreeHost(h_data);

        quant = mem::CreateHostSpaceAndMemcpyFromDevice(d_quant, len);
        cudaFree(d_quant);

        Index<3>::idx_t part_dims{part0, part1, part2};
        Index<3>::idx_t block_strides{1, block_stride1, block_stride2};
        Index<3>::idx_t global_strides{1, ap->stride4._1, ap->stride4._2};

        for (auto pk = 0U; pk < num_part2; pk++) {
            for (auto pj = 0U; pj < num_part1; pj++) {
                for (auto pi = 0U; pi < num_part0; pi++) {
                    auto start = pk * part2 * ap->stride4._2 + pj * part1 * ap->stride4._1 + pi * part0;
                    CopyToBuffer_3D(quant_buffer, quant, start, part_dims, block_strides, global_strides);
                    lossless::interface::HuffmanEncodeWithTree_3D<Quant, Huff>(
                        Index<3>::idx_t{pi, pj, pk}, subfiles.compress.huff_base, quant_buffer, buffer_size,
                        ap->dict_size);
                }
            }
        }

        delete[] quant_buffer;
        delete[] quant;

        exit(0);
    }

    // --------------------------------------------------------------------------------
    // analyze compressibility
    // --------------------------------------------------------------------------------
    // TODO merge this Analyzer instance
    Analyzer analyzer{};

    // histogram
    auto dict_size = ap->dict_size;
    auto d_freq    = mem::CreateCUDASpace<unsigned int>(dict_size);
    // TODO substitute with Analyzer method
    wrapper::GetFrequency(d_quant, len, d_freq, dict_size);

    auto h_freq = mem::CreateHostSpaceAndMemcpyFromDevice(d_freq, dict_size);

    // get codebooks
    static const auto type_bitcount = sizeof(Huff) * 8;
    auto              d_canon_cb    = mem::CreateCUDASpace<Huff>(dict_size, 0xff);
    // first, entry, reversed codebook; TODO CHANGED first and entry to H type
    auto _nbyte       = sizeof(Huff) * (2 * type_bitcount) + sizeof(Quant) * dict_size;
    auto d_reverse_cb = mem::CreateCUDASpace<uint8_t>(_nbyte);
    lossless::par_huffman::ParGetCodebook<Quant, Huff>(dict_size, d_freq, d_canon_cb, d_reverse_cb);
    cudaDeviceSynchronize();

    // analysis
    {
        auto h_canon_cb = mem::CreateHostSpaceAndMemcpyFromDevice(d_canon_cb, dict_size);
        analyzer  //
            .EstimateFromHistogram(h_freq, dict_size)
            .template GetHuffmanCodebookStat<Huff>(h_freq, h_canon_cb, len, dict_size)
            .PrintCompressibilityInfo(true);
    }

    // internal evaluation, not stored in sz archive
    if (wf.exp_export_codebook) draft::ExportCodebook(d_canon_cb, subfiles.compress.huff_base, dict_size);

    delete[] h_freq;

    // --------------------------------------------------------------------------------
    // decide if skipping Huffman coding
    // --------------------------------------------------------------------------------
    if (wf.skip_huffman_enc) {
        quant = mem::CreateHostSpaceAndMemcpyFromDevice(d_quant, len);
        io::WriteArrayToBinary(subfiles.compress.out_quant, quant, len);

        LogAll(log_info, "to store quant.code directly (Huffman enc skipped)");

        return;
    }
    // --------------------------------------------------------------------------------

    std::tie(num_bits, num_uints, huff_meta_size, nvcomp_in_use) = lossless::interface::HuffmanEncode<Quant, Huff>(
        subfiles.compress.huff_base, d_quant, d_canon_cb, d_reverse_cb, _nbyte, len, ap->huffman_chunk,
        wf.lossless_nvcomp_cascade, ap->dict_size);

    LogAll(log_dbg, "to store Huffman encoded quant.code (default)");

    cudaFree(d_quant);
    cudaFree(d_freq), cudaFree(d_canon_cb);
    cudaFree(d_reverse_cb);
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

    auto m   = static_cast<size_t>(ceil(sqrt(ap->len)));
    auto mxm = m * m;

    LogAll(log_info, "invoke lossy-reconstruction");

    DataPack<Quant> quant("quant code");
    quant.SetLen(ap->len).AllocHostSpace().AllocDeviceSpace();

    // step 1: read from filesystem or do Huffman decoding to get quant code
    if (wf.skip_huffman_enc) {
        LogAll(log_info, "load quant.code from filesystem");
        quant.template Move<transfer::fs2h>(subfiles.decompress.in_quant).template Move<transfer::h2d>();
    }
    else {
        LogAll(log_info, "Huffman decode -> quant.code");
        lossless::interface::HuffmanDecode<Quant, Huff>(
            subfiles.path2file, &quant, ap->len, ap->huffman_chunk, total_uint, nvcomp_in_use, ap->dict_size);
        if (wf.verify_huffman) {
            LogAll(log_warn, "Verifying Huffman is temporarily disabled in this version (2021 Week 3");
            /*
            // TODO check in argpack
            if (subfiles.decompress.in_origin == "") {
                cerr << log_err << "use \"--origin /path/to/origin_data\" to specify the original datum." << endl;
                exit(-1);
            }
            cout << log_info << "Verifying Huffman codec..." << endl;
            ::cusz::impl::VerifyHuffman<Data, Quant>(subfiles.decompress.in_origin, len, xq, ap->huffman_chunk, dims,
            eb_variants);
             */
        }
    }

    DataPack<Data> _data("xdata and outlier");
    auto           xdata   = &_data;
    auto           outlier = &_data;

    // need more padding more than pseudo-matrix (for failsafe in reconstruction kernels)
    outlier->SetLen(ap->len).AllocDeviceSpace(mxm + MetadataTrait<1>::Block - ap->len);

    // CUDA 10 or earlier //
    {
        ::cusz::impl::ScatterFromCSR<Data>(
            outlier->dptr(), mxm, m /*lda*/, m /*m*/, m /*n*/, &nnz_outlier, &subfiles.decompress.in_outlier);
    }

// CUDA 11 onward //
#ifdef TO_REPLACE
    {
        struct CompressedSparseRow<Data> csr(m, m, nnz_outlier);
        struct DenseMatrix<Data>         mat(m, m);
        mat.mat = outlier->dptr();

        Index<3>::idx_t trio{
            static_cast<unsigned int>(csr.num_offsets() * sizeof(int)),
            static_cast<unsigned int>(csr.sp_size.columns * sizeof(int)),
            static_cast<unsigned int>(csr.sp_size.values * sizeof(Data))};

        DataPack<uint8_t> _csr_bytes("csr bytes");
        DataPack<int>     sp_csr_offsets("csr_offsets");
        DataPack<int>     sp_csr_cols("csr cols");
        DataPack<Data>    sp_csr_vals("csr vals");

        _csr_bytes.SetLen(trio._0 + trio._1 + trio._2)
            .AllocHostSpace()
            .template Move<transfer::fs2h>(subfiles.decompress.in_outlier);
        sp_csr_offsets  //
            .SetLen(csr.num_offsets())
            .SetHostSpace(reinterpret_cast<int*>(_csr_bytes.hptr()))
            .AllocDeviceSpace()
            .template Move<transfer::h2d>();
        sp_csr_cols  //
            .SetLen(csr.sp_size.columns)
            .SetHostSpace(reinterpret_cast<int*>(_csr_bytes.hptr() + trio._0))
            .AllocDeviceSpace()
            .template Move<transfer::h2d>();
        sp_csr_vals  //
            .SetLen(csr.sp_size.values)
            .SetHostSpace(reinterpret_cast<Data*>(_csr_bytes.hptr() + trio._0 + trio._1))
            .AllocDeviceSpace()
            .template Move<transfer::h2d>();

        csr.offsets = sp_csr_offsets.dptr();
        csr.columns = sp_csr_cols.dptr();
        csr.values  = sp_csr_vals.dptr();

        SparseOps<Data> op(&mat, &csr);
        op.template Scatter<cuSPARSEver::cuda11_onward>();
    }
#endif

    {
        if (ap->ndim == 1) {  // TODO expose problem size and more clear binding with Dg/Db
            LorenzoNdConfig<1, Data, workflow::unzip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            kernel::x_lorenzo_1d1l_cub<Data, Quant><<<lc.cfg.Dg.x, lc.cfg.Db.x / MetadataTrait<1>::Sequentiality>>>  //
                (lc.x_ctx, xdata->dptr(), outlier->dptr(), quant.dptr());
        }
        else if (ap->ndim == 2) {  // y-sequentiality == 8
            LorenzoNdConfig<2, Data, workflow::unzip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            kernel::x_lorenzo_2d1l_v1_16x16data_mapto_16x2<Data, Quant><<<lc.cfg.Dg, dim3(16, 2)>>>  //
                (lc.x_ctx, xdata->dptr(), outlier->dptr(), quant.dptr());
        }
        else if (ap->ndim == 3) {  // y-sequentiality == 8
            LorenzoNdConfig<3, Data, workflow::unzip> lc(ap->dim4, ap->stride4, ap->nblk4, ap->radius, ap->eb);
            kernel::x_lorenzo_3d1l_v4_8x8x8data_mapto_8x1x8<Data, Quant><<<lc.cfg.Dg, dim3(8, 1, 8)>>>  //
                (lc.x_ctx, xdata->dptr(), outlier->dptr(), quant.dptr());

            // kernel::x_lorenzo_3d1l_v5_32x8x8data_mapto_32x1x8                                    //
            //     <<<dim3((ap->dim4._0 + 32) / 32, (ap->dim4._1 + 8) / 8, (ap->dim4._2 + 8) / 8),  //
            //        dim3(32, 1, 8)>>>                                                             //
            //     (lc.x_ctx, xdata->dptr(), outlier->dptr(), quant.dptr());
        }
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    xdata->AllocHostSpace().template Move<transfer::d2h>();

    LogAll(log_info, "reconstruct error-bounded datum");

    size_t archive_bytes = 0;
    // TODO huffman chunking metadata
    if (not wf.skip_huffman_enc)
        archive_bytes += total_uint * sizeof(Huff)  // Huffman coded
                         + huffman_metadata_size;   // chunking metadata and reverse codebook
    else
        archive_bytes += ap->len * sizeof(Quant);
    archive_bytes += nnz_outlier * (sizeof(Data) + sizeof(int)) + (m + 1) * sizeof(int);

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
    if (!subfiles.decompress.in_origin.empty()) {
        LogAll(log_info, "load the original datum for comparison");

        DataPack<Data> odata("original data");
        odata.SetLen(ap->len).AllocHostSpace().template Move<transfer::fs2h>(subfiles.decompress.in_origin);

        analysis::VerifyData(&ap->stat, xdata->hptr(), odata.hptr(), ap->len);
        analysis::PrintMetrics<Data>(&ap->stat, false, ap->eb, archive_bytes, wf.pre_binning ? 4 : 1);
    }
    LogAll(log_info, "output:", subfiles.path2file + ".szx");

    if (wf.skip_write_output)
        LogAll(log_dbg, "skip writing unzipped to filesystem");
    else {
        io::WriteArrayToBinary(subfiles.decompress.out_xdata, xdata, ap->len);
    }
}

namespace szin = cusz::interface;

// TODO top-level instantiation really reduce compilation time?
// clang-format off
template void szin::Compress<true, 4, 1, 4>(argpack*, DataPack<float>*, int&, size_t&, size_t&, size_t&, bool&);
template void szin::Compress<true, 4, 1, 8>(argpack*, DataPack<float>*, int&, size_t&, size_t&, size_t&, bool&);
template void szin::Compress<true, 4, 2, 4>(argpack*, DataPack<float>*, int&, size_t&, size_t&, size_t&, bool&);
template void szin::Compress<true, 4, 2, 8>(argpack*, DataPack<float>*, int&, size_t&, size_t&, size_t&, bool&);

template void szin::Decompress<true, 4, 1, 4>(argpack*, int&, size_t&, size_t&, size_t&, bool);
template void szin::Decompress<true, 4, 1, 8>(argpack*, int&, size_t&, size_t&, size_t&, bool);
template void szin::Decompress<true, 4, 2, 4>(argpack*, int&, size_t&, size_t&, size_t&, bool);
template void szin::Decompress<true, 4, 2, 8>(argpack*, int&, size_t&, size_t&, size_t&, bool);
