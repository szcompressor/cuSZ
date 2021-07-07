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
#include <type_traits>
#include <typeinfo>

#include "analysis/analyzer.hh"
#include "argparse.hh"
#include "cusz_interface.cuh"
#include "kernel/dryrun.h"
#include "kernel/lorenzo.h"
#include "metadata.hh"
#include "type_trait.hh"
#include "utils/cuda_err.cuh"
#include "utils/cuda_mem.cuh"
#include "utils/format.hh"
#include "utils/io.hh"
#include "utils/verify.hh"
#include "wrapper/deprecated_lossless_huffman.h"
#include "wrapper/deprecated_sparsity.h"
#include "wrapper/par_huffman.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

auto demangle = [](const char* name) -> string {
    int   status = -4;
    char* res    = abi::__cxa_demangle(name, nullptr, nullptr, &status);

    const char* const demangled_name = (status == 0) ? res : name;
    string            ret_val(demangled_name);
    free(res);
    return ret_val;
};

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

namespace {
auto get_npart = [](auto size, auto subsize) { return (size + subsize - 1) / subsize; };
}

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
auto copy2buffer3d(
    T* __restrict buffer_dst,
    T* __restrict origin_src,
    size_t          portal,
    Index<3>::idx_t part_dims,
    Index<3>::idx_t block_stride,
    Index<3>::idx_t global_stride)
{
    // clang-format off
    for (auto k = 0; k < part_dims._2; k++) for (auto j = 0; j < part_dims._1; j++) for (auto i = 0; i < part_dims._0; i++)
        buffer_dst[i + j * block_stride._1 + k * block_stride._2] = origin_src[portal + (i + j * global_stride._1 + k * global_stride._2)];
    // clang-format on
}

template <typename T, int N = 3>
auto print_buffer3d(T* data, size_t start, Integer3 strides)
{
    cout << "printing buffer\n";
    // clang-format off
    for (auto k = 0; k < N; k++) { for (auto j = 0; j < N; j++) { for (auto i = 0; i < N; i++) {  //
                cout << data[start + (i + j * strides._1 + k * strides._2)] << " "; }
            cout << "\n";
    }}
    // clang-format on
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
    size_t&  huff_meta_size
    )
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

    auto radius = ap->radius;
    auto eb     = ap->eb;
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / (eb * 2);

    // --------------------------------------------------------------------------------
    // dryrun
    // --------------------------------------------------------------------------------

    if (wf.lossy_dryrun) {
        LogAll(log_info, "invoke dry-run");

        constexpr auto SEQ          = 4;
        constexpr auto DATA_SUBSIZE = 256;
        auto           dim_block    = DATA_SUBSIZE / SEQ;
        auto           dim_grid     = get_npart(len, DATA_SUBSIZE);

        cusz::dual_quant_dryrun<Data, float, DATA_SUBSIZE, SEQ><<<dim_grid, dim_block>>>  //
            (d_data, len, ebx2_r, ebx2);
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

    auto tuple_dim4 = ap->dim4, tuple_stride4 = ap->stride4;
    auto dimx    = tuple_dim4._0;
    auto dimy    = tuple_dim4._1;
    auto dimz    = tuple_dim4._2;
    auto stridey = tuple_stride4._1;
    auto stridez = tuple_stride4._2;

    {  // prediction-quantization
        dim3 dim_block, dim_grid;
        if (ap->ndim == 1) {
            constexpr auto SEQ          = 4;  // y-sequentiality == 4 (A100) or 8
            constexpr auto DATA_SUBSIZE = MetadataTrait<1>::Block;
            dim_block                   = DATA_SUBSIZE / SEQ;
            dim_grid                    = get_npart(dimx, DATA_SUBSIZE);
        }
        else if (ap->ndim == 2) {
            dim_block = dim3(16, 2);  // y-sequentiality == 8
            dim_grid  = dim3(get_npart(dimx, 16), get_npart(dimy, 16));
        }
        else if (ap->ndim == 3) {
            dim_block = dim3(32, 1, 8);  // y-sequentiality == 8
            dim_grid  = dim3(get_npart(dimx, 32), get_npart(dimy, 8), get_npart(dimz, 8));
        }


        if (ap->ndim == 1) {
            constexpr auto SEQ          = 4;  // y-sequentiality == 4 (A100) or 8
            constexpr auto DATA_SUBSIZE = MetadataTrait<1>::Block;
            cusz::c_lorenzo_1d1l<Data, Quant, float, DATA_SUBSIZE, SEQ><<<dim_grid, dim_block>>>  //
                (d_data, d_quant, dimx, radius, ebx2_r);
        }
        else if (ap->ndim == 2) {
            cusz::c_lorenzo_2d1l_16x16data_mapto16x2<Data, Quant, float><<<dim_grid, dim_block>>>  //
                (d_data, d_quant, dimx, dimy, stridey, radius, ebx2_r);
        }
        else if (ap->ndim == 3) {
            cusz::c_lorenzo_3d1l_32x8x8data_mapto32x1x8<Data, Quant><<<dim_grid, dim_block>>>  //
                (d_data, d_quant, dimx, dimy, dimz, stridey, stridez, radius, ebx2_r);
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
    // #ifdef TO_REPLACE
    //     {
    //         DataPack<Data> sp_csr_val("csr vals");
    //         DataPack<int>  sp_csr_cols("csr cols");
    //         DataPack<int>  sp_csr_offsets("csr offsets");

    //         struct CompressedSparseRow<Data> csr(m, m);  // squarified
    //         struct DenseMatrix<Data>         mat(m, m);

    //         sp_csr_offsets.SetLen(csr.num_offsets()).AllocDeviceSpace();  // set sp_csr_offsets size after creating
    //         `csr` sp_csr_cols.Note(placeholder::length_unknown).Note(placeholder::alloc_in_called_func);
    //         sp_csr_val.Note(placeholder::length_unknown).Note(placeholder::alloc_in_called_func);

    //         // set csr and mat afterward
    //         csr.offsets = sp_csr_offsets.dptr();
    //         mat.mat     = datapack->dptr();

    //         SparseOps<Data> op(&mat, &csr);
    //         op.template Gather<cuSPARSEver::cuda11_onward>();
    //         auto total_bytelen = op.get_total_bytelen();
    //         auto outbin        = new u_int8_t[total_bytelen]();
    //         op.ExportCSR(outbin);
    //         io::WriteArrayToBinary(subfiles.compress.out_outlier, outbin, total_bytelen);
    //         delete[] outbin;
    //         nnz_outlier = csr.sp_size.nnz;
    //     }
    // #endif

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
        unsigned int part0 = ap->part4._0, part1 = ap->part4._1, part2 = ap->part4._2;
        unsigned int num_part0 = get_npart(ap->dim4._0, part0), num_part1 = get_npart(ap->dim4._1, part1),
                     num_part2 = get_npart(ap->dim4._2, part2);

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
                    copy2buffer3d(quant_buffer, quant, start, part_dims, block_strides, global_strides);
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

    std::tie(num_bits, num_uints, huff_meta_size) = lossless::interface::HuffmanEncode<Quant, Huff>(
        subfiles.compress.huff_base, d_quant, d_canon_cb, d_reverse_cb, _nbyte, len, ap->huffman_chunk, ap->dict_size);

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
    size_t&  huffman_metadata_size)
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
            subfiles.path2file, &quant, ap->len, ap->huffman_chunk, total_uint, ap->dict_size);
        if (wf.verify_huffman) {
            LogAll(log_warn, "Verifying Huffman is disabled in this version (2021 July");
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

    auto tuple_dim4 = ap->dim4, tuple_stride4 = ap->stride4;
    auto dimx    = tuple_dim4._0;
    auto dimy    = tuple_dim4._1;
    auto dimz    = tuple_dim4._2;
    auto stridey = tuple_stride4._1;
    auto stridez = tuple_stride4._2;

    auto radius = ap->radius;
    auto eb     = ap->eb;
    auto ebx2   = (eb * 2);

    {
        if (ap->ndim == 1) {  // y-sequentiality == 8
            static const auto SEQ          = 8;
            static const auto DATA_SUBSIZE = MetadataTrait<1>::Block;
            auto              dim_block    = DATA_SUBSIZE / SEQ;
            auto              dim_grid     = get_npart(dimx, DATA_SUBSIZE);

            cusz::x_lorenzo_1d1l<Data, Quant, float, DATA_SUBSIZE, SEQ><<<dim_grid, dim_block>>>  //
                (xdata->dptr(), quant.dptr(), dimx, radius, ebx2);
        }
        else if (ap->ndim == 2) {  // y-sequentiality == 8

            auto dim_block = dim3(16, 2);
            auto dim_grid  = dim3(get_npart(dimx, 16), get_npart(dimy, 16));

            cusz::x_lorenzo_2d1l_16x16data_mapto16x2<Data, Quant><<<dim_grid, dim_block>>>  //
                (xdata->dptr(), quant.dptr(), dimx, dimy, stridey, radius, ebx2);
        }
        else if (ap->ndim == 3) {  // y-sequentiality == 8

            auto dim_block = dim3(32, 1, 8);
            auto dim_grid  = dim3(get_npart(dimx, 32), get_npart(dimy, 8), get_npart(dimz, 8));

            cusz::x_lorenzo_3d1l_32x8x8data_mapto32x1x8<<<dim_grid, dim_block>>>  //
                (xdata->dptr(), quant.dptr(), dimx, dimy, dimz, stridey, stridez, radius, ebx2);
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
template void szin::Compress<true, 4, 1, 4>(argpack*, DataPack<float>*, int&, size_t&, size_t&, size_t&);
template void szin::Compress<true, 4, 1, 8>(argpack*, DataPack<float>*, int&, size_t&, size_t&, size_t&);
template void szin::Compress<true, 4, 2, 4>(argpack*, DataPack<float>*, int&, size_t&, size_t&, size_t&);
template void szin::Compress<true, 4, 2, 8>(argpack*, DataPack<float>*, int&, size_t&, size_t&, size_t&);

template void szin::Decompress<true, 4, 1, 4>(argpack*, int&, size_t&, size_t&, size_t&);
template void szin::Decompress<true, 4, 1, 8>(argpack*, int&, size_t&, size_t&, size_t&);
template void szin::Decompress<true, 4, 2, 4>(argpack*, int&, size_t&, size_t&, size_t&);
template void szin::Decompress<true, 4, 2, 8>(argpack*, int&, size_t&, size_t&, size_t&);
