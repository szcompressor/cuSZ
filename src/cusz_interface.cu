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
#include "wrapper/extrap_lorenzo.h"
#include "wrapper/handle_sparsity.h"
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

auto display_throughput(float time, size_t nbyte)
{
    auto throughput = nbyte * 1.0 / (1024 * 1024 * 1024) / (time * 1e-3);
    cout << throughput << "GiB/s\n";
}

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
    // auto m      = datapack->sqrt_ceil;

    auto& wf       = ap->szwf;
    auto& subfiles = ap->subfiles;

    auto radius = ap->radius;
    auto eb     = ap->eb;
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / (eb * 2);

    // dryrun
    if (wf.lossy_dryrun) {
        LogAll(log_info, "invoke dry-run");
        constexpr auto SEQ          = 4;
        constexpr auto DATA_SUBSIZE = 256;
        auto           dim_block    = DATA_SUBSIZE / SEQ;
        auto           dim_grid     = get_npart(len, DATA_SUBSIZE);
        cusz::dual_quant_dryrun<Data, float, DATA_SUBSIZE, SEQ><<<dim_grid, dim_block>>>(d_data, len, ebx2_r, ebx2);
        HANDLE_ERROR(cudaDeviceSynchronize());
        auto data_lossy = new Data[len]();
        cudaMemcpy(data_lossy, d_data, len * sizeof(Data), cudaMemcpyDeviceToHost);
        analysis::VerifyData<Data>(&ap->stat, data_lossy, h_data, len);
        analysis::PrintMetrics<Data>(&ap->stat, false, ap->eb, 0);
        cudaFreeHost(h_data), cudaFree(d_data), exit(0);
    }
    LogAll(log_info, "invoke lossy-construction");

    Quant* quant;
    // TODO add hoc padding
    auto d_quant = mem::CreateCUDASpace<Quant>(len + HuffConfig::Db_encode);  // quant. code is not needed for dry-run

    auto tuple_dim4 = ap->dim4;
    auto dimx       = tuple_dim4._0;
    auto dimy       = tuple_dim4._1;
    auto dimz       = tuple_dim4._2;

    float time_lossy{0}, time_outlier{0}, time_hist{0}, time_book{0}, time_lossless{0};

    // constructing quant code
    {
        compress_lorenzo_construct<Data, Quant, float>(
            d_data, d_quant, dim3(dimx, dimy, dimz), ap->ndim, eb, radius, time_lossy);
    }

    // gather outlier
    {
        struct OutlierDescriptor<Data> csr(len);

        uint8_t *pool, *dump;

        auto dummy_nnz    = len / 10;
        auto pool_bytelen = csr.compress_query_pool_bytelen(dummy_nnz);
        cudaMalloc((void**)&pool, pool_bytelen);
        csr.compress_configure_pool(pool, dummy_nnz);

        compress_gather_CUDA10(&csr, d_data, time_outlier);

        auto dump_bytelen = csr.compress_query_csr_bytelen();
        cudaMallocHost((void**)&dump, dump_bytelen);

        csr.compress_archive_outlier(dump, nnz_outlier);
        io::WriteArrayToBinary(subfiles.compress.out_outlier, dump, dump_bytelen);

        cudaFree(pool), cudaFreeHost(dump);
    }

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

    // analyze compressibility
    // TODO merge this Analyzer instance
    Analyzer analyzer{};

    // histogram
    auto dict_size = ap->dict_size;
    auto d_freq    = mem::CreateCUDASpace<unsigned int>(dict_size);
    // TODO substitute with Analyzer method
    wrapper::GetFrequency(d_quant, len, d_freq, dict_size, time_hist);

    auto h_freq = mem::CreateHostSpaceAndMemcpyFromDevice(d_freq, dict_size);

    // get codebooks
    static const auto type_bitcount = sizeof(Huff) * 8;
    auto              d_canon_cb    = mem::CreateCUDASpace<Huff>(dict_size, 0xff);
    // first, entry, reversed codebook; TODO CHANGED first and entry to H type
    auto _nbyte       = sizeof(Huff) * (2 * type_bitcount) + sizeof(Quant) * dict_size;
    auto d_reverse_cb = mem::CreateCUDASpace<uint8_t>(_nbyte);

    {
        auto t = new cuda_timer_t;
        t->timer_start();
        lossless::par_huffman::ParGetCodebook<Quant, Huff>(dict_size, d_freq, d_canon_cb, d_reverse_cb);
        time_book = t->timer_end_get_elapsed_time();
        cudaDeviceSynchronize();
        delete t;
    }

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

    // decide if skipping Huffman coding
    if (wf.skip_huffman_enc) {
        quant = mem::CreateHostSpaceAndMemcpyFromDevice(d_quant, len);
        io::WriteArrayToBinary(subfiles.compress.out_quant, quant, len);
        LogAll(log_info, "to store quant.code directly (Huffman enc skipped)");
        exit(0);
    }
    // --------------------------------------------------------------------------------

    std::tie(num_bits, num_uints, huff_meta_size) = lossless::interface::HuffmanEncode<Quant, Huff>(
        subfiles.compress.huff_base, d_quant, d_canon_cb, d_reverse_cb, _nbyte, len, ap->huffman_chunk, ap->dict_size,
        time_lossless);

    cout << "\nTIME in milliseconds\t================================================================\n";
    float time_nonbook = time_lossy + time_outlier + time_hist + time_lossless;

    printf("TIME\tconstruct:\t%f\t", time_lossy), display_throughput(time_lossy, len * sizeof(Data));
    printf("TIME\toutlier:\t%f\t", time_outlier), display_throughput(time_outlier, len * sizeof(Data));
    printf("TIME\thistogram:\t%f\t", time_hist), display_throughput(time_hist, len * sizeof(Data));
    printf("TIME\tencode:\t%f\t", time_lossless), display_throughput(time_lossless, len * sizeof(Data));

    cout << "TIME\t--------------------------------------------------------------------------------\n";
    printf("TIME\tnon-book kernels (sum):\t%f\t", time_nonbook), display_throughput(time_nonbook, len * sizeof(Data));
    cout << "TIME\t================================================================================\n";
    printf("TIME\tbuild book (not counted in prev section):\t%f\t", time_book),
        display_throughput(time_book, len * sizeof(Data));
    printf("TIME\t*all* kernels (sum, count book time):\t%f\t", time_nonbook + time_book),
        display_throughput(time_nonbook + time_book, len * sizeof(Data));
    cout << "TIME\t================================================================================\n\n";

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

    float time_lossy{0}, time_outlier{0}, time_lossless{0};

    auto  len      = ap->len;
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
            subfiles.path2file, &quant, ap->len, ap->huffman_chunk, total_uint, ap->dict_size, time_lossless);
        if (wf.verify_huffman) { LogAll(log_warn, "Verifying Huffman is disabled in this version (2021 July"); }
    }

    DataPack<Data> _data("xdata and outlier");
    auto           xdata   = &_data;
    auto           outlier = &_data;

    // need more padding more than pseudo-matrix (for failsafe in reconstruction kernels)
    outlier->SetLen(ap->len).AllocDeviceSpace(mxm + MetadataTrait<1>::Block - ap->len);

    {
        struct OutlierDescriptor<Data> csr(ap->len, nnz_outlier);

        uint8_t *h_csr_file, *d_csr_file;
        cudaMallocHost((void**)&h_csr_file, csr.bytelen.total);
        cudaMalloc((void**)&d_csr_file, csr.bytelen.total);

        io::ReadBinaryToArray<uint8_t>(subfiles.decompress.in_outlier, h_csr_file, csr.bytelen.total);
        cudaMemcpy(d_csr_file, h_csr_file, csr.bytelen.total, cudaMemcpyHostToDevice);

        csr.decompress_extract_outlier(d_csr_file);

        decompress_scatter_CUDA10(&csr, outlier->dptr(), time_outlier);
    }

    auto tuple_dim4 = ap->dim4;
    auto dimx       = tuple_dim4._0;
    auto dimy       = tuple_dim4._1;
    auto dimz       = tuple_dim4._2;

    auto radius = ap->radius;
    auto eb     = ap->eb;
    auto ebx2   = (eb * 2);

    {
        decompress_lorenzo_reconstruct(
            xdata->dptr(), quant.dptr(), dim3(dimx, dimy, dimz), ap->ndim, eb, radius, time_lossy);
    }

    cout << "\nTIME in milliseconds\t================================================================\n";
    float time_all = time_lossy + time_outlier + time_lossless;

    printf("TIME\tscatter outlier:\t%f\t", time_outlier), display_throughput(time_outlier, len * sizeof(Data));
    printf("TIME\tHuffman decode:\t%f\t", time_lossless), display_throughput(time_lossless, len * sizeof(Data));
    printf("TIME\treconstruct:\t%f\t", time_lossy), display_throughput(time_lossy, len * sizeof(Data));

    cout << "TIME\t--------------------------------------------------------------------------------\n";

    printf("TIME\tdecompress (sum):\t%f\t", time_all), display_throughput(time_all, len * sizeof(Data));

    cout << "TIME\t================================================================================\n\n";

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
