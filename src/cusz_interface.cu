/**
 * @file cusz_workflow.cu
 * @author Jiannan Tian
 * @brief Workflow of cuSZ.
 * @version 0.3
 * @date 2021-07-12
 * (create) 2020-02-12; (release) 2020-09-20; (rev.1) 2021-01-16; (rev.2) 2021-07-12
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
#include "cusz_interface.h"
#include "kernel/dryrun.h"
#include "kernel/lorenzo.h"
#include "metadata.hh"
#include "type_trait.hh"
#include "utils.hh"
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

template <typename Quant, typename Huff>
unsigned int get_revbook_nbyte(unsigned dict_size)
{
    constexpr auto type_bitcount = sizeof(Huff) * 8;
    return sizeof(Huff) * (2 * type_bitcount) + sizeof(Quant) * dict_size;
}

namespace draft {
template <typename Huff>
void export_codebook(Huff* d_book, const string& basename, size_t dict_size)
{
    auto              h_book = mem::create_devspace_memcpy_d2h(d_book, dict_size);
    std::stringstream s;
    s << basename + "-" << dict_size << "-ui" << sizeof(Huff) << ".lean-book";
    logging(log_dbg, "export \"lean\" codebook (of dict_size) as", s.str());
    io::write_array_to_binary(s.str(), h_book, dict_size);
    delete[] h_book;
    h_book = nullptr;
}
}  // namespace draft

namespace {
auto get_npart = [](auto size, auto subsize) { return (size + subsize - 1) / subsize; };

template <typename Data>
void report_compression_time(size_t len, float lossy, float outlier, float hist, float book, float lossless)
{
    cout << "\nTIME in milliseconds\t================================================================\n";
    float nonbook = lossy + outlier + hist + lossless;

    printf("TIME\tconstruct:\t%f\t", lossy), display_throughput(lossy, len * sizeof(Data));
    printf("TIME\toutlier:\t%f\t", outlier), display_throughput(outlier, len * sizeof(Data));
    printf("TIME\thistogram:\t%f\t", hist), display_throughput(hist, len * sizeof(Data));
    printf("TIME\tencode:\t%f\t", lossless), display_throughput(lossless, len * sizeof(Data));

    cout << "TIME\t--------------------------------------------------------------------------------\n";
    printf("TIME\tnon-book kernels (sum):\t%f\t", nonbook), display_throughput(nonbook, len * sizeof(Data));
    cout << "TIME\t================================================================================\n";
    printf("TIME\tbuild book (not counted in prev section):\t%f\t", book), display_throughput(book, len * sizeof(Data));
    printf("TIME\t*all* kernels (sum, count book time):\t%f\t", nonbook + book),
        display_throughput(nonbook + book, len * sizeof(Data));
    cout << "TIME\t================================================================================\n\n";
}

template <typename Data>
void report_decompression_time(size_t len, float lossy, float outlier, float lossless)
{
    cout << "\nTIME in milliseconds\t================================================================\n";
    float all = lossy + outlier + lossless;

    printf("TIME\tscatter outlier:\t%f\t", outlier), display_throughput(outlier, len * sizeof(Data));
    printf("TIME\tHuffman decode:\t%f\t", lossless), display_throughput(lossless, len * sizeof(Data));
    printf("TIME\treconstruct:\t%f\t", lossy), display_throughput(lossy, len * sizeof(Data));

    cout << "TIME\t--------------------------------------------------------------------------------\n";

    printf("TIME\tdecompress (sum):\t%f\t", all), display_throughput(all, len * sizeof(Data));

    cout << "TIME\t================================================================================\n\n";
}

unsigned int tune_deflate_chunksize(size_t len)
{
    int current_dev = 0;
    cudaSetDevice(current_dev);
    cudaDeviceProp dev_prop{};
    cudaGetDeviceProperties(&dev_prop, current_dev);

    auto nSM                = dev_prop.multiProcessorCount;
    auto allowed_block_dim  = dev_prop.maxThreadsPerBlock;
    auto deflate_nthread    = allowed_block_dim * nSM / HuffConfig::deflate_constant;
    auto optimal_chunk_size = (len + deflate_nthread - 1) / deflate_nthread;
    optimal_chunk_size      = ((optimal_chunk_size - 1) / HuffConfig::Db_deflate + 1) * HuffConfig::Db_deflate;

    return optimal_chunk_size;
}

void PackMetadata(argpack* ap, metadata_pack* mp, const int nnz)
{
    mp->dim4    = ap->dim4;
    mp->stride4 = ap->stride4;
    mp->nblk4   = ap->nblk4;
    mp->ndim    = ap->ndim;
    mp->eb      = ap->eb;
    mp->len     = ap->len;

    mp->nnz = nnz;

    if (ap->dtype == "f32") mp->dtype = DataType::kF32;
    if (ap->dtype == "f64") mp->dtype = DataType::kF64;

    mp->quant_byte    = ap->quant_byte;
    mp->huff_byte     = ap->huff_byte;
    mp->huffman_chunk = ap->huffman_chunk;
    mp->skip_huffman  = ap->sz_workflow.skip_huffman;
}

void UnpackMetadata(argpack* ap, metadata_pack* mp, int& nnz)
{
    ap->dim4    = mp->dim4;
    ap->stride4 = mp->stride4;
    ap->nblk4   = mp->nblk4;
    ap->ndim    = mp->ndim;
    ap->eb      = mp->eb;
    ap->len     = mp->len;

    nnz = mp->nnz;

    if (mp->dtype == DataType::kF32) ap->dtype = "f32";
    if (mp->dtype == DataType::kF64) ap->dtype = "f64";

    ap->quant_byte               = mp->quant_byte;
    ap->huff_byte                = mp->huff_byte;
    ap->huffman_chunk            = mp->huffman_chunk;
    ap->sz_workflow.skip_huffman = mp->skip_huffman;
}

}  // namespace

#define DATATYPE struct PartialData<typename DataTrait<If_FP, DataByte>::Data>

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_compress(argpack* ap, DATATYPE* in_data, dim3 xyz, metadata_pack* mp, unsigned int optional_w)
{
    using Data  = typename DataTrait<If_FP, DataByte>::Data;
    using Quant = typename QuantTrait<QuantByte>::Quant;
    using Huff  = typename HuffTrait<HuffByte>::Huff;

    size_t len = ap->len;

    int    nnz_outlier = 0;
    size_t num_bits, num_uints, huff_meta_size;

    auto& workflow = ap->sz_workflow;
    auto& subfiles = ap->subfiles;

    auto radius = ap->radius;
    auto eb     = ap->eb;
    auto ebx2   = eb * 2;
    auto ebx2_r = 1 / (eb * 2);

    // dryrun
    if (workflow.dryrun) {
        logging(log_info, "invoke dry-run");
        constexpr auto SEQ       = 4;
        constexpr auto SUBSIZE   = 256;
        auto           dim_block = SUBSIZE / SEQ;
        auto           dim_grid  = get_npart(len, SUBSIZE);

        cusz::dual_quant_dryrun<Data, float, SUBSIZE, SEQ><<<dim_grid, dim_block>>>(in_data->dptr, len, ebx2_r, ebx2);
        HANDLE_ERROR(cudaDeviceSynchronize());

        Data* dryrun_result;
        cudaMallocHost(&dryrun_result, len * sizeof(Data));
        cudaMemcpy(dryrun_result, in_data->dptr, len * sizeof(Data), cudaMemcpyDeviceToHost);

        analysis::verify_data<Data>(&ap->stat, dryrun_result, in_data->hptr, len);
        analysis::print_data_quality_metrics<Data>(&ap->stat, false, ap->eb, 0);

        return;
    }
    logging(log_info, "invoke lossy-construction");

    struct PartialData<Quant> quant(len + HuffConfig::Db_encode);
    cudaMalloc(&quant.dptr, quant.nbyte());

    float time_lossy{0}, time_outlier{0}, time_hist{0}, time_book{0}, time_lossless{0};

    /********************************************************************************
     * constructing quant code
     ********************************************************************************/
    compress_lorenzo_construct<Data, Quant, float>(in_data->dptr, quant.dptr, xyz, ap->ndim, eb, radius, time_lossy);

    /********************************************************************************
     * gather outlier
     ********************************************************************************/
    {
        OutlierHandler<Data> csr(len);

        uint8_t *pool, *dump;
        auto     dummy_nnz    = len / 10;
        auto     pool_bytelen = csr.query_pool_size(dummy_nnz);
        cudaMalloc((void**)&pool, pool_bytelen);

        csr.configure(pool, dummy_nnz).gather_CUDA10(in_data->dptr, time_outlier);

        auto dump_bytelen = csr.query_csr_bytelen();
        cudaMallocHost((void**)&dump, dump_bytelen);

        csr.archive(dump, nnz_outlier);
        io::write_array_to_binary(subfiles.compress.out_outlier, dump, dump_bytelen);

        cudaFree(pool), cudaFreeHost(dump);
    }

    auto fmt_nnz = "(" + std::to_string(nnz_outlier / 1.0 / len * 100) + "%)";
    logging(log_info, "nnz/#outlier:", nnz_outlier, fmt_nnz, "saved");
    cudaFree(in_data->dptr);  // ad-hoc, release memory for large dataset

    /********************************************************************************
     * autotuning Huffman chunksize
     ********************************************************************************/
    if (workflow.autotune_huffchunk) ap->huffman_chunk = tune_deflate_chunksize(len);
    // logging(log_dbg, "Huffman chunk size:", ap->huffman_chunk, "thread num:", (len - 1) / ap->huffman_chunk + 1);

    auto dict_size = ap->dict_size;

    struct PartialData<unsigned int> freq(dict_size);
    cudaMalloc(&freq.dptr, freq.nbyte());

    struct PartialData<Huff> book(dict_size);
    cudaMalloc(&book.dptr, book.nbyte()), book.memset(0xff);

    auto                        revbook_nbyte = get_revbook_nbyte<Quant, Huff>(dict_size);
    struct PartialData<uint8_t> revbook(revbook_nbyte);
    cudaMalloc(&revbook.dptr, revbook.nbyte());

    // histogram, TODO substitute with Analyzer method
    wrapper::get_frequency(quant.dptr, len, freq.dptr, dict_size, time_hist);

    {
        auto t = new cuda_timer_t;
        t->timer_start();
        lossless::par_huffman::par_get_codebook<Quant, Huff>(dict_size, freq.dptr, book.dptr, revbook.dptr);
        time_book = t->timer_end_get_elapsed_time();
        cudaDeviceSynchronize();
        delete t;
    }

    /********************************************************************************
     * analyze compressibility
     ********************************************************************************/
    if (ap->report.compressibility) {
        cudaMallocHost(&freq.hptr, freq.nbyte()), freq.d2h();
        cudaMallocHost(&book.hptr, book.nbyte()), book.d2h();

        Analyzer analyzer{};
        analyzer  //
            .EstimateFromHistogram(freq.hptr, dict_size)
            .template GetHuffmanCodebookStat<Huff>(freq.hptr, book.hptr, len, dict_size)
            .PrintCompressibilityInfo(true);

        cudaFreeHost(freq.hptr);
        cudaFreeHost(book.hptr);
    }

    // internal evaluation, not stored in sz archive
    if (workflow.export_book) {  //
        draft::export_codebook(book.dptr, subfiles.compress.huff_base, dict_size);
        logging(log_info, "exporting codebook as binary; suffix: \".lean-book\"");
    }

    if (workflow.export_quant) {  //
        cudaMallocHost(&quant.hptr, quant.nbyte());
        quant.d2h();

        io::write_array_to_binary(subfiles.compress.raw_quant, quant.hptr, len);
        logging(log_info, "exporting quant as binary; suffix: \".lean-quant\"");
        logging(log_info, "exiting");
        exit(0);
    }

    // decide if skipping Huffman coding
    if (workflow.skip_huffman) {
        cudaMallocHost(&quant.hptr, quant.nbyte());
        quant.d2h();

        io::write_array_to_binary(subfiles.compress.out_quant, quant.hptr, len);
        logging(log_info, "to store quant.code directly (Huffman enc skipped)");
        exit(0);
    }
    // --------------------------------------------------------------------------------

    std::tie(num_bits, num_uints, huff_meta_size) = lossless::interface::HuffmanEncode<Quant, Huff>(
        subfiles.compress.huff_base, quant.dptr, book.dptr, revbook.dptr, revbook_nbyte, len, ap->huffman_chunk,
        ap->dict_size, time_lossless);

    /********************************************************************************
     * report time
     ********************************************************************************/
    if (ap->report.time)
        report_compression_time<Data>(len, time_lossy, time_outlier, time_hist, time_book, time_lossless);

    cudaFree(quant.dptr), cudaFree(freq.dptr), cudaFree(book.dptr), cudaFree(revbook.dptr);

    PackMetadata(ap, mp, nnz_outlier);
    mp->num_bits       = num_bits;
    mp->num_uints      = num_uints;
    mp->huff_meta_size = huff_meta_size;
}

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_decompress(argpack* ap, metadata_pack* mp)
{
    using Data  = typename DataTrait<If_FP, DataByte>::Data;
    using Quant = typename QuantTrait<QuantByte>::Quant;
    using Huff  = typename HuffTrait<HuffByte>::Huff;

    int nnz_outlier = 0;
    UnpackMetadata(ap, mp, nnz_outlier);
    auto num_uints      = mp->num_uints;
    auto huff_meta_size = mp->huff_meta_size;

    auto xyz = dim3(ap->dim4._0, ap->dim4._1, ap->dim4._2);

    float time_lossy{0}, time_outlier{0}, time_lossless{0};

    auto  len      = ap->len;
    auto& workflow = ap->sz_workflow;
    auto& subfiles = ap->subfiles;

    auto m   = static_cast<size_t>(ceil(sqrt(ap->len)));
    auto mxm = m * m;

    auto radius = ap->radius;
    auto eb     = ap->eb;
    auto ebx2   = (eb * 2);

    logging(log_info, "invoke lossy-reconstruction");

    struct PartialData<Quant> quant(len);
    cudaMalloc(&quant.dptr, quant.nbyte());
    cudaMallocHost(&quant.hptr, quant.nbyte());

    struct PartialData<Data> _data(mxm + MetadataTrait<1>::Block);  // TODO ad hoc size
    cudaMalloc(&_data.dptr, _data.nbyte());
    cudaMallocHost(&_data.hptr, _data.nbyte());
    auto xdata   = _data.dptr;
    auto outlier = _data.dptr;

    // step 1: read from filesystem or do Huffman decoding to get quant code
    if (workflow.skip_huffman) {
        logging(log_info, "load quant.code from filesystem");
        io::read_binary_to_array(subfiles.decompress.in_quant, quant.hptr, quant.len);
        quant.h2d();
    }
    else {
        logging(log_info, "Huffman decode -> quant.code");
        lossless::interface::HuffmanDecode<Quant, Huff>(
            subfiles.path2file, &quant, ap->len, ap->huffman_chunk, num_uints, ap->dict_size, time_lossless);
    }

    {
        OutlierHandler<Data> csr(ap->len, nnz_outlier);

        uint8_t *h_csr_file, *d_csr_file;
        cudaMallocHost((void**)&h_csr_file, csr.bytelen.total);
        cudaMalloc((void**)&d_csr_file, csr.bytelen.total);

        io::read_binary_to_array<uint8_t>(subfiles.decompress.in_outlier, h_csr_file, csr.bytelen.total);
        cudaMemcpy(d_csr_file, h_csr_file, csr.bytelen.total, cudaMemcpyHostToDevice);

        csr.extract(d_csr_file).scatter_CUDA10(outlier, time_outlier);
    }

    /********************************************************************************
     * lorenzo reconstruction
     ********************************************************************************/
    decompress_lorenzo_reconstruct(xdata, quant.dptr, xyz, ap->ndim, eb, radius, time_lossy);

    /********************************************************************************
     * report time
     ********************************************************************************/
    if (ap->report.time) report_decompression_time<Data>(len, time_lossy, time_outlier, time_lossless);

    // copy decompressed data to host
    _data.d2h();

    logging(log_info, "reconstruct error-bounded datum");

    size_t archive_bytes = 0;
    // TODO huffman chunking metadata
    if (not workflow.skip_huffman)
        archive_bytes += num_uints * sizeof(Huff)  // Huffman coded
                         + huff_meta_size;         // chunking metadata and reverse codebook
    else
        archive_bytes += ap->len * sizeof(Quant);
    archive_bytes += nnz_outlier * (sizeof(Data) + sizeof(int)) + (m + 1) * sizeof(int);

    if (workflow.skip_huffman) {
        logging(
            log_info, "dtype is \"", demangle(typeid(Data).name()), "\", and quant. code type is \"",
            demangle(typeid(Quant).name()), "\"; a CR of no greater than ", (sizeof(Data) / sizeof(Quant)),
            " is expected when Huffman codec is skipped.");
    }

    if (workflow.pre_binning) logging(log_info, "Because of 2x2->1 binning, extra 4x CR is added.");

    // TODO move CR out of verify_data
    if (not subfiles.decompress.in_origin.empty() and ap->report.quality) {
        logging(log_info, "load the original datum for comparison");

        auto odata = io::read_binary_to_new_array<Data>(subfiles.decompress.in_origin, len);

        analysis::verify_data(&ap->stat, _data.hptr, odata, len);
        analysis::print_data_quality_metrics<Data>(
            &ap->stat, false, ap->eb, archive_bytes, workflow.pre_binning ? 4 : 1, true);
    }
    logging(log_info, "output:", subfiles.path2file + ".szx");

    if (workflow.skip_write2disk)
        logging(log_dbg, "skip writing unzipped to filesystem");
    else {
        io::write_array_to_binary(subfiles.decompress.out_xdata, xdata, ap->len);
    }
}

#define CUSZ_COMPRESS(DBYTE, QBYTE, HBYTE)                  \
    template void cusz_compress<true, DBYTE, QBYTE, HBYTE>( \
        argpack*, struct PartialData<float>*, dim3, metadata_pack*, unsigned int);

CUSZ_COMPRESS(4, 1, 4)
CUSZ_COMPRESS(4, 1, 8)
CUSZ_COMPRESS(4, 2, 4)
CUSZ_COMPRESS(4, 2, 8)

#define CUSZ_DECOMPRESS(DBYTE, QBYTE, HBYTE) \
    template void cusz_decompress<true, DBYTE, QBYTE, HBYTE>(argpack*, metadata_pack*);

CUSZ_DECOMPRESS(4, 1, 4)
CUSZ_DECOMPRESS(4, 1, 8)
CUSZ_DECOMPRESS(4, 2, 4)
CUSZ_DECOMPRESS(4, 2, 8)
