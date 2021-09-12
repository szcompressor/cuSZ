/**
 * @file nvgpusz.cu
 * @author Jiannan Tian
 * @brief Workflow of cuSZ.
 * @version 0.3
 * @date 2021-07-12
 * (create) 2020-02-12; (release) 2020-09-20; (rev.1) 2021-01-16; (rev.2) 2021-07-12; (rev.3) 2021-09-06
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
#include <fstream>
#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "analysis/analyzer.hh"
#include "argparse.hh"
#include "kernel/dryrun.cuh"
#include "kernel/lorenzo.cuh"
#include "metadata.hh"
#include "nvgpusz.cuh"
#include "type_trait.hh"
#include "utils.hh"
#include "wrapper/extrap_lorenzo.cuh"
#include "wrapper/handle_sparsity.cuh"
#include "wrapper/huffman_enc_dec.cuh"
#include "wrapper/huffman_parbook.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

////////////////////////////////////////////////////////////////////////////////

#define COMPR_TYPE template <typename Data, typename Quant, typename Huff, typename FP>
#define COMPRESSOR Compressor<Data, Quant, Huff, FP>

COMPR_TYPE
unsigned int COMPRESSOR::tune_deflate_chunksize(size_t len)
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

COMPR_TYPE
void COMPRESSOR::report_compression_time(size_t len, float lossy, float outlier, float hist, float book, float lossless)
{
    auto display_throughput = [](float milliseconds, size_t nbyte) {
        auto GiB        = 1.0 * 1024 * 1024 * 1024;
        auto seconds    = milliseconds * 1e-3;
        auto throughput = nbyte / GiB / seconds;
        cout << throughput << "GiB/s\n";
    };
    //
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

COMPR_TYPE
COMPRESSOR::Compressor(argpack* _ctx) : ctx(_ctx)
{
    ndim = ctx->ndim;

    config.radius = ctx->radius;

    length.data      = ctx->data_len;
    length.quant     = length.data;  // TODO if lorenzo
    length.dict_size = ctx->dict_size;

    config.eb     = ctx->eb;
    config.ebx2   = ctx->eb * 2;
    config.ebx2_r = 1 / (ctx->eb * 2);
    config.eb_r   = 1 / ctx->eb;

    if (ctx->task_is.autotune_huffchunk) ctx->huffman_chunk = tune_deflate_chunksize(length.data);

    csr = new OutlierHandler<Data>(length.data, &sp.workspace_nbyte);
}

COMPR_TYPE
void COMPRESSOR::lorenzo_dryrun(Capsule<Data>* in_data)
{
    auto get_npart = [](auto size, auto subsize) { return (size + subsize - 1) / subsize; };

    if (ctx->task_is.dryrun) {
        auto len    = length.data;
        auto eb     = config.eb;
        auto ebx2_r = config.ebx2_r;
        auto ebx2   = config.ebx2;

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

        analysis::verify_data<Data>(&ctx->stat, dryrun_result, in_data->hptr, len);
        analysis::print_data_quality_metrics<Data>(&ctx->stat, false, eb, 0);

        cudaFreeHost(dryrun_result);

        exit(0);
    }
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::predict_quantize(Capsule<Data>* data, dim3 xyz, Capsule<Data>* anchor, Capsule<Quant>* quant)
{
    logging(log_info, "compressing...");
    // TODO "predictor" -> "prediction"
    if (ctx->task_is.predictor == "lorenzo") {
        // TODO class lorenzo
        compress_lorenzo_construct<Data, Quant, float>(
            data->dptr, quant->dptr, xyz, ctx->ndim, config.eb, config.radius, time.lossy);
    }
    else if (ctx->task_is.predictor == "spline3d") {
        if (ctx->ndim != 3) throw std::runtime_error("must be 3D data.");
        // TODO timer
        spline3->predict_quantize();
    }
    else {
        throw std::runtime_error("must be \"lorenzo\" or \"spline3d\"");
    }

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::gather_outlier(Capsule<Data>* in_data)
{
    // can be known on Compressor init
    cudaMalloc((void**)&sp.workspace, sp.workspace_nbyte);
    cudaMallocHost((void**)&sp.dump, sp.workspace_nbyte);

    csr->configure(sp.workspace)  //
        .gather_CUDA10(in_data->dptr, sp.dump_nbyte, time.outlier)
        .archive(sp.dump, length.nnz_outlier);

    data_seg.nbyte_raw.at("outlier") = sp.dump_nbyte;

    cudaFree(sp.workspace);

    auto fmt_nnz = "(" + std::to_string(length.nnz_outlier / 1.0 / length.data * 100) + "%)";
    logging(log_info, "#outlier = ", length.nnz_outlier, fmt_nnz);

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::get_freq_and_codebook(
    Capsule<Quant>*        quant,
    Capsule<unsigned int>* freq,
    Capsule<Huff>*         book,
    Capsule<uint8_t>*      revbook)
{
    wrapper::get_frequency<Quant>(quant->dptr, length.quant, freq->dptr, length.dict_size, time.hist);

    {  // This is end-to-end time for parbook.
        auto t = new cuda_timer_t;
        t->timer_start();
        lossless::par_get_codebook<Quant, Huff>(length.dict_size, freq->dptr, book->dptr, revbook->dptr);
        time.book = t->timer_end_get_elapsed_time();
        cudaDeviceSynchronize();
        delete t;
    }

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::analyze_compressibility(
    Capsule<unsigned int>* freq,  //
    Capsule<Huff>*         book)
{
    if (ctx->report.compressibility) {
        cudaMallocHost(&freq->hptr, freq->nbyte()), freq->d2h();
        cudaMallocHost(&book->hptr, book->nbyte()), book->d2h();

        Analyzer analyzer{};
        analyzer  //
            .EstimateFromHistogram(freq->hptr, length.dict_size)
            .template GetHuffmanCodebookStat<Huff>(freq->hptr, book->hptr, length.data, length.dict_size)
            .PrintCompressibilityInfo(true);

        cudaFreeHost(freq->hptr);
        cudaFreeHost(book->hptr);
    }

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::internal_eval_try_export_book(Capsule<Huff>* book)
{
    // internal evaluation, not stored in sz archive
    if (ctx->task_is.export_book) {
        cudaMallocHost(&book->hptr, length.dict_size * sizeof(decltype(book->hptr)));
        book->d2h();

        std::stringstream s;
        s << ctx->fnames.path_basename + "-" << length.dict_size << "-ui" << sizeof(Huff) << ".lean-book";

        // TODO as part of dump
        io::write_array_to_binary(s.str(), book->hptr, length.dict_size);

        cudaFreeHost(book->hptr);
        book->hptr = nullptr;

        logging(log_info, "exporting codebook as binary; suffix: \".lean-book\"");

        data_seg.nbyte_raw.at("book") = length.dict_size * sizeof(Huff);
    }
    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::internal_eval_try_export_quant(Capsule<Quant>* quant)
{
    // internal_eval
    if (ctx->task_is.export_quant) {  //
        cudaMallocHost(&quant->hptr, quant->nbyte());
        quant->d2h();

        data_seg.nbyte_raw.at("quant") = quant->nbyte();

        // TODO as part of dump
        io::write_array_to_binary(ctx->fnames.path_basename + ".lean-quant", quant->hptr, length.quant);
        logging(log_info, "exporting quant as binary; suffix: \".lean-quant\"");
        logging(log_info, "exiting");
        exit(0);
    }
    return *this;
}

COMPR_TYPE
void COMPRESSOR::try_skip_huffman(Capsule<Quant>* quant)
{
    // decide if skipping Huffman coding
    if (ctx->task_is.skip_huffman) {
        cudaMallocHost(&quant->hptr, quant->nbyte());
        quant->d2h();

        // TODO: as part of cusza
        io::write_array_to_binary(ctx->fnames.path_basename + ".quant", quant->hptr, length.quant);
        logging(log_info, "to store quant.code directly (Huffman enc skipped)");
        exit(0);
    }
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::try_report_time()
{
    if (ctx->report.time)
        report_compression_time(length.data, time.lossy, time.outlier, time.hist, time.book, time.lossless);
    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::export_revbook(Capsule<uint8_t>* revbook)
{
    revbook->d2h();
    data_seg.nbyte_raw.at("revbook") = get_revbook_nbyte();

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::huffman_encode(
    Capsule<Quant>* quant,  //
    Capsule<Huff>*  book)
{
    // fix-length space, padding improvised
    cudaMalloc(&huffman.array.d_encspace, sizeof(Huff) * (length.quant + ctx->huffman_chunk + HuffConfig::Db_encode));

    auto nchunk = (length.quant + ctx->huffman_chunk - 1) / ctx->huffman_chunk;
    ctx->nchunk = nchunk;

    // gather metadata (without write) before gathering huff as sp on GPU
    cudaMallocHost(&huffman.array.h_counts, nchunk * 3 * sizeof(size_t));
    cudaMalloc(&huffman.array.d_counts, nchunk * 3 * sizeof(size_t));

    auto dev_bits    = huffman.array.d_counts;
    auto dev_uints   = huffman.array.d_counts + nchunk;
    auto dev_entries = huffman.array.d_counts + nchunk * 2;

    lossless::HuffmanEncode<Quant, Huff, false>(
        huffman.array.d_encspace, dev_bits, dev_uints, dev_entries, huffman.array.h_counts,
        //
        nullptr,
        //
        quant->dptr, book->dptr, length.quant, ctx->huffman_chunk, ctx->dict_size, &huffman.meta.num_bits,
        &huffman.meta.num_uints, time.lossless);

    // --------------------------------------------------------------------------------
    cudaMallocHost(&huffman.array.h_bitstream, huffman.meta.num_uints * sizeof(Huff));
    cudaMalloc(&huffman.array.d_bitstream, huffman.meta.num_uints * sizeof(Huff));

    lossless::HuffmanEncode<Quant, Huff, true>(
        huffman.array.d_encspace, nullptr, dev_uints, dev_entries, nullptr,
        //
        huffman.array.d_bitstream,
        //
        nullptr, nullptr, length.quant, ctx->huffman_chunk, 0, nullptr, nullptr, time.lossless);

    // --------------------------------------------------------------------------------
    cudaMemcpy(
        huffman.array.h_bitstream, huffman.array.d_bitstream, huffman.meta.num_uints * sizeof(Huff),
        cudaMemcpyDeviceToHost);

    // TODO size_t -> MetadataT
    data_seg.nbyte_raw.at("huff-meta")      = sizeof(size_t) * (2 * nchunk);
    data_seg.nbyte_raw.at("huff-bitstream") = sizeof(Huff) * huffman.meta.num_uints;

    cudaFree(huffman.array.d_encspace);

    huffman.meta.revbook_nbyte = get_revbook_nbyte();

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::pack_metadata()
{
    header->x    = ctx->x;
    header->y    = ctx->y;
    header->z    = ctx->z;
    header->w    = ctx->w;
    header->ndim = ctx->ndim;
    header->eb   = ctx->eb;

    header->outlier.nnz        = length.nnz_outlier;
    header->data_len           = ctx->data_len;
    header->config.quant_nbyte = ctx->quant_nbyte;
    header->config.huff_nbyte  = ctx->huff_nbyte;
    header->huffman.chunk      = ctx->huffman_chunk;
    header->skip_huffman       = ctx->task_is.skip_huffman;

    // header->outlier.num_bits  = huffman.meta.num_bits;
    header->huffman.num_uints = huffman.meta.num_uints;

    header->nbyte.revbook = huffman.meta.revbook_nbyte;

    return *this;
}

////////////////////////////////////////////////////////////////////////////////

#define DECOMPR_TYPE template <typename Data, typename Quant, typename Huff, typename FP>
#define DECOMPRESSOR Decompressor<Data, Quant, Huff, FP>

DECOMPR_TYPE
void DECOMPRESSOR::unpack_metadata()
{
    ctx->x    = header->x;
    ctx->y    = header->y;
    ctx->z    = header->z;
    ctx->w    = header->w;
    ctx->ndim = header->ndim;
    ctx->eb   = header->eb;

    ctx->data_len = header->data_len;

    ctx->quant_nbyte          = header->config.quant_nbyte;
    ctx->huff_nbyte           = header->config.huff_nbyte;
    ctx->huffman_chunk        = header->huffman.chunk;
    ctx->task_is.skip_huffman = header->skip_huffman;

    //
    length.nnz_outlier         = header->outlier.nnz;
    huffman.meta.num_uints     = header->huffman.num_uints;
    huffman.meta.revbook_nbyte = header->nbyte.revbook;

    length.data  = ctx->data_len;
    length.quant = length.data;  // TODO if lorenzo

    config.eb     = ctx->eb;
    config.ebx2   = config.eb * 2;
    config.ebx2_r = 1 / (config.eb * 2);
    config.eb_r   = 1 / config.eb;
}

DECOMPR_TYPE
void DECOMPRESSOR::report_decompression_time(size_t len, float lossy, float outlier, float lossless)
{
    auto display_throughput = [](float time, size_t nbyte) {
        auto throughput = nbyte * 1.0 / (1024 * 1024 * 1024) / (time * 1e-3);
        cout << throughput << "GiB/s\n";
    };
    //
    cout << "\nTIME in milliseconds\t================================================================\n";
    float all = lossy + outlier + lossless;

    printf("TIME\tscatter outlier:\t%f\t", outlier), display_throughput(outlier, len * sizeof(Data));
    printf("TIME\tHuffman decode:\t%f\t", lossless), display_throughput(lossless, len * sizeof(Data));
    printf("TIME\treconstruct:\t%f\t", lossy), display_throughput(lossy, len * sizeof(Data));

    cout << "TIME\t--------------------------------------------------------------------------------\n";

    printf("TIME\tdecompress (sum):\t%f\t", all), display_throughput(all, len * sizeof(Data));

    cout << "TIME\t================================================================================\n\n";
}

DECOMPR_TYPE
DECOMPRESSOR::Decompressor(cusz_header* _header, argpack* _ctx) : header(_header), ctx(_ctx)
{
    logging(log_info, "decompressing...");

    unpack_metadata();

    m   = static_cast<size_t>(ceil(sqrt(length.data)));
    mxm = m * m;

    // TODO is ctx still needed?
    xyz = dim3(header->x, header->y, header->z);

    csr     = new OutlierHandler<Data>(length.data, length.nnz_outlier);
    spline3 = new Spline3<Data*, Quant*, float>();
}

DECOMPR_TYPE
DECOMPRESSOR& DECOMPRESSOR::huffman_decode(Capsule<Quant>* quant)
{
    if (ctx->task_is.skip_huffman) {
        // logging(log_info, "load quant.code from filesystem");
        io::read_binary_to_array(ctx->fnames.path_basename + ".quant", quant->hptr, quant->len);
        quant->h2d();
    }
    else {
        // logging(log_info, "Huffman decode -> quant.code");

        auto basename      = ctx->fnames.path2file;
        auto nchunk        = (ctx->data_len - 1) / ctx->huffman_chunk + 1;
        auto num_uints     = header->huffman.num_uints;
        auto revbook_nbyte = data_seg.nbyte_raw.at("revbook");

        auto host_revbook =
            reinterpret_cast<BYTE*>(consolidated_dump.whole + offsets.at(data_seg.name2order.at("revbook")));

        auto host_in_bitstream =
            reinterpret_cast<Huff*>(consolidated_dump.whole + offsets.at(data_seg.name2order.at("huff-bitstream")));

        auto host_bits_entries =
            reinterpret_cast<size_t*>(consolidated_dump.whole + offsets.at(data_seg.name2order.at("huff-meta")));

        auto dev_out_bitstream = mem::create_devspace_memcpy_h2d(host_in_bitstream, num_uints);
        auto dev_bits_entries  = mem::create_devspace_memcpy_h2d(host_bits_entries, 2 * nchunk);
        auto dev_revbook       = mem::create_devspace_memcpy_h2d(host_revbook, revbook_nbyte);

        lossless::HuffmanDecode<Quant, Huff>(
            dev_out_bitstream, dev_bits_entries, dev_revbook,
            //
            quant, ctx->data_len, ctx->huffman_chunk, huffman.meta.num_uints, ctx->dict_size, time.lossless);

        cudaFree(dev_out_bitstream);
        cudaFree(dev_bits_entries);
        cudaFree(dev_revbook);
    }
    return *this;
}

DECOMPR_TYPE
DECOMPRESSOR& DECOMPRESSOR::scatter_outlier(Data* outlier)
{
    csr_file.host = reinterpret_cast<BYTE*>(consolidated_dump.whole + offsets.at(data_seg.name2order.at("outlier")));
    cudaMalloc((void**)&csr_file.dev, csr->bytelen.total);

    cudaMemcpy(csr_file.dev, csr_file.host, csr->bytelen.total, cudaMemcpyHostToDevice);

    csr->extract(csr_file.dev).scatter_CUDA10(outlier, time.outlier);

    cudaFree(csr_file.dev);

    return *this;
}

DECOMPR_TYPE
DECOMPRESSOR& DECOMPRESSOR::reversed_predict_quantize(Data* xdata, dim3 xyz, Data* anchor, Quant* quant)
{
    if (ctx->task_is.predictor == "lorenzo") {
        // TODO lorenzo class
        decompress_lorenzo_reconstruct<Data, Quant, FP>(
            xdata, quant, xyz, ctx->ndim, config.eb, ctx->radius, time.lossy);
    }
    else if (ctx->task_is.predictor == "spline3d") {
        throw std::runtime_error("spline not impl'ed");
        if (ctx->ndim != 3) throw std::runtime_error("Spline3D must be for 3D data.");
        // TODO
        spline3->reversed_predict_quantize();
    }
    else {
        throw std::runtime_error("need to specify predcitor");
    }

    return *this;
}

DECOMPR_TYPE
DECOMPRESSOR& DECOMPRESSOR::calculate_archive_nbyte()
{
    auto demangle = [](const char* name) -> string {
        int   status = -4;
        char* res    = abi::__cxa_demangle(name, nullptr, nullptr, &status);

        const char* const demangled_name = (status == 0) ? res : name;
        string            ret_val(demangled_name);
        free(res);
        return ret_val;
    };

    if (not ctx->task_is.skip_huffman)
        archive_bytes += huffman.meta.num_uints * sizeof(Huff)  // Huffman coded
                         + huffman.meta.revbook_nbyte;          // chunking metadata and reverse codebook
    else
        archive_bytes += length.quant * sizeof(Quant);
    archive_bytes += length.nnz_outlier * (sizeof(Data) + sizeof(int)) + (m + 1) * sizeof(int);

    if (ctx->task_is.skip_huffman) {
        logging(
            log_info, "dtype is \"", demangle(typeid(Data).name()), "\", and quant. code type is \"",
            demangle(typeid(Quant).name()), "\"; a CR of no greater than ", (sizeof(Data) / sizeof(Quant)),
            " is expected when Huffman codec is skipped.");
    }

    if (ctx->task_is.pre_binning) logging(log_info, "Because of 2x2->1 binning, extra 4x CR is added.");

    return *this;
}

DECOMPR_TYPE
DECOMPRESSOR& DECOMPRESSOR::try_report_time()
{
    if (ctx->report.time) report_decompression_time(length.data, time.lossy, time.outlier, time.lossless);

    return *this;
}

DECOMPR_TYPE
DECOMPRESSOR& DECOMPRESSOR::try_compare(Data* xdata)
{
    // TODO move CR out of verify_data
    if (not ctx->fnames.origin_cmp.empty() and ctx->report.quality) {
        logging(log_info, "compare to the original");

        auto odata = io::read_binary_to_new_array<Data>(ctx->fnames.origin_cmp, length.data);

        analysis::verify_data(&ctx->stat, xdata, odata, length.data);
        analysis::print_data_quality_metrics<Data>(
            &ctx->stat, false, ctx->eb, archive_bytes, ctx->task_is.pre_binning ? 4 : 1, true);

        delete[] odata;
    }
    return *this;
}

DECOMPR_TYPE
DECOMPRESSOR& DECOMPRESSOR::try_write2disk(Data* host_xdata)
{
    logging(log_info, "output:", ctx->fnames.path_basename + ".cuszx");

    if (ctx->task_is.skip_write2disk)
        logging(log_dbg, "skip writing unzipped to filesystem");
    else {
        io::write_array_to_binary(ctx->fnames.path_basename + "cuszx", host_xdata, ctx->data_len);
    }

    return *this;
}

////////////////////////////////////////////////////////////////////////////////

#define DATATYPE Capsule<typename DataTrait<If_FP, DataByte>::Data>

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_compress(argpack* ctx, DATATYPE* in_data)
{
    using Data  = typename DataTrait<If_FP, DataByte>::Data;
    using Quant = typename QuantTrait<QuantByte>::Quant;
    using Huff  = typename HuffTrait<HuffByte>::Huff;

    auto xyz = dim3(ctx->x, ctx->y, ctx->z);

    Compressor<Data, Quant, Huff, float> cuszc(ctx);
    cuszc.header = new cusz_header();

    // TODO lorenzo class::get_len_quant
    auto lorenzo_get_len_quant = [&]() -> unsigned int { return ctx->data_len + HuffConfig::Db_encode; };

    unsigned int len_quant = ctx->task_is.predictor == "spline3"  //
                                 ? 1
                                 : lorenzo_get_len_quant();

    cuszc.lorenzo_dryrun(in_data);  // subject to change

    Capsule<Quant> quant(len_quant);
    cudaMalloc(&quant.dptr, quant.nbyte());

    Capsule<Data>* anchor = nullptr;
    if (ctx->task_is.predictor == "spline3") {
        // TODO
    }

    Capsule<unsigned int> freq(ctx->dict_size);
    cudaMalloc(&freq.dptr, freq.nbyte());

    Capsule<Huff> book(ctx->dict_size);
    cudaMalloc(&book.dptr, book.nbyte()), book.memset(0xff);

    Capsule<uint8_t> revbook(cuszc.get_revbook_nbyte());
    cudaMalloc(&revbook.dptr, revbook.nbyte());
    cudaMallocHost(&revbook.hptr, revbook.nbyte());  // to write to disk later

    cuszc.huffman.array.h_revbook = revbook.hptr;

    cuszc  //
        .predict_quantize(in_data, xyz, anchor, &quant)
        .gather_outlier(in_data)
        .try_skip_huffman(&quant);

    // release in_data; subject to change
    cudaFree(in_data->dptr);

    cuszc.get_freq_and_codebook(&quant, &freq, &book, &revbook)
        .analyze_compressibility(&freq, &book)
        .internal_eval_try_export_book(&book)
        .internal_eval_try_export_quant(&quant)
        .export_revbook(&revbook)
        .huffman_encode(&quant, &book)
        .try_report_time()
        .pack_metadata()
        .consolidate();

    cudaFree(quant.dptr), cudaFree(freq.dptr), cudaFree(book.dptr), cudaFree(revbook.dptr);
    cudaFreeHost(revbook.hptr);
    delete cuszc.header;
}

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_decompress(argpack* ctx)
{
    using Data  = typename DataTrait<If_FP, DataByte>::Data;
    using Quant = typename QuantTrait<QuantByte>::Quant;
    using Huff  = typename HuffTrait<HuffByte>::Huff;

    auto __cusz_get_filesize = [](std::string fname) -> size_t {
        std::ifstream in(fname.c_str(), std::ifstream::ate | std::ifstream::binary);
        return in.tellg();
    };

    auto fname_dump = ctx->fnames.path2file + ".cusza";
    auto dump_nbyte = __cusz_get_filesize(fname_dump);
    auto h_dump     = io::read_binary_to_new_array<BYTE>(fname_dump, dump_nbyte);

    // cout << "dump-nbyte by tellg()\t" << dump_nbyte << '\n';
    auto header = reinterpret_cast<cusz_header*>(h_dump);

    // TODO Decompressor encapsulate dump
    // TODO float -> another parameter FP
    Decompressor<Data, Quant, Huff, float> cuszd(header, ctx);

    cuszd.consolidated_dump.whole = h_dump;

    cuszd.read_array_nbyte_from_header();
    cuszd.get_data_seg_offsets();

    // TODO lorenzo class::get_len_quant
    auto lorenzo_get_len_quant = [&]() -> unsigned int { return ctx->data_len; };

    unsigned int len_quant = ctx->task_is.predictor == "spline3"  //
                                 ? cuszd.spline3->get_len_quant()
                                 : lorenzo_get_len_quant();

    Capsule<Data>* anchor =
        new Capsule<Data>(cuszd.spline3->get_len_anchor());  // TODO this .dptr is nullable, error-prone
    if (ctx->task_is.predictor == "spline3") {
        cudaMalloc(&anchor->dptr, anchor->nbyte());
        cudaMallocHost(&anchor->hptr, anchor->nbyte());

        // TODO dummy, need source
    }

    Capsule<Quant> quant(len_quant);
    cudaMalloc(&quant.dptr, quant.nbyte());
    cudaMallocHost(&quant.hptr, quant.nbyte());

    // TODO cuszd.get_len_data_space()
    Capsule<Data> decomp_space(cuszd.mxm + MetadataTrait<1>::Block);  // TODO ad hoc size
    cudaMalloc(&decomp_space.dptr, decomp_space.nbyte());
    cudaMallocHost(&decomp_space.hptr, decomp_space.nbyte());
    auto xdata = decomp_space.dptr, outlier = decomp_space.dptr;

    cuszd.huffman_decode(&quant)
        .scatter_outlier(outlier)
        .reversed_predict_quantize(xdata, cuszd.xyz, anchor->dptr, quant.dptr)
        .try_report_time();

    // copy decompressed data to host
    decomp_space.d2h();

    cuszd
        .calculate_archive_nbyte()  //
        .try_compare(decomp_space.hptr)
        .try_write2disk(decomp_space.hptr);
}

////////////////////////////////////////////////////////////////////////////////

template class Compressor<float, uint8_t, uint32_t, float>;
template class Compressor<float, uint16_t, uint32_t, float>;
template class Compressor<float, uint32_t, uint32_t, float>;
template class Compressor<float, uint8_t, unsigned long long, float>;
template class Compressor<float, uint16_t, unsigned long long, float>;
template class Compressor<float, uint32_t, unsigned long long, float>;

template class Decompressor<float, uint8_t, uint32_t, float>;
template class Decompressor<float, uint16_t, uint32_t, float>;
template class Decompressor<float, uint32_t, uint32_t, float>;
template class Decompressor<float, uint8_t, unsigned long long, float>;
template class Decompressor<float, uint16_t, unsigned long long, float>;
template class Decompressor<float, uint32_t, unsigned long long, float>;

#define CUSZ_COMPRESS(DBYTE, QBYTE, HBYTE) \
    template void cusz_compress<true, DBYTE, QBYTE, HBYTE>(argpack*, Capsule<float>*);

CUSZ_COMPRESS(4, 1, 4)
CUSZ_COMPRESS(4, 1, 8)
CUSZ_COMPRESS(4, 2, 4)
CUSZ_COMPRESS(4, 2, 8)

#define CUSZ_DECOMPRESS(DBYTE, QBYTE, HBYTE) template void cusz_decompress<true, DBYTE, QBYTE, HBYTE>(argpack*);

CUSZ_DECOMPRESS(4, 1, 4)
CUSZ_DECOMPRESS(4, 1, 8)
CUSZ_DECOMPRESS(4, 2, 4)
CUSZ_DECOMPRESS(4, 2, 8)
