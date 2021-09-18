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

#include <bitset>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "analysis/analyzer.hh"
#include "context.hh"
#include "kernel/dryrun.cuh"
#include "kernel/lorenzo.cuh"
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

#define COMPR_TYPE template <typename T, typename E, typename H, typename FP>
#define COMPRESSOR Compressor<T, E, H, FP>

COMPR_TYPE
unsigned int COMPRESSOR::tune_deflate_chunksize(size_t len)
{
    int current_dev = 0;
    cudaSetDevice(current_dev);
    cudaDeviceProp dev_prop{};
    cudaGetDeviceProperties(&dev_prop, current_dev);

    auto nSM                = dev_prop.multiProcessorCount;
    auto allowed_block_dim  = dev_prop.maxThreadsPerBlock;
    auto deflate_nthread    = allowed_block_dim * nSM / HuffmanHelper::DEFLATE_CONSTANT;
    auto optimal_chunk_size = ConfigHelper::get_npart(len, deflate_nthread);
    optimal_chunk_size      = ConfigHelper::get_npart(optimal_chunk_size, HuffmanHelper::BLOCK_DIM_DEFLATE) *
                         HuffmanHelper::BLOCK_DIM_DEFLATE;

    return optimal_chunk_size;
}

COMPR_TYPE
void COMPRESSOR::report_compression_time()
{
    auto  nbyte   = ctx->data_len * sizeof(T);
    float nonbook = time.lossy + time.outlier + time.hist + time.lossless;

    ReportHelper::print_throughput_tablehead("compression");

    ReportHelper::print_throughput_line("construct", time.lossy, nbyte);
    ReportHelper::print_throughput_line("gather-outlier", time.outlier, nbyte);
    ReportHelper::print_throughput_line("histogram", time.hist, nbyte);
    ReportHelper::print_throughput_line("Huff-encode", time.lossless, nbyte);
    ReportHelper::print_throughput_line("(subtotal)", nonbook, nbyte);
    printf("\e[2m");
    ReportHelper::print_throughput_line("book", time.book, nbyte);
    ReportHelper::print_throughput_line("(total)", nonbook + time.book, nbyte);
    printf("\e[0m");
}

COMPR_TYPE
COMPRESSOR::Compressor(cuszCTX* _ctx) : ctx(_ctx)
{
    header = new cusz_header();

    ctx->quant_len = ctx->data_len;  // TODO if lorenzo

    ConfigHelper::set_eb_series(ctx->eb, config);

    if (ctx->task_is.autotune_huffchunk) ctx->huffman_chunk = tune_deflate_chunksize(ctx->data_len);

    csr = new OutlierHandler<T>(ctx->data_len, &sp.workspace_nbyte);
    // can be known on Compressor init
    cudaMalloc((void**)&sp.workspace, sp.workspace_nbyte);
    cudaMallocHost((void**)&sp.dump, sp.workspace_nbyte);

    xyz_v2 = dim3(ctx->x, ctx->y, ctx->z);

    // TODO encapsulation
    auto lorenzo_get_len_quant = [&]() -> unsigned int { return ctx->data_len + HuffmanHelper::BLOCK_DIM_ENCODE; };

    ctx->quant_len = ctx->task_is.predictor == "spline3"  //
                         ? 1
                         : lorenzo_get_len_quant();
}

COMPR_TYPE
void COMPRESSOR::lorenzo_dryrun(Capsule<T>* in_data)
{
    if (ctx->task_is.dryrun) {
        auto len = ctx->data_len;

        LOGGING(LOG_INFO, "invoke dry-run");
        constexpr auto SEQ       = 4;
        constexpr auto SUBSIZE   = 256;
        auto           dim_block = SUBSIZE / SEQ;
        auto           dim_grid  = ConfigHelper::get_npart(len, SUBSIZE);

        cusz::dual_quant_dryrun<T, float, SUBSIZE, SEQ>
            <<<dim_grid, dim_block>>>(in_data->dptr, len, config.ebx2_r, config.ebx2);
        HANDLE_ERROR(cudaDeviceSynchronize());

        T* dryrun_result;
        cudaMallocHost(&dryrun_result, len * sizeof(T));
        cudaMemcpy(dryrun_result, in_data->dptr, len * sizeof(T), cudaMemcpyDeviceToHost);

        analysis::verify_data<T>(&ctx->stat, dryrun_result, in_data->hptr, len);
        analysis::print_data_quality_metrics<T>(&ctx->stat, 0, false);

        cudaFreeHost(dryrun_result);

        exit(0);
    }
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::predict_quantize(Capsule<T>* data, dim3 xyz, Capsule<T>* anchor, Capsule<E>* quant)
{
    LOGGING(LOG_INFO, "compressing...");
    // TODO "predictor" -> "prediction"
    if (ctx->task_is.predictor == "lorenzo") {
        // TODO class lorenzo
        compress_lorenzo_construct<T, E, float>(
            data->dptr, quant->dptr, xyz, ctx->ndim, ctx->eb, ctx->radius, time.lossy);
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
COMPRESSOR& COMPRESSOR::gather_outlier(Capsule<T>* in_data)
{
    csr->gather(in_data->dptr, sp.workspace, sp.dump, sp.dump_nbyte, ctx->nnz_outlier);

    time.outlier = csr->get_milliseconds();

    data_seg.nbyte.at("outlier") = sp.dump_nbyte;

    cudaFree(sp.workspace);

    auto fmt_nnz = "(" + std::to_string(ctx->nnz_outlier / 1.0 / ctx->data_len * 100) + "%)";
    LOGGING(LOG_INFO, "#outlier = ", ctx->nnz_outlier, fmt_nnz);

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::get_freq_and_codebook(
    Capsule<E>*        quant,
    Capsule<uint32_t>* freq,
    Capsule<H>*        book,
    Capsule<uint8_t>*  revbook)
{
    wrapper::get_frequency<E>(quant->dptr, ctx->quant_len, freq->dptr, ctx->dict_size, time.hist);

    {  // This is end-to-end time for parbook.
        auto t = new cuda_timer_t;
        t->timer_start();
        lossless::par_get_codebook<E, H>(ctx->dict_size, freq->dptr, book->dptr, revbook->dptr);
        time.book = t->timer_end_get_elapsed_time();
        cudaDeviceSynchronize();
        delete t;
    }

    this->analyze_compressibility(freq, book)  //
        .internal_eval_try_export_book(book)
        .internal_eval_try_export_quant(quant);

    revbook->d2h();  // need processing on CPU
    data_seg.nbyte.at("revbook") = HuffmanHelper::get_revbook_nbyte<E, H>(ctx->dict_size);

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::analyze_compressibility(
    Capsule<unsigned int>* freq,  //
    Capsule<H>*            book)
{
    if (ctx->report.compressibility) {
        cudaMallocHost(&freq->hptr, freq->nbyte()), freq->d2h();
        cudaMallocHost(&book->hptr, book->nbyte()), book->d2h();

        Analyzer analyzer{};
        analyzer  //
            .EstimateFromHistogram(freq->hptr, ctx->dict_size)
            .template GetHuffmanCodebookStat<H>(freq->hptr, book->hptr, ctx->data_len, ctx->dict_size)
            .PrintCompressibilityInfo(true);

        cudaFreeHost(freq->hptr);
        cudaFreeHost(book->hptr);
    }

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::internal_eval_try_export_book(Capsule<H>* book)
{
    // internal evaluation, not stored in sz archive
    if (ctx->task_is.export_book) {
        cudaMallocHost(&book->hptr, ctx->dict_size * sizeof(decltype(book->hptr)));
        book->d2h();

        std::stringstream s;
        s << ctx->fnames.path_basename + "-" << ctx->dict_size << "-ui" << sizeof(H) << ".lean-book";

        // TODO as part of dump
        io::write_array_to_binary(s.str(), book->hptr, ctx->dict_size);

        cudaFreeHost(book->hptr);
        book->hptr = nullptr;

        LOGGING(LOG_INFO, "exporting codebook as binary; suffix: \".lean-book\"");

        data_seg.nbyte.at("book") = ctx->dict_size * sizeof(H);
    }
    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::internal_eval_try_export_quant(Capsule<E>* quant)
{
    // internal_eval
    if (ctx->task_is.export_quant) {  //
        cudaMallocHost(&quant->hptr, quant->nbyte());
        quant->d2h();

        data_seg.nbyte.at("quant") = quant->nbyte();

        // TODO as part of dump
        io::write_array_to_binary(ctx->fnames.path_basename + ".lean-quant", quant->hptr, ctx->quant_len);
        LOGGING(LOG_INFO, "exporting quant as binary; suffix: \".lean-quant\"");
        LOGGING(LOG_INFO, "exiting");
        exit(0);
    }
    return *this;
}

COMPR_TYPE
void COMPRESSOR::try_skip_huffman(Capsule<E>* quant)
{
    // decide if skipping Huffman coding
    if (ctx->task_is.skip_huffman) {
        cudaMallocHost(&quant->hptr, quant->nbyte());
        quant->d2h();

        // TODO: as part of cusza
        io::write_array_to_binary(ctx->fnames.path_basename + ".quant", quant->hptr, ctx->quant_len);
        LOGGING(LOG_INFO, "to store quant.code directly (Huffman enc skipped)");
        exit(0);
    }
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::try_report_time()
{
    if (ctx->report.time) report_compression_time();
    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::huffman_encode(
    Capsule<E>* quant,  //
    Capsule<H>* book)
{
    // fix-length space, padding improvised
    cudaMalloc(
        &huffman.d_encspace, sizeof(H) * (ctx->quant_len + ctx->huffman_chunk + HuffmanHelper::BLOCK_DIM_ENCODE));

    auto nchunk = ConfigHelper::get_npart(ctx->quant_len, ctx->huffman_chunk);
    ctx->nchunk = nchunk;

    // gather metadata (without write) before gathering huff as sp on GPU
    cudaMallocHost(&huffman.h_counts, nchunk * 3 * sizeof(size_t));
    cudaMalloc(&huffman.d_counts, nchunk * 3 * sizeof(size_t));

    auto dev_bits    = huffman.d_counts;
    auto dev_uints   = huffman.d_counts + nchunk;
    auto dev_entries = huffman.d_counts + nchunk * 2;

    lossless::HuffmanEncode<E, H, false>(
        huffman.d_encspace, dev_bits, dev_uints, dev_entries, huffman.h_counts,
        //
        nullptr,
        //
        quant->dptr, book->dptr, ctx->quant_len, ctx->huffman_chunk, ctx->dict_size, &ctx->huffman_num_bits,
        &ctx->huffman_num_uints, time.lossless);

    // --------------------------------------------------------------------------------
    cudaMallocHost(&huffman.h_bitstream, ctx->huffman_num_uints * sizeof(H));
    cudaMalloc(&huffman.d_bitstream, ctx->huffman_num_uints * sizeof(H));

    lossless::HuffmanEncode<E, H, true>(
        huffman.d_encspace, nullptr, dev_uints, dev_entries, nullptr,
        //
        huffman.d_bitstream,
        //
        nullptr, nullptr, ctx->quant_len, ctx->huffman_chunk, 0, nullptr, nullptr, time.lossless);

    // --------------------------------------------------------------------------------
    cudaMemcpy(huffman.h_bitstream, huffman.d_bitstream, ctx->huffman_num_uints * sizeof(H), cudaMemcpyDeviceToHost);

    // TODO size_t -> MetadataT
    data_seg.nbyte.at("huff-meta")      = sizeof(size_t) * (2 * nchunk);
    data_seg.nbyte.at("huff-bitstream") = sizeof(H) * ctx->huffman_num_uints;

    cudaFree(huffman.d_encspace);

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::pack_metadata()
{
    ConfigHelper::deep_copy_config_items(/* dst */ header, /* src */ ctx);
    return *this;
}

COMPR_TYPE
void COMPRESSOR::consolidate(bool on_cpu, bool on_gpu)
{
    // put in header
    header->nbyte.book           = data_seg.nbyte.at("book");
    header->nbyte.revbook        = data_seg.nbyte.at("revbook");
    header->nbyte.outlier        = data_seg.nbyte.at("outlier");
    header->nbyte.huff_meta      = data_seg.nbyte.at("huff-meta");
    header->nbyte.huff_bitstream = data_seg.nbyte.at("huff-bitstream");

    // consolidate
    std::vector<uint32_t> offsets = {0};

    ReportHelper::print_datasegment_tablehead();

    // print long numbers with thousand separator
    // https://stackoverflow.com/a/7455282
    // https://stackoverflow.com/a/11695246
    setlocale(LC_ALL, "");

    for (auto i = 0; i < 7; i++) {
        const auto& name = data_seg.order2name.at(i);

        auto o = offsets.back() + __cusz_get_alignable_len<BYTE, 128>(data_seg.nbyte.at(name));
        offsets.push_back(o);

        printf("  %-18s\t%'12u\t%'15u\t%'15u\n", name.c_str(), data_seg.nbyte.at(name), offsets.at(i), offsets.back());
    }

    auto total_nbyte = offsets.back();

    printf("\ncompression ratio:\t%.4f\n", ctx->data_len * sizeof(T) * 1.0 / total_nbyte);

    BYTE* h_dump = nullptr;
    // BYTE* d_dump = nullptr;

    cout << "dump on CPU\t" << on_cpu << '\n';
    cout << "dump on GPU\t" << on_gpu << '\n';

    auto both = on_cpu and on_gpu;
    if (both) {
        //
        throw runtime_error("[consolidate on both] not implemented");
    }
    else {
        if (on_cpu) {
            //
            cudaMallocHost(&h_dump, total_nbyte);

            /* 0 */  // header
            cudaMemcpy(
                h_dump + offsets.at(0),           //
                reinterpret_cast<BYTE*>(header),  //
                data_seg.nbyte.at("header"),      //
                cudaMemcpyHostToHost);
            /* 1 */  // book
            /* 2 */  // quant
            /* 3 */  // revbook
            cudaMemcpy(
                h_dump + offsets.at(3),                      //
                reinterpret_cast<BYTE*>(huffman.h_revbook),  //
                data_seg.nbyte.at("revbook"),                //
                cudaMemcpyHostToHost);
            /* 4 */  // outlier
            cudaMemcpy(
                h_dump + offsets.at(4),            //
                reinterpret_cast<BYTE*>(sp.dump),  //
                data_seg.nbyte.at("outlier"),      //
                cudaMemcpyHostToHost);
            /* 5 */  // huff_meta
            cudaMemcpy(
                h_dump + offsets.at(5),                                   //
                reinterpret_cast<BYTE*>(huffman.h_counts + ctx->nchunk),  //
                data_seg.nbyte.at("huff-meta"),                           //
                cudaMemcpyHostToHost);
            /* 6 */  // huff_bitstream
            cudaMemcpy(
                h_dump + offsets.at(6),                        //
                reinterpret_cast<BYTE*>(huffman.h_bitstream),  //
                data_seg.nbyte.at("huff-bitstream"),           //
                cudaMemcpyHostToHost);

            auto output_name = ctx->fnames.path_basename + ".cusza";
            cout << "output:\t" << output_name << '\n';

            io::write_array_to_binary(output_name, h_dump, total_nbyte);

            cudaFreeHost(h_dump);
        }
        else {
            throw runtime_error("[consolidate on both] not implemented");
        }
    }
}

COMPR_TYPE
void COMPRESSOR::compress(Capsule<T>* in_data)
{
    lorenzo_dryrun(in_data);  // subject to change

    Capsule<E>            quant(ctx->quant_len);
    Capsule<unsigned int> freq(ctx->dict_size);
    Capsule<H>            book(ctx->dict_size);
    Capsule<uint8_t>      revbook(HuffmanHelper::get_revbook_nbyte<E, H>(ctx->dict_size));
    cudaMalloc(&quant.dptr, quant.nbyte());
    cudaMalloc(&freq.dptr, freq.nbyte());
    cudaMalloc(&book.dptr, book.nbyte()), book.memset(0xff);
    cudaMalloc(&revbook.dptr, revbook.nbyte());
    cudaMallocHost(&revbook.hptr, revbook.nbyte());  // to write to disk later

    huffman.h_revbook = revbook.hptr;

    this->predict_quantize(in_data, xyz_v2, nullptr, &quant)  //
        .gather_outlier(in_data)
        .try_skip_huffman(&quant);

    // release in_data; subject to change
    cudaFree(in_data->dptr);

    this->get_freq_and_codebook(&quant, &freq, &book, &revbook)
        .huffman_encode(&quant, &book)
        .try_report_time()
        .pack_metadata()
        .consolidate();

    cudaFree(quant.dptr), cudaFree(freq.dptr), cudaFree(book.dptr), cudaFree(revbook.dptr);
    cudaFreeHost(revbook.hptr);
    delete header;
}

////////////////////////////////////////////////////////////////////////////////

#define DECOMPR_TYPE template <typename T, typename E, typename H, typename FP>
#define DECOMPRESSOR Decompressor<T, E, H, FP>

DECOMPR_TYPE
void DECOMPRESSOR::try_report_decompression_time()
{
    if (not ctx->report.time) return;

    auto  nbyte = ctx->data_len * sizeof(T);
    float all   = time.lossy + time.outlier + time.lossless;

    ReportHelper::print_throughput_tablehead("decompression");

    ReportHelper::print_throughput_line("scatter-outlier", time.outlier, nbyte);
    ReportHelper::print_throughput_line("Huff-decode", time.lossless, nbyte);
    ReportHelper::print_throughput_line("reconstruct", time.lossy, nbyte);
    ReportHelper::print_throughput_line("(total)", all, nbyte);

    printf("\n");
}

DECOMPR_TYPE
void DECOMPRESSOR::unpack_metadata()
{
    data_seg.nbyte.at("book")           = header->nbyte.book;
    data_seg.nbyte.at("revbook")        = header->nbyte.revbook;
    data_seg.nbyte.at("outlier")        = header->nbyte.outlier;
    data_seg.nbyte.at("huff-meta")      = header->nbyte.huff_meta;
    data_seg.nbyte.at("huff-bitstream") = header->nbyte.huff_bitstream;

    /* 0 header */ offsets.push_back(0);

    if (ctx->verbose) { printf("(verification)"), ReportHelper::print_datasegment_tablehead(), setlocale(LC_ALL, ""); }

    for (auto i = 0; i < 7; i++) {
        const auto& name  = data_seg.order2name.at(i);
        auto        nbyte = data_seg.nbyte.at(name);
        auto        o     = offsets.back() + __cusz_get_alignable_len<BYTE, 128>(nbyte);
        offsets.push_back(o);

        if (ctx->verbose) {
            printf(
                "  %-18s\t%'12u\t%'15u\t%'15u\n", name.c_str(), data_seg.nbyte.at(name), offsets.at(i), offsets.back());
        }
    }
    if (ctx->verbose) printf("\n");

    ConfigHelper::deep_copy_config_items(/* dst */ ctx, /* src */ header);
    ConfigHelper::set_eb_series(ctx->eb, config);
}

DECOMPR_TYPE
DECOMPRESSOR::Decompressor(cuszCTX* _ctx) : ctx(_ctx)
{
    auto fname_dump   = ctx->fnames.path2file + ".cusza";
    cusza_nbyte       = ConfigHelper::get_filesize(fname_dump);
    consolidated_dump = io::read_binary_to_new_array<BYTE>(fname_dump, cusza_nbyte);
    header            = reinterpret_cast<cusz_header*>(consolidated_dump);

    unpack_metadata();

    m   = Reinterpret1DTo2D::get_square_size(ctx->data_len);
    mxm = m * m;

    // TODO is ctx still needed?
    xyz = dim3(header->x, header->y, header->z);

    csr           = new OutlierHandler<T>(ctx->data_len, ctx->nnz_outlier);
    csr_file.host = reinterpret_cast<BYTE*>(consolidated_dump + offsets.at(data_seg.name2order.at("outlier")));
    cudaMalloc((void**)&csr_file.dev, csr->get_total_nbyte());
    cudaMemcpy(csr_file.dev, csr_file.host, csr->get_total_nbyte(), cudaMemcpyHostToDevice);

    spline3 = new Spline3<T*, E*, float>();

    LOGGING(LOG_INFO, "decompressing...");
}

DECOMPR_TYPE
void DECOMPRESSOR::huffman_decode(Capsule<E>* quant)
{
    if (ctx->task_is.skip_huffman) {  //
        throw std::runtime_error("not implemented when huffman is skipped");
    }
    else {
        auto nchunk        = ConfigHelper::get_npart(ctx->data_len, ctx->huffman_chunk);
        auto num_uints     = header->huffman_num_uints;
        auto revbook_nbyte = data_seg.nbyte.at("revbook");

        auto host_revbook = reinterpret_cast<BYTE*>(consolidated_dump + offsets.at(data_seg.name2order.at("revbook")));

        auto host_in_bitstream =
            reinterpret_cast<H*>(consolidated_dump + offsets.at(data_seg.name2order.at("huff-bitstream")));

        auto host_bits_entries =
            reinterpret_cast<size_t*>(consolidated_dump + offsets.at(data_seg.name2order.at("huff-meta")));

        auto dev_out_bitstream = mem::create_devspace_memcpy_h2d(host_in_bitstream, num_uints);
        auto dev_bits_entries  = mem::create_devspace_memcpy_h2d(host_bits_entries, 2 * nchunk);
        auto dev_revbook       = mem::create_devspace_memcpy_h2d(host_revbook, revbook_nbyte);

        lossless::HuffmanDecode<E, H>(
            dev_out_bitstream, dev_bits_entries, dev_revbook,
            //
            quant, ctx->data_len, ctx->huffman_chunk, ctx->huffman_num_uints, ctx->dict_size, time.lossless);

        cudaFree(dev_out_bitstream);
        cudaFree(dev_bits_entries);
        cudaFree(dev_revbook);
    }
}

DECOMPR_TYPE
void DECOMPRESSOR::reversed_predict_quantize(T* xdata, dim3 xyz, T* anchor, E* quant)
{
    if (ctx->task_is.predictor == "lorenzo") {
        // TODO lorenzo class
        decompress_lorenzo_reconstruct<T, E, FP>(xdata, quant, xyz, ctx->ndim, config.eb, ctx->radius, time.lossy);
    }
    else if (ctx->task_is.predictor == "spline3d") {
        throw std::runtime_error("spline not impl'ed");
        if (ctx->ndim != 3) throw std::runtime_error("Spline3D must be for 3D data.");
        // TODO
        spline3->reversed_predict_quantize();
    }
    else {
        throw std::runtime_error("need to specify predictor");
    }
}

DECOMPR_TYPE
void DECOMPRESSOR::try_compare_with_origin(T* xdata)
{
    // TODO move CR out of verify_data
    if (not ctx->fnames.origin_cmp.empty() and ctx->report.quality) {
        LOGGING(LOG_INFO, "compare to the original");

        auto odata = io::read_binary_to_new_array<T>(ctx->fnames.origin_cmp, ctx->data_len);

        analysis::verify_data(&ctx->stat, xdata, odata, ctx->data_len);
        analysis::print_data_quality_metrics<T>(&ctx->stat, cusza_nbyte, false);

        delete[] odata;
    }
}

DECOMPR_TYPE
void DECOMPRESSOR::try_write2disk(T* host_xdata)
{
    if (ctx->task_is.skip_write2disk)
        LOGGING(LOG_INFO, "output: skipped");
    else {
        LOGGING(LOG_INFO, "output:", ctx->fnames.path_basename + ".cuszx");
        io::write_array_to_binary(ctx->fnames.path_basename + ".cuszx", host_xdata, ctx->data_len);
    }
}

DECOMPR_TYPE
void DECOMPRESSOR::decompress()
{
    // TODO lorenzo class::get_len_quant
    auto lorenzo_get_len_quant = [&]() -> unsigned int { return ctx->data_len; };

    ctx->quant_len = ctx->task_is.predictor == "spline3"  //
                         ? spline3->get_len_quant()
                         : lorenzo_get_len_quant();

    Capsule<E> quant(ctx->quant_len);
    cudaMalloc(&quant.dptr, quant.nbyte());
    cudaMallocHost(&quant.hptr, quant.nbyte());

    // TODO cuszd.get_len_data_space()
    Capsule<T> decomp_space(mxm + MetadataTrait<1>::Block);  // TODO ad hoc size
    cudaMalloc(&decomp_space.dptr, decomp_space.nbyte());
    cudaMallocHost(&decomp_space.hptr, decomp_space.nbyte());
    auto xdata = decomp_space.dptr, outlier = decomp_space.dptr;

    huffman_decode(&quant);

    csr->scatter(csr_file.dev, outlier);
    time.outlier = csr->get_milliseconds();

    reversed_predict_quantize(xdata, xyz, nullptr, quant.dptr);

    try_report_decompression_time();

    // copy decompressed data to host
    decomp_space.d2h();

    try_compare_with_origin(decomp_space.hptr);
    try_write2disk(decomp_space.hptr);
}

////////////////////////////////////////////////////////////////////////////////

// template class Compressor<float, uint8_t, uint32_t, float>;
template class Compressor<float, uint16_t, uint32_t, float>;
// template class Compressor<float, uint32_t, uint32_t, float>;
// template class Compressor<float, uint8_t, unsigned long long, float>;
template class Compressor<float, uint16_t, unsigned long long, float>;
// template class Compressor<float, uint32_t, unsigned long long, float>;

// template class Decompressor<float, uint8_t, uint32_t, float>;
template class Decompressor<float, uint16_t, uint32_t, float>;
// template class Decompressor<float, uint32_t, uint32_t, float>;
// template class Decompressor<float, uint8_t, unsigned long long, float>;
template class Decompressor<float, uint16_t, unsigned long long, float>;
// template class Decompressor<float, uint32_t, unsigned long long, float>;
