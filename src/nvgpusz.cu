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
#include "common/type_traits.hh"
#include "context.hh"
#include "kernel/dryrun.cuh"
#include "kernel/lorenzo.cuh"
#include "nvgpusz.cuh"
#include "utils.hh"
#include "wrapper/extrap_lorenzo.cuh"
#include "wrapper/huffman_enc_dec.cuh"
#include "wrapper/huffman_parbook.cuh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

#define COMPR_TYPE template <typename T, typename E, typename H, typename FP>
#define COMPRESSOR Compressor<T, E, H, FP>

//// constructor
////////////////////////////////////////////////////////////////////////////////

COMPR_TYPE
COMPRESSOR::Compressor(cuszCTX* _ctx, cusz::WHEN _timing) : ctx(_ctx)
{
    timing = _timing;

    if (timing == cusz::WHEN::COMPRESS or    //
        timing == cusz::WHEN::EXPERIMENT or  //
        timing == cusz::WHEN::COMPRESS_DRYRUN) {
        header = new cusz_header();

        ConfigHelper::set_eb_series(ctx->eb, config);

        if (ctx->on_off.autotune_huffchunk) ctx->huffman_chunk = tune_deflate_chunksize(ctx->data_len);

        csr = new cusz::OutlierHandler10<T>(ctx->data_len, &sp.workspace_nbyte);
        // can be known on Compressor init
        cudaMalloc((void**)&sp.workspace, sp.workspace_nbyte);
        cudaMallocHost((void**)&sp.dump, sp.workspace_nbyte);

        xyz = dim3(ctx->x, ctx->y, ctx->z);

        predictor = new cusz::PredictorLorenzo<T, E, FP>(xyz, ctx->eb, ctx->radius, false);

        ctx->quant_len  = predictor->get_quant_len();
        ctx->anchor_len = predictor->get_anchor_len();

        LOGGING(LOG_INFO, "compressing...");
    }
    else if (timing == cusz::WHEN::DECOMPRESS) {
        auto fname_dump = ctx->fnames.path2file + ".cusza";
        cusza_nbyte     = ConfigHelper::get_filesize(fname_dump);
        dump            = io::read_binary_to_new_array<BYTE>(fname_dump, cusza_nbyte);
        header          = reinterpret_cast<cusz_header*>(dump);

        unpack_metadata();

        m   = Reinterpret1DTo2D::get_square_size(ctx->data_len);
        mxm = m * m;

        xyz = dim3(header->x, header->y, header->z);

        csr           = new cusz::OutlierHandler10<T>(ctx->data_len, ctx->nnz_outlier);
        csr_file.host = reinterpret_cast<BYTE*>(dump + dataseg.get_offset("outlier"));
        cudaMalloc((void**)&csr_file.dev, csr->get_total_nbyte());
        cudaMemcpy(csr_file.dev, csr_file.host, csr->get_total_nbyte(), cudaMemcpyHostToDevice);

        predictor = new cusz::PredictorLorenzo<T, E, FP>(xyz, ctx->eb, ctx->radius, false);

        reducer = new cusz::HuffmanWork<E, H>(
            header->quant_len, dump,  //
            header->huffman_chunk, header->huffman_num_uints, header->dict_size);

        LOGGING(LOG_INFO, "decompressing...");
    }
}

COMPR_TYPE
COMPRESSOR::~Compressor()
{
    if (timing == cusz::WHEN::COMPRESS) {  // release small-size arrays
        cudaFree(huffman.d_counts);
        cudaFree(huffman.d_bitstream);

        cudaFreeHost(huffman.h_bitstream);
        cudaFreeHost(huffman.h_counts);

        cudaFree(sp.workspace);
        cudaFreeHost(sp.dump);
    }
    else {
        cudaFree(csr_file.dev);
    }

    delete csr;
    delete predictor;
}

////////////////////////////////////////////////////////////////////////////////

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
COMPRESSOR& COMPRESSOR::get_freq_and_codebook(
    Capsule<E>*          quant,
    Capsule<cusz::FREQ>* freq,
    Capsule<H>*          book,
    Capsule<uint8_t>*    revbook)
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
    dataseg.nbyte.at("revbook") = HuffmanHelper::get_revbook_nbyte<E, H>(ctx->dict_size);

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
            .estimate_compressibility_from_histogram(freq->hptr, ctx->dict_size)
            .template get_stat_from_huffman_book<H>(freq->hptr, book->hptr, ctx->data_len, ctx->dict_size)
            .print_compressibility(true);

        cudaFreeHost(freq->hptr);
        cudaFreeHost(book->hptr);
    }

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::internal_eval_try_export_book(Capsule<H>* book)
{
    // internal evaluation, not stored in sz archive
    if (ctx->export_raw.book) {
        cudaMallocHost(&book->hptr, ctx->dict_size * sizeof(decltype(book->hptr)));
        book->d2h();

        std::stringstream s;
        s << ctx->fnames.path_basename + "-" << ctx->dict_size << "-ui" << sizeof(H) << ".lean-book";

        // TODO as part of dump
        io::write_array_to_binary(s.str(), book->hptr, ctx->dict_size);

        cudaFreeHost(book->hptr);
        book->hptr = nullptr;

        LOGGING(LOG_INFO, "exporting codebook as binary; suffix: \".lean-book\"");

        dataseg.nbyte.at("book") = ctx->dict_size * sizeof(H);
    }
    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::internal_eval_try_export_quant(Capsule<E>* quant)
{
    // internal_eval
    if (ctx->export_raw.quant) {  //
        cudaMallocHost(&quant->hptr, quant->nbyte());
        quant->d2h();

        dataseg.nbyte.at("quant") = quant->nbyte();

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
    if (ctx->to_skip.huffman) {
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

    H*    tmp_space;
    auto  in_len     = ctx->quant_len;
    auto  chunk_size = ctx->huffman_chunk;
    auto  dict_size  = ctx->dict_size;
    auto& num_bits   = ctx->huffman_num_bits;
    auto& num_uints  = ctx->huffman_num_uints;

    ctx->nchunk = ConfigHelper::get_npart(in_len, chunk_size);
    auto nchunk = ctx->nchunk;

    auto& h_counts = huffman.h_counts;
    auto& d_counts = huffman.d_counts;

    auto& h_bitstream = huffman.h_bitstream;
    auto& d_bitstream = huffman.d_bitstream;

    // arguments above

    cudaMalloc(&tmp_space, sizeof(H) * (in_len + chunk_size + HuffmanHelper::BLOCK_DIM_ENCODE));

    // gather metadata (without write) before gathering huff as sp on GPU
    cudaMallocHost(&h_counts, nchunk * 3 * sizeof(size_t));
    cudaMalloc(&d_counts, nchunk * 3 * sizeof(size_t));

    auto d_bits    = d_counts;
    auto d_uints   = d_counts + nchunk;
    auto d_entries = d_counts + nchunk * 2;

    lossless::HuffmanEncode<E, H, false>(
        tmp_space, d_bits, d_uints, d_entries, h_counts,
        //
        nullptr,
        //
        quant->dptr, book->dptr, in_len, chunk_size, dict_size, &num_bits, &num_uints, time.lossless);

    // --------------------------------------------------------------------------------
    cudaMallocHost(&h_bitstream, num_uints * sizeof(H));
    cudaMalloc(&d_bitstream, num_uints * sizeof(H));

    lossless::HuffmanEncode<E, H, true>(
        tmp_space, nullptr, d_uints, d_entries, nullptr,
        //
        d_bitstream,
        //
        nullptr, nullptr, in_len, chunk_size, 0, nullptr, nullptr, time.lossless);

    // --------------------------------------------------------------------------------
    // TODO change to `d2h()`
    cudaMemcpy(h_bitstream, d_bitstream, num_uints * sizeof(H), cudaMemcpyDeviceToHost);

    // TODO size_t -> MetadataT
    dataseg.nbyte.at("huff-meta")      = sizeof(size_t) * (2 * nchunk);
    dataseg.nbyte.at("huff-bitstream") = sizeof(H) * num_uints;

    cudaFree(tmp_space);

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
    header->nbyte.book           = dataseg.nbyte.at("book");
    header->nbyte.revbook        = dataseg.nbyte.at("revbook");
    header->nbyte.outlier        = dataseg.nbyte.at("outlier");
    header->nbyte.huff_meta      = dataseg.nbyte.at("huff-meta");
    header->nbyte.huff_bitstream = dataseg.nbyte.at("huff-bitstream");

    // consolidate
    std::vector<uint32_t> offsets = {0};

    ReportHelper::print_datasegment_tablehead();

    // print long numbers with thousand separator
    // https://stackoverflow.com/a/7455282
    // https://stackoverflow.com/a/11695246
    setlocale(LC_ALL, "");

    for (auto i = 0; i < 7; i++) {
        const auto& name = dataseg.order2name.at(i);

        auto o = offsets.back() + __cusz_get_alignable_len<BYTE, 128>(dataseg.nbyte.at(name));
        offsets.push_back(o);

        printf(
            "  %-18s\t%'12lu\t%'15lu\t%'15lu\n", name.c_str(),  //
            (size_t)dataseg.nbyte.at(name), (size_t)offsets.at(i), (size_t)offsets.back());
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
                dataseg.nbyte.at("header"),       //
                cudaMemcpyHostToHost);
            /* 1 */  // book
            /* 2 */  // quant

            if (dataseg.nbyte.at("anchor") != 0) /* a */  // anchor
                cudaMemcpy(
                    h_dump + offsets.at(0xa),                    //
                    reinterpret_cast<BYTE*>(huffman.h_revbook),  //
                    dataseg.nbyte.at("anchor"),                  //
                    cudaMemcpyHostToHost);

            /* 3 */  // revbook
            if (dataseg.nbyte.at("revbook") != 0)
                cudaMemcpy(
                    h_dump + offsets.at(3),                      //
                    reinterpret_cast<BYTE*>(huffman.h_revbook),  //
                    dataseg.nbyte.at("revbook"),                 //
                    cudaMemcpyHostToHost);
            /* 4 */  // outlier
            if (dataseg.nbyte.at("outlier") != 0)
                cudaMemcpy(
                    h_dump + offsets.at(4),            //
                    reinterpret_cast<BYTE*>(sp.dump),  //
                    dataseg.nbyte.at("outlier"),       //
                    cudaMemcpyHostToHost);
            /* 5 */  // huff_meta
            if (dataseg.nbyte.at("huff-meta") != 0)
                cudaMemcpy(
                    h_dump + offsets.at(5),                                   //
                    reinterpret_cast<BYTE*>(huffman.h_counts + ctx->nchunk),  //
                    dataseg.nbyte.at("huff-meta"),                            //
                    cudaMemcpyHostToHost);
            /* 6 */  // huff_bitstream
            if (dataseg.nbyte.at("huff-bitstream") != 0)
                cudaMemcpy(
                    h_dump + offsets.at(6),                        //
                    reinterpret_cast<BYTE*>(huffman.h_bitstream),  //
                    dataseg.nbyte.at("huff-bitstream"),            //
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

    Capsule<E>        quant(ctx->quant_len);
    Capsule<uint32_t> freq(ctx->dict_size);
    Capsule<H>        book(ctx->dict_size);
    Capsule<uint8_t>  revbook(HuffmanHelper::get_revbook_nbyte<E, H>(ctx->dict_size));
    cudaMalloc(&quant.dptr, quant.nbyte());
    cudaMalloc(&freq.dptr, freq.nbyte());
    cudaMalloc(&book.dptr, book.nbyte()), book.memset(0xff);
    cudaMalloc(&revbook.dptr, revbook.nbyte());
    cudaMallocHost(&revbook.hptr, revbook.nbyte());  // to write to disk later

    huffman.h_revbook = revbook.hptr;

    predictor->construct(in_data->dptr, nullptr, quant.dptr);
    csr->gather(in_data->dptr, sp.workspace, sp.dump, sp.dump_nbyte, ctx->nnz_outlier);

    time.lossy   = predictor->get_time_elapsed();
    time.outlier = csr->get_time_elapsed();

    dataseg.nbyte.at("outlier") = sp.dump_nbyte;  // do before consolidate

    LOGGING(LOG_INFO, "#outlier = ", ctx->nnz_outlier, StringHelper::nnz_percentage(ctx->nnz_outlier, ctx->data_len));

    try_skip_huffman(&quant);

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

COMPR_TYPE
void COMPRESSOR::try_report_decompression_time()
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

COMPR_TYPE
void COMPRESSOR::unpack_metadata()
{
    dataseg.nbyte.at("book")           = header->nbyte.book;
    dataseg.nbyte.at("revbook")        = header->nbyte.revbook;
    dataseg.nbyte.at("outlier")        = header->nbyte.outlier;
    dataseg.nbyte.at("huff-meta")      = header->nbyte.huff_meta;
    dataseg.nbyte.at("huff-bitstream") = header->nbyte.huff_bitstream;

    /* 0 header */ dataseg.offset.push_back(0);

    if (ctx->verbose) { printf("(verification)"), ReportHelper::print_datasegment_tablehead(), setlocale(LC_ALL, ""); }

    for (auto i = 0; i < 7; i++) {
        const auto& name  = dataseg.order2name.at(i);
        auto        nbyte = dataseg.nbyte.at(name);
        auto        o     = dataseg.offset.back() + __cusz_get_alignable_len<BYTE, 128>(nbyte);
        dataseg.offset.push_back(o);

        if (ctx->verbose) {
            printf(
                "  %-18s\t%'12lu\t%'15lu\t%'15lu\n", name.c_str(), (size_t)dataseg.nbyte.at(name),
                (size_t)dataseg.offset.at(i), (size_t)dataseg.offset.back());
        }
    }
    if (ctx->verbose) printf("\n");

    ConfigHelper::deep_copy_config_items(/* dst */ ctx, /* src */ header);
    ConfigHelper::set_eb_series(ctx->eb, config);
}

COMPR_TYPE
void COMPRESSOR::try_compare_with_origin(T* xdata)
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

COMPR_TYPE
void COMPRESSOR::try_write2disk(T* host_xdata)
{
    if (ctx->to_skip.write2disk)
        LOGGING(LOG_INFO, "output: skipped");
    else {
        LOGGING(LOG_INFO, "output:", ctx->fnames.path_basename + ".cuszx");
        io::write_array_to_binary(ctx->fnames.path_basename + ".cuszx", host_xdata, ctx->data_len);
    }
}

COMPR_TYPE
void COMPRESSOR::decompress()
{
    Capsule<E> quant(ctx->quant_len);
    cudaMalloc(&quant.dptr, quant.nbyte());
    cudaMallocHost(&quant.hptr, quant.nbyte());

    Capsule<T> decomp_space(mxm + ChunkingTrait<1>::BLOCK);  // TODO ad hoc size
    cudaMalloc(&decomp_space.dptr, decomp_space.nbyte());
    cudaMallocHost(&decomp_space.hptr, decomp_space.nbyte());
    auto xdata = decomp_space.dptr, outlier = decomp_space.dptr;

    // reducer->decode(nullptr, quant.dptr);
    using Mtype = typename cusz::HuffmanWork<E, H>::Mtype;

    // TODO pass dump and dataseg description
    // problem statement:
    // Data are described in two ways:
    // 1) fields of singleton, which are found&accessed by offset, or
    // 2) scattered, which are access f&a by addr (in absolute value)
    // Therefore, reducer::decode() should be
    // decode(WHERE, FROM_DUMP, dump, offset, output)

    reducer->decode(
        cusz::WHERE::HOST,                                                  //
        reinterpret_cast<H*>(dump + dataseg.get_offset("huff-bitstream")),  //
        reinterpret_cast<Mtype*>(dump + dataseg.get_offset("huff-meta")),   //
        reinterpret_cast<BYTE*>(dump + dataseg.get_offset("revbook")),      //
        quant.dptr);

    csr->scatter(csr_file.dev, outlier);
    predictor->reconstruct(nullptr, quant.dptr, xdata);

    time.lossless = reducer->get_time_elapsed();
    time.outlier  = csr->get_time_elapsed();
    time.lossy    = predictor->get_time_elapsed();

    try_report_decompression_time();

    decomp_space.d2h();

    try_compare_with_origin(decomp_space.hptr);
    try_write2disk(decomp_space.hptr);
}

////////////////////////////////////////////////////////////////////////////////

template class Compressor<float, ErrCtrlTrait<2>::type, HuffTrait<4>::type, FastLowPrecisionTrait<true>::type>;
template class Compressor<float, ErrCtrlTrait<2>::type, HuffTrait<8>::type, FastLowPrecisionTrait<true>::type>;
