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
#include "common.hh"
#include "common/capsule.hh"
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

namespace {

void header_nbyte_from_dataseg(cuszHEADER* header, DataSeg& dataseg)
{
    header->nbyte.book           = dataseg.nbyte.at(cuszSEG::BOOK);
    header->nbyte.revbook        = dataseg.nbyte.at(cuszSEG::REVBOOK);
    header->nbyte.outlier        = dataseg.nbyte.at(cuszSEG::OUTLIER);
    header->nbyte.huff_meta      = dataseg.nbyte.at(cuszSEG::HUFF_META);
    header->nbyte.huff_bitstream = dataseg.nbyte.at(cuszSEG::HUFF_DATA);
}

void dataseg_nbyte_from_header(cuszHEADER* header, DataSeg& dataseg)
{
    dataseg.nbyte.at(cuszSEG::BOOK)      = header->nbyte.book;
    dataseg.nbyte.at(cuszSEG::REVBOOK)   = header->nbyte.revbook;
    dataseg.nbyte.at(cuszSEG::OUTLIER)   = header->nbyte.outlier;
    dataseg.nbyte.at(cuszSEG::HUFF_META) = header->nbyte.huff_meta;
    dataseg.nbyte.at(cuszSEG::HUFF_DATA) = header->nbyte.huff_bitstream;
}

void compress_time_conslidate_report(DataSeg& dataseg, std::vector<uint32_t>& offsets)
{
    ReportHelper::print_datasegment_tablehead();

    // print long numbers with thousand separator
    // https://stackoverflow.com/a/7455282
    // https://stackoverflow.com/a/11695246
    setlocale(LC_ALL, "");

    for (auto i = 0; i < 8; i++) {
        const auto& name       = dataseg.order2name.at(i);
        auto        this_nbyte = dataseg.nbyte.at(name);

        auto o = offsets.back() + __cusz_get_alignable_len<BYTE, 128>(this_nbyte);
        offsets.push_back(o);

        if (this_nbyte != 0)
            printf(
                "  %-18s\t%'12lu\t%'15lu\t%'15lu\n",  //
                dataseg.get_namestr(name).c_str(),    //
                (size_t)this_nbyte,                   //
                (size_t)offsets.at(i),                //
                (size_t)offsets.back());
    }
}

}  // namespace

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
COMPRESSOR& COMPRESSOR::lorenzo_dryrun(Capsule<T>* in_data)
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
    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::prescan()
{
    if (ctx->mode == "r2r") {
        auto result = Analyzer::get_maxmin_rng                         //
            <T, ExecutionPolicy::cuda_device, AnalyzerMethod::thrust>  //
            (in_data->dptr, in_data->len);
        if (ctx->verbose) LOGGING(LOG_DBG, "time scanning:", result.seconds, "sec");
        if (ctx->mode == "r2r") ctx->eb *= result.rng;
    }

    // TODO data "policy": (non-)destructive
    if (ctx->preprocess.binning) {
        LOGGING(LOG_ERR, "Binning is not working temporarily  (ver. 0.2.9)");
        exit(1);
    }

    return *this;
}

COMPR_TYPE
COMPRESSOR&
COMPRESSOR::get_freq_codebook(Capsule<E>* quant, Capsule<cusz::FREQ>* freq, Capsule<H>* book, Capsule<uint8_t>* revbook)
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

    revbook->device2host();  // need processing on CPU
    dataseg.nbyte.at(cuszSEG::REVBOOK) = HuffmanHelper::get_revbook_nbyte<E, H>(ctx->dict_size);

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::analyze_compressibility(Capsule<cusz::FREQ>* freq, Capsule<H>* book)
{
    if (ctx->report.compressibility) {
        cudaMallocHost(&freq->hptr, freq->nbyte()), freq->device2host();
        cudaMallocHost(&book->hptr, book->nbyte()), book->device2host();

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
        book->device2host();

        std::stringstream s;
        s << ctx->fnames.path_basename + "-" << ctx->dict_size << "-ui" << sizeof(H) << ".lean-book";

        // TODO as part of dump
        io::write_array_to_binary(s.str(), book->hptr, ctx->dict_size);

        cudaFreeHost(book->hptr);
        book->hptr = nullptr;

        LOGGING(LOG_INFO, "exporting codebook as binary; suffix: \".lean-book\"");

        dataseg.nbyte.at(cuszSEG::BOOK) = ctx->dict_size * sizeof(H);
    }
    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::internal_eval_try_export_quant(Capsule<E>* quant)
{
    // internal_eval
    if (ctx->export_raw.quant) {  //
        cudaMallocHost(&quant->hptr, quant->nbyte());
        quant->device2host();

        dataseg.nbyte.at(cuszSEG::QUANT) = quant->nbyte();

        // TODO as part of dump
        io::write_array_to_binary(ctx->fnames.path_basename + ".lean-quant", quant->hptr, ctx->quant_len);
        LOGGING(LOG_INFO, "exporting quant as binary; suffix: \".lean-quant\"");
        LOGGING(LOG_INFO, "exiting");
        exit(0);
    }
    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::huffman_encode(Capsule<E>* quant, Capsule<H>* book)
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

    // arguments above

    cudaMalloc(&tmp_space, sizeof(H) * (in_len + chunk_size + HuffmanHelper::BLOCK_DIM_ENCODE));

    // gather metadata (without write) before gathering huff as sp on GPU
    huff_counts  //
        .set_len(nchunk * 3)
        .template alloc<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();

    auto d_bits    = huff_counts.dptr;
    auto d_uints   = huff_counts.dptr + nchunk;
    auto d_entries = huff_counts.dptr + nchunk * 2;

    lossless::HuffmanEncode<E, H, false>(
        tmp_space, d_bits, d_uints, d_entries, huff_counts.hptr,
        //
        nullptr,
        //
        quant->dptr, book->dptr, in_len, chunk_size, dict_size, &num_bits, &num_uints, time.lossless);

    // --------------------------------------------------------------------------------
    huff_data  //
        .set_len(num_uints)
        .template alloc<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();

    lossless::HuffmanEncode<E, H, true>(
        tmp_space, nullptr, d_uints, d_entries, nullptr,
        //
        huff_data.dptr,
        //
        nullptr, nullptr, in_len, chunk_size, 0, nullptr, nullptr, time.lossless);

    // --------------------------------------------------------------------------------
    huff_data.device2host();

    // TODO size_t -> MetadataT
    dataseg.nbyte.at(cuszSEG::HUFF_META) = sizeof(size_t) * (2 * nchunk);
    dataseg.nbyte.at(cuszSEG::HUFF_DATA) = sizeof(H) * num_uints;

    cudaFree(tmp_space);

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::try_report_compress_time()
{
    if (not ctx->report.time) return *this;

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

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::try_report_decompress_time()
{
    if (not ctx->report.time) return *this;

    auto  nbyte = ctx->data_len * sizeof(T);
    float all   = time.lossy + time.outlier + time.lossless;

    ReportHelper::print_throughput_tablehead("decompression");
    ReportHelper::print_throughput_line("scatter-outlier", time.outlier, nbyte);
    ReportHelper::print_throughput_line("Huff-decode", time.lossless, nbyte);
    ReportHelper::print_throughput_line("reconstruct", time.lossy, nbyte);
    ReportHelper::print_throughput_line("(total)", all, nbyte);
    printf("\n");

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::try_skip_huffman(Capsule<E>* quant)
{
    // decide if skipping Huffman coding
    if (ctx->to_skip.huffman) {
        cudaMallocHost(&quant->hptr, quant->nbyte());
        quant->device2host();

        // TODO: as part of cusza
        io::write_array_to_binary(ctx->fnames.path_basename + ".quant", quant->hptr, ctx->quant_len);
        LOGGING(LOG_INFO, "to store quant.code directly (Huffman enc skipped)");
        exit(0);
    }

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::try_compare_with_origin(T* xdata)
{
    if (not ctx->fnames.origin_cmp.empty() and ctx->report.quality) {
        LOGGING(LOG_INFO, "compare to the original");

        auto odata = io::read_binary_to_new_array<T>(ctx->fnames.origin_cmp, ctx->data_len);

        analysis::verify_data(&ctx->stat, xdata, odata, ctx->data_len);
        analysis::print_data_quality_metrics<T>(&ctx->stat, in_dump->nbyte(), false);

        delete[] odata;
    }

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::try_write2disk(T* host_xdata)
{
    if (ctx->to_skip.write2disk)
        LOGGING(LOG_INFO, "output: skipped");
    else {
        LOGGING(LOG_INFO, "output:", ctx->fnames.path_basename + ".cuszx");
        io::write_array_to_binary(ctx->fnames.path_basename + ".cuszx", host_xdata, ctx->data_len);
    }

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::pack_metadata()
{
    ConfigHelper::deep_copy_config_items(/* dst */ header, /* src */ ctx);
    header_nbyte_from_dataseg(header, dataseg);
    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::unpack_metadata()
{
    dataseg_nbyte_from_header(header, dataseg);

    /* 0 header */ dataseg.offset.push_back(0);

    if (ctx->verbose) { printf("(verification)"), ReportHelper::print_datasegment_tablehead(), setlocale(LC_ALL, ""); }

    for (auto i = 0; i < 8; i++) {
        const auto& name       = dataseg.order2name.at(i);
        auto        this_nbyte = dataseg.nbyte.at(name);
        auto        o          = dataseg.offset.back() + __cusz_get_alignable_len<BYTE, 128>(this_nbyte);
        dataseg.offset.push_back(o);

        if (ctx->verbose) {
            if (this_nbyte != 0)
                printf(
                    "  %-18s\t%'12lu\t%'15lu\t%'15lu\n",  //
                    dataseg.get_namestr(name).c_str(),    //
                    (size_t)this_nbyte,                   //
                    (size_t)dataseg.offset.at(i),         //
                    (size_t)dataseg.offset.back());
        }
    }
    if (ctx->verbose) printf("\n");

    ConfigHelper::deep_copy_config_items(/* dst */ ctx, /* src */ header);
    ConfigHelper::set_eb_series(ctx->eb, config);

    return *this;
}

COMPR_TYPE
COMPRESSOR::Compressor(cuszCTX* _ctx, Capsule<BYTE>* _in_dump) : ctx(_ctx), in_dump(_in_dump)
{
    timing    = cuszWHEN::DECOMPRESS;
    auto dump = in_dump->hptr;

    header = reinterpret_cast<cusz_header*>(dump);

    unpack_metadata();

    m = Reinterpret1DTo2D::get_square_size(ctx->data_len), mxm = m * m;

    xyz = dim3(header->x, header->y, header->z);

    csr = new cusz::OutlierHandler10<T>(ctx->data_len, ctx->nnz_outlier);

    sp_use  //
        .set_len(csr->get_total_nbyte())
        .from_existing_on<cuszLOC::HOST>(  //
            reinterpret_cast<BYTE*>(dump + dataseg.get_offset(cuszSEG::OUTLIER)))
        .alloc<cuszDEV::DEV, cuszLOC::DEVICE>()
        .host2device();

    predictor = new cusz::PredictorLorenzo<T, E, FP>(xyz, ctx->eb, ctx->radius, false);

    reducer = new cusz::HuffmanWork<E, H>(
        header->quant_len, dump,  //
        header->huffman_chunk, header->huffman_num_uints, header->dict_size);

    LOGGING(LOG_INFO, "decompressing...");
}

COMPR_TYPE
COMPRESSOR::Compressor(cuszCTX* _ctx, Capsule<T>* _in_data) : ctx(_ctx), in_data(_in_data)
{
    timing = cuszWHEN::COMPRESS;
    header = new cusz_header();

    prescan();  // internally change eb (regarding value range)

    ConfigHelper::set_eb_series(ctx->eb, config);

    if (ctx->on_off.autotune_huffchunk) ctx->huffman_chunk = tune_deflate_chunksize(ctx->data_len);

    csr = new cusz::OutlierHandler10<T>(ctx->data_len);

    sp_use
        .set_len(  //
            SparseMethodSetup::get_init_csr_nbyte<T, int>(ctx->data_len))
        .template alloc<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();

    xyz = dim3(ctx->x, ctx->y, ctx->z);

    predictor = new cusz::PredictorLorenzo<T, E, FP>(xyz, ctx->eb, ctx->radius, false);

    ctx->quant_len  = predictor->get_quant_len();
    ctx->anchor_len = predictor->get_anchor_len();

    LOGGING(LOG_INFO, "compressing...");
}

COMPR_TYPE
COMPRESSOR::~Compressor()
{
    if (timing == cuszWHEN::COMPRESS) {  // release small-size arrays

        huff_data.template free<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();
        huff_counts.template free<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();
        sp_use.template free<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();
        quant.template free<cuszDEV::DEV, cuszLOC::DEVICE>();
        freq.template free<cuszDEV::DEV, cuszLOC::DEVICE>();
        book.template free<cuszDEV::DEV, cuszLOC::DEVICE>();
        revbook.template free<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();

        delete header;
    }
    else {
        cudaFree(sp_use.dptr);
    }

    delete csr;
    delete predictor;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::compress()
{
    lorenzo_dryrun(in_data);  // subject to change

    quant.set_len(ctx->quant_len).template alloc<cuszDEV::DEV, cuszLOC::DEVICE>();
    freq.set_len(ctx->dict_size).template alloc<cuszDEV::DEV, cuszLOC::DEVICE>();
    book.set_len(ctx->dict_size).template alloc<cuszDEV::DEV, cuszLOC::DEVICE>();
    revbook
        .set_len(  //
            HuffmanHelper::get_revbook_nbyte<E, H>(ctx->dict_size))
        .template alloc<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();

    predictor->construct(in_data->dptr, nullptr, quant.dptr);
    csr->gather(in_data->dptr, sp_use.dptr, sp_use.hptr, sp_dump_nbyte, ctx->nnz_outlier);

    time.lossy   = predictor->get_time_elapsed();
    time.outlier = csr->get_time_elapsed();

    dataseg.nbyte.at(cuszSEG::OUTLIER) = sp_dump_nbyte;  // do before consolidate

    LOGGING(LOG_INFO, "#outlier = ", ctx->nnz_outlier, StringHelper::nnz_percentage(ctx->nnz_outlier, ctx->data_len));

    try_skip_huffman(&quant);

    // release in_data; subject to change
    cudaFree(in_data->dptr);

    this->get_freq_codebook(&quant, &freq, &book, &revbook)
        .huffman_encode(&quant, &book)
        .try_report_compress_time()
        .pack_metadata();

    return *this;
}

COMPR_TYPE
template <cuszLOC FROM, cuszLOC TO>
COMPRESSOR& COMPRESSOR::consolidate(BYTE** dump_ptr)
{
    constexpr auto        DIRECTION = CopyDirection<FROM, TO>::direction;
    std::vector<uint32_t> offsets   = {0};

    auto REINTERP = [](auto* ptr) { return reinterpret_cast<BYTE*>(ptr); };
    auto ADDR     = [&](int seg_id) { return *dump_ptr + offsets.at(seg_id); };
    auto COPY     = [&](cuszSEG seg, auto src) {
        auto dst      = ADDR(dataseg.name2order.at(seg));
        auto src_byte = REINTERP(src);
        auto len      = dataseg.nbyte.at(seg);
        if (len != 0) cudaMemcpy(dst, src_byte, len, DIRECTION);
    };

    compress_time_conslidate_report(dataseg, offsets);
    auto total_nbyte = offsets.back();
    printf("\ncompression ratio:\t%.4f\n", ctx->data_len * sizeof(T) * 1.0 / total_nbyte);

    if CONSTEXPR (TO == cuszLOC::HOST)
        cudaMallocHost(dump_ptr, total_nbyte);
    else if (TO == cuszLOC::DEVICE)
        cudaMalloc(dump_ptr, total_nbyte);
    else
        throw std::runtime_error("[COMPRESSOR::consolidate] undefined behavior");

    COPY(cuszSEG::HEADER, header);
    COPY(cuszSEG::ANCHOR, anchor.template get<FROM>());
    COPY(cuszSEG::REVBOOK, revbook.template get<FROM>());
    COPY(cuszSEG::OUTLIER, sp_use.template get<FROM>());
    COPY(cuszSEG::HUFF_META, huff_counts.template get<FROM>() + ctx->nchunk);
    COPY(cuszSEG::HUFF_DATA, huff_data.template get<FROM>());

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::decompress(Capsule<T>* decomp_space)
{
    quant.set_len(ctx->quant_len).template alloc<cuszDEV::DEV, cuszLOC::DEVICE>();
    auto xdata = decomp_space->dptr, outlier = decomp_space->dptr;

    using Mtype = typename cusz::HuffmanWork<E, H>::Mtype;

    // TODO pass dump and dataseg description
    // problem statement:
    // Data are described in two ways:
    // 1) fields of singleton, which are found&accessed by offset, or
    // 2) scattered, which are access f&a by addr (in absolute value)
    // Therefore, reducer::decode() should be
    // decode(WHERE, FROM_DUMP, dump, offset, output)

    reducer->decode(
        cuszLOC::HOST,                                                                     //
        reinterpret_cast<H*>(in_dump->hptr + dataseg.get_offset(cuszSEG::HUFF_DATA)),      //
        reinterpret_cast<Mtype*>(in_dump->hptr + dataseg.get_offset(cuszSEG::HUFF_META)),  //
        reinterpret_cast<BYTE*>(in_dump->hptr + dataseg.get_offset(cuszSEG::REVBOOK)),     //
        quant.dptr);

    csr->scatter(sp_use.dptr, outlier);
    predictor->reconstruct(nullptr, quant.dptr, xdata);

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::backmatter(Capsule<T>* decomp_space)
{
    decomp_space->device2host();

    time.lossless = reducer->get_time_elapsed();
    time.outlier  = csr->get_time_elapsed();
    time.lossy    = predictor->get_time_elapsed();
    try_report_decompress_time();

    try_compare_with_origin(decomp_space->hptr);
    try_write2disk(decomp_space->hptr);

    return *this;
}

////////////////////////////////////////////////////////////////////////////////

#define Compressor424fast \
    Compressor<float, ErrCtrlTrait<2>::type, HuffTrait<4>::type, FastLowPrecisionTrait<true>::type>

template class Compressor424fast;

template Compressor424fast& Compressor424fast::consolidate<cuszLOC::HOST, cuszLOC::HOST>(BYTE**);
template Compressor424fast& Compressor424fast::consolidate<cuszLOC::HOST, cuszLOC::DEVICE>(BYTE**);
template Compressor424fast& Compressor424fast::consolidate<cuszLOC::DEVICE, cuszLOC::HOST>(BYTE**);
template Compressor424fast& Compressor424fast::consolidate<cuszLOC::DEVICE, cuszLOC::DEVICE>(BYTE**);

#define Compressor428fast \
    Compressor<float, ErrCtrlTrait<2>::type, HuffTrait<8>::type, FastLowPrecisionTrait<true>::type>

template class Compressor428fast;

template Compressor428fast& Compressor428fast::consolidate<cuszLOC::HOST, cuszLOC::HOST>(BYTE**);
template Compressor428fast& Compressor428fast::consolidate<cuszLOC::HOST, cuszLOC::DEVICE>(BYTE**);
template Compressor428fast& Compressor428fast::consolidate<cuszLOC::DEVICE, cuszLOC::HOST>(BYTE**);
template Compressor428fast& Compressor428fast::consolidate<cuszLOC::DEVICE, cuszLOC::DEVICE>(BYTE**);
