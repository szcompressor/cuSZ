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
    cout << "autotuning deflate chunksize\n";

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
    auto display_throughput = [](float time, size_t nbyte) {
        auto throughput = nbyte * 1.0 / (1024 * 1024 * 1024) / (time * 1e-3);
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
void COMPRESSOR::export_codebook(Huff* d_book, const string& basename, size_t dict_size)
{
    auto              h_book = mem::create_devspace_memcpy_d2h(d_book, dict_size);
    std::stringstream s;
    s << basename + "-" << dict_size << "-ui" << sizeof(Huff) << ".lean-book";
    logging(log_dbg, "export \"lean\" codebook (of dict_size) as", s.str());
    io::write_array_to_binary(s.str(), h_book, dict_size);
    cudaFreeHost(h_book);
    h_book = nullptr;
}

COMPR_TYPE
COMPRESSOR::Compressor(argpack* _ap, unsigned int _data_len, double _eb)
{
    ctx = _ap;

    ndim = ctx->ndim;

    config.radius = ctx->radius;

    length.data      = _data_len;
    length.quant     = length.data;  // TODO if lorenzo
    length.dict_size = ctx->dict_size;

    config.eb     = _eb;
    config.ebx2   = _eb * 2;
    config.ebx2_r = 1 / (_eb * 2);
    config.eb_r   = 1 / _eb;

    if (ctx->task_is.autotune_huffchunk) ctx->huffman_chunk = tune_deflate_chunksize(length.data);
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
    logging(log_info, "invoke lossy-construction");
    // TODO "predictor" -> "prediction"
    if (ctx->task_is.predictor == "lorenzo") {
        // TODO class lorenzo
        compress_lorenzo_construct<Data, Quant, float>(
            data->dptr, quant->dptr, xyz, ctx->ndim, config.eb, config.radius, time.lossy);
    }
    else if (ctx->task_is.predictor == "spline3d") {
        if (ctx->ndim != 3) throw std::runtime_error("must be 3D data.");
        // TODO timer
        spline3->predict_quantize(data->dptr, anchor->dptr, quant->dptr, config.radius);
    }
    else {
        throw std::runtime_error("must be \"lorenzo\" or \"spline3d\"");
    }

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::gather_outlier(Capsule<Data>* in_data)
{
    unsigned int workspace_nbyte, dump_nbyte;
    uint8_t *    workspace, *dump;
    workspace_nbyte = get_compress_sparse_workspace<Data>(length.data);
    cudaMalloc((void**)&workspace, workspace_nbyte);
    cudaMallocHost((void**)&dump, workspace_nbyte);

    OutlierHandler<Data> csr(length.data);
    csr.configure(workspace)  //
        .gather_CUDA10(in_data->dptr, dump_nbyte, time.outlier)
        .archive(dump, length.nnz_outlier);
    io::write_array_to_binary(ctx->subfiles.compress.out_outlier, dump, dump_nbyte);

    cudaFree(workspace), cudaFreeHost(dump);

    auto fmt_nnz = "(" + std::to_string(length.nnz_outlier / 1.0 / length.data * 100) + "%)";
    logging(log_info, "nnz/#outlier:", length.nnz_outlier, fmt_nnz, "saved");

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
    if (ctx->task_is.export_book) {  //
        export_codebook(book->dptr, ctx->subfiles.compress.huff_base, length.dict_size);
        logging(log_info, "exporting codebook as binary; suffix: \".lean-book\"");
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

        io::write_array_to_binary(ctx->subfiles.compress.raw_quant, quant->hptr, length.quant);
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

        io::write_array_to_binary(ctx->subfiles.compress.out_quant, quant->hptr, length.quant);
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
    io::write_array_to_binary(ctx->subfiles.compress.huff_base + ".canon", revbook->hptr, get_revbook_nbyte());
    cudaFreeHost(revbook->hptr);

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

    cudaMemcpy(
        huffman.array.h_bitstream, huffman.array.d_bitstream, huffman.meta.num_uints * sizeof(Huff),
        cudaMemcpyDeviceToHost);
    io::write_array_to_binary(
        ctx->subfiles.compress.huff_base + ".hbyte", huffman.array.h_bitstream, huffman.meta.num_uints);

    io::write_array_to_binary(ctx->subfiles.compress.huff_base + ".hmeta", huffman.array.h_counts + nchunk, 2 * nchunk);

    cudaFree(huffman.array.d_encspace);
    cudaFree(huffman.array.d_counts);
    cudaFree(huffman.array.d_bitstream);

    cudaFreeHost(huffman.array.h_bitstream);
    cudaFreeHost(huffman.array.h_counts);

    huffman.meta.revbook_nbyte = get_revbook_nbyte();

    return *this;
}

COMPR_TYPE
COMPRESSOR& COMPRESSOR::pack_metadata(cusz_header* header)
{
    header->dim4    = ctx->dim4;
    header->stride4 = ctx->stride4;
    header->nblk4   = ctx->nblk4;
    header->ndim    = ctx->ndim;
    header->eb      = ctx->eb;
    header->len     = ctx->len;

    header->nnz = length.nnz_outlier;

    if (ctx->dtype == "f32") header->dtype = DataType::kF32;
    if (ctx->dtype == "f64") header->dtype = DataType::kF64;

    header->quant_byte    = ctx->quant_byte;
    header->huff_byte     = ctx->huff_byte;
    header->huffman_chunk = ctx->huffman_chunk;
    header->skip_huffman  = ctx->task_is.skip_huffman;

    header->num_bits      = huffman.meta.num_bits;
    header->num_uints     = huffman.meta.num_uints;
    header->revbook_nbyte = huffman.meta.revbook_nbyte;

    return *this;
}

////////////////////////////////////////////////////////////////////////////////

#define DECOMPR_TYPE template <typename Data, typename Quant, typename Huff, typename FP>
#define DECOMPRESSOR Decompressor<Data, Quant, Huff, FP>

DECOMPR_TYPE
void DECOMPRESSOR::unpack_metadata(cusz_header* header, argpack* ctx)
{
    ctx->dim4    = header->dim4;
    ctx->stride4 = header->stride4;
    ctx->nblk4   = header->nblk4;
    ctx->ndim    = header->ndim;
    ctx->eb      = header->eb;
    ctx->len     = header->len;

    if (header->dtype == DataType::kF32) ctx->dtype = "f32";
    if (header->dtype == DataType::kF64) ctx->dtype = "f64";

    ctx->quant_byte           = header->quant_byte;
    ctx->huff_byte            = header->huff_byte;
    ctx->huffman_chunk        = header->huffman_chunk;
    ctx->task_is.skip_huffman = header->skip_huffman;
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
DECOMPRESSOR::Decompressor(cusz_header* header, argpack* _ap)
{
    logging(log_info, "invoke lossy-reconstruction");

    unpack_metadata(header, _ap);

    length.nnz_outlier         = header->nnz;
    huffman.meta.num_uints     = header->num_uints;
    huffman.meta.revbook_nbyte = header->revbook_nbyte;

    ctx          = _ap;
    length.data  = ctx->len;
    length.quant = length.data;  // TODO if lorenzo

    config.eb     = ctx->eb;
    config.ebx2   = config.eb * 2;
    config.ebx2_r = 1 / (config.eb * 2);
    config.eb_r   = 1 / config.eb;

    m   = static_cast<size_t>(ceil(sqrt(length.data)));
    mxm = m * m;
}

DECOMPR_TYPE
DECOMPRESSOR& DECOMPRESSOR::huffman_decode(Capsule<Quant>* quant)
{
    if (ctx->task_is.skip_huffman) {
        logging(log_info, "load quant.code from filesystem");
        io::read_binary_to_array(ctx->subfiles.decompress.in_quant, quant->hptr, quant->len);
        quant->h2d();
    }
    else {
        logging(log_info, "Huffman decode -> quant.code");
        lossless::HuffmanDecode<Quant, Huff>(
            ctx->subfiles.path2file, quant, ctx->len, ctx->huffman_chunk, huffman.meta.num_uints, ctx->dict_size,
            time.lossless);
    }
    return *this;
}

DECOMPR_TYPE
DECOMPRESSOR& DECOMPRESSOR::scatter_outlier(Data* outlier)
{
    OutlierHandler<Data> csr(length.data, length.nnz_outlier);

    uint8_t *h_csr_file, *d_csr_file;
    cudaMallocHost((void**)&h_csr_file, csr.bytelen.total);
    cudaMalloc((void**)&d_csr_file, csr.bytelen.total);

    io::read_binary_to_array<uint8_t>(ctx->subfiles.decompress.in_outlier, h_csr_file, csr.bytelen.total);
    cudaMemcpy(d_csr_file, h_csr_file, csr.bytelen.total, cudaMemcpyHostToDevice);

    csr.extract(d_csr_file).scatter_CUDA10(outlier, time.outlier);

    cudaFreeHost(h_csr_file);
    cudaFree(d_csr_file);

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
        spline3->reversed_predict_quantize(xdata, anchor, quant, ctx->radius);
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
    if (not ctx->subfiles.decompress.in_origin.empty() and ctx->report.quality) {
        logging(log_info, "load the original datum for comparison");

        auto odata = io::read_binary_to_new_array<Data>(ctx->subfiles.decompress.in_origin, length.data);

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
    logging(log_info, "output:", ctx->subfiles.path2file + ".szx");

    if (ctx->task_is.skip_write2disk)
        logging(log_dbg, "skip writing unzipped to filesystem");
    else {
        io::write_array_to_binary(ctx->subfiles.decompress.out_xdata, host_xdata, ctx->len);
    }

    return *this;
}

////////////////////////////////////////////////////////////////////////////////

#define DATATYPE Capsule<typename DataTrait<If_FP, DataByte>::Data>

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_compress(argpack* ctx, DATATYPE* in_data, dim3 xyz, cusz_header* header, unsigned int optional_w)
{
    using Data  = typename DataTrait<If_FP, DataByte>::Data;
    using Quant = typename QuantTrait<QuantByte>::Quant;
    using Huff  = typename HuffTrait<HuffByte>::Huff;

    Spline3<Data*, Quant*, float>        spline3(xyz, ctx->eb);
    Compressor<Data, Quant, Huff, float> cuszc(ctx, ctx->len, ctx->eb);
    cuszc.register_spline3(&spline3);

    // TODO lorenzo class::get_len_quant
    auto lorenzo_get_len_quant = [&]() -> unsigned int { return ctx->len + HuffConfig::Db_encode; };

    unsigned int len_quant = ctx->task_is.predictor == "spline3"  //
                                 ? spline3.get_len_quant()
                                 : lorenzo_get_len_quant();

    cuszc.lorenzo_dryrun(in_data);  // subject to change

    Capsule<Quant> quant(len_quant);
    cudaMalloc(&quant.dptr, quant.nbyte());

    Capsule<Data>* anchor = nullptr;
    if (ctx->task_is.predictor == "spline3") {
        anchor = new Capsule<Data>(spline3.get_len_anchor());
        cudaMalloc(&anchor->dptr, anchor->nbyte());
    }

    Capsule<unsigned int> freq(ctx->dict_size);
    cudaMalloc(&freq.dptr, freq.nbyte());

    Capsule<Huff> book(ctx->dict_size);
    cudaMalloc(&book.dptr, book.nbyte()), book.memset(0xff);

    Capsule<uint8_t> revbook(cuszc.get_revbook_nbyte());
    cudaMalloc(&revbook.dptr, revbook.nbyte());
    cudaMallocHost(&revbook.hptr, revbook.nbyte());  // to write to disk later

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
        .pack_metadata(header);

    cudaFree(quant.dptr), cudaFree(freq.dptr), cudaFree(book.dptr), cudaFree(revbook.dptr);
}

template <bool If_FP, int DataByte, int QuantByte, int HuffByte>
void cusz_decompress(argpack* ctx, cusz_header* header)
{
    using Data  = typename DataTrait<If_FP, DataByte>::Data;
    using Quant = typename QuantTrait<QuantByte>::Quant;
    using Huff  = typename HuffTrait<HuffByte>::Huff;

    // TODO "header, ctx" -> "ctx, header"
    // TODO float -> another parameter FP
    Decompressor<Data, Quant, Huff, float> cuszd(header, ctx);

    auto xyz = dim3(ctx->dim4._0, ctx->dim4._1, ctx->dim4._2);

    Spline3<Data*, Quant*, float> spline3(xyz, ctx->eb);
    cuszd.register_spline3(&spline3);

    // TODO lorenzo class::get_len_quant
    auto lorenzo_get_len_quant = [&]() -> unsigned int { return ctx->len; };

    unsigned int len_quant = ctx->task_is.predictor == "spline3"  //
                                 ? spline3.get_len_quant()
                                 : lorenzo_get_len_quant();

    Capsule<Data>* anchor = new Capsule<Data>(spline3.get_len_anchor());  // TODO this .dptr is nullable, error-prone
    if (ctx->task_is.predictor == "spline3") {
        cudaMalloc(&anchor->dptr, anchor->nbyte());
        cudaMallocHost(&anchor->hptr, anchor->nbyte());

        // TODO dummy, need source
    }

    Capsule<Quant> quant(len_quant);
    cudaMalloc(&quant.dptr, quant.nbyte());
    cudaMallocHost(&quant.hptr, quant.nbyte());

    // TODO cuszd.get_len_data_space()
    Capsule<Data> _data(cuszd.mxm + MetadataTrait<1>::Block);  // TODO ad hoc size
    cudaMalloc(&_data.dptr, _data.nbyte());
    cudaMallocHost(&_data.hptr, _data.nbyte());
    auto xdata = _data.dptr, outlier = _data.dptr;

    cuszd.huffman_decode(&quant)
        .scatter_outlier(outlier)
        .reversed_predict_quantize(xdata, xyz, anchor->dptr, quant.dptr)
        .try_report_time();

    // copy decompressed data to host
    _data.d2h();

    cuszd
        .calculate_archive_nbyte()  //
        .try_compare(_data.hptr)
        .try_write2disk(_data.hptr);
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
    template void cusz_compress<true, DBYTE, QBYTE, HBYTE>(argpack*, Capsule<float>*, dim3, cusz_header*, unsigned int);

CUSZ_COMPRESS(4, 1, 4)
CUSZ_COMPRESS(4, 1, 8)
CUSZ_COMPRESS(4, 2, 4)
CUSZ_COMPRESS(4, 2, 8)

#define CUSZ_DECOMPRESS(DBYTE, QBYTE, HBYTE) \
    template void cusz_decompress<true, DBYTE, QBYTE, HBYTE>(argpack*, cusz_header*);

CUSZ_DECOMPRESS(4, 1, 4)
CUSZ_DECOMPRESS(4, 1, 8)
CUSZ_DECOMPRESS(4, 2, 4)
CUSZ_DECOMPRESS(4, 2, 8)
