/**
 * @file default_path.cu
 * @author Jiannan Tian
 * @brief cuSZ compressor of the default path
 * @version 0.3
 * @date 2021-10-05
 * (create) 2020-02-12; (release) 2020-09-20;
 * (rev.1) 2021-01-16; (rev.2) 2021-07-12; (rev.3) 2021-09-06; (rev.4) 2021-10-05
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "analysis/analyzer.hh"
#include "default_path.cuh"
#include "wrapper.hh"

#define DPCOMPRESSOR_TYPE template <class BINDING>
#define DPCOMPRESSOR DefaultPathCompressor<BINDING>

DPCOMPRESSOR_TYPE
unsigned int DPCOMPRESSOR::tune_deflate_chunksize(size_t len)
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

DPCOMPRESSOR_TYPE
DPCOMPRESSOR& DPCOMPRESSOR::analyze_compressibility()
{
    if (this->ctx->report.compressibility) {
        // cudaMallocHost(&this->freq.hptr, this->freq.nbyte()), this->freq.device2host();
        // cudaMallocHost(&book.hptr, book.nbyte()), book.device2host();
        this->freq.template alloc<cuszDEV::DEV, cuszLOC::HOST>().device2host();
        book.template alloc<cuszDEV::DEV, cuszLOC::HOST>().device2host();

        Analyzer analyzer{};
        analyzer  //
            .estimate_compressibility_from_histogram(this->freq.hptr, this->ctx->dict_size)
            .template get_stat_from_huffman_book<H>(
                this->freq.hptr, book.hptr, this->ctx->data_len, this->ctx->dict_size)
            .print_compressibility(true);

        cudaFreeHost(this->freq.hptr);
        cudaFreeHost(book.hptr);
    }

    return *this;
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR& DPCOMPRESSOR::internal_eval_try_export_book()
{
    // internal evaluation, not stored in sz archive
    if (this->ctx->export_raw.book) {
        cudaMallocHost(&book.hptr, this->ctx->dict_size * sizeof(decltype(book.hptr)));
        book.device2host();

        std::stringstream s;
        s << this->ctx->fnames.path_basename + "-" << this->ctx->dict_size << "-ui" << sizeof(H) << ".lean-book";

        // TODO as part of dump
        io::write_array_to_binary(s.str(), book.hptr, this->ctx->dict_size);

        cudaFreeHost(book.hptr);
        book.hptr = nullptr;

        LOGGING(LOG_INFO, "exporting codebook as binary; suffix: \".lean-book\"");

        this->dataseg.nbyte.at(cuszSEG::BOOK) = this->ctx->dict_size * sizeof(H);
    }
    return *this;
}

DPCOMPRESSOR_TYPE DPCOMPRESSOR& DPCOMPRESSOR::internal_eval_try_export_quant()
{
    // internal_eval
    if (this->ctx->export_raw.quant) {  //
        this->quant.template alloc<cuszDEV::DEV, cuszLOC::HOST>();
        this->quant.device2host();

        this->dataseg.nbyte.at(cuszSEG::QUANT) = this->quant.nbyte();

        // TODO as part of dump
        io::write_array_to_binary(
            this->ctx->fnames.path_basename + ".lean-this->quant", this->quant.hptr, this->ctx->quant_len);
        LOGGING(LOG_INFO, "exporting this->quant as binary; suffix: \".lean-this->quant\"");
        LOGGING(LOG_INFO, "exiting");
        exit(0);
    }
    return *this;
}

DPCOMPRESSOR_TYPE DPCOMPRESSOR& DPCOMPRESSOR::try_skip_huffman()
{
    // decide if skipping Huffman coding
    if (this->ctx->to_skip.huffman) {
        // cudaMallocHost(&this->quant.hptr, this->quant.nbyte());
        this->quant  //
            .template alloc<cuszDEV::DEV, cuszLOC::HOST>()
            .device2host();

        // TODO: as part of cusza
        io::write_array_to_binary(
            this->ctx->fnames.path_basename + ".this->quant", this->quant.hptr, this->ctx->quant_len);
        LOGGING(LOG_INFO, "to store this->quant.code directly (Huffman enc skipped)");
        exit(0);
    }

    return *this;
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR& DPCOMPRESSOR::get_freq_codebook()
{
    wrapper::get_frequency<E>(
        this->quant.dptr, this->ctx->quant_len, this->freq.dptr, this->ctx->dict_size, this->time.hist);

    {  // This is end-to-end time for parbook.
        auto t = new cuda_timer_t;
        t->timer_start();
        lossless::par_get_codebook<E, H>(this->ctx->dict_size, this->freq.dptr, book.dptr, revbook.dptr);
        this->time.book = t->timer_end_get_elapsed_time();
        cudaDeviceSynchronize();
        delete t;
    }

    this->analyze_compressibility()  //
        .internal_eval_try_export_book()
        .internal_eval_try_export_quant();

    revbook.device2host();  // need processing on CPU
    this->dataseg.nbyte.at(cuszSEG::REVBOOK) = HuffmanHelper::get_revbook_nbyte<E, H>(this->ctx->dict_size);

    return *this;
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR& DPCOMPRESSOR::huffman_encode()
{
    // fix-length space, padding improvised

    H*    tmp_space;
    auto  in_len     = this->ctx->quant_len;
    auto  chunk_size = this->ctx->huffman_chunk;
    auto  dict_size  = this->ctx->dict_size;
    auto& num_bits   = this->ctx->huffman_num_bits;
    auto& num_uints  = this->ctx->huffman_num_uints;

    this->ctx->nchunk = ConfigHelper::get_npart(in_len, chunk_size);
    auto nchunk       = this->ctx->nchunk;

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
        this->quant.dptr, book.dptr, in_len, chunk_size, dict_size, &num_bits, &num_uints, this->time.lossless);

    // --------------------------------------------------------------------------------
    huff_data  //
        .set_len(num_uints)
        .template alloc<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();

    lossless::HuffmanEncode<E, H, true>(
        tmp_space, nullptr, d_uints, d_entries, nullptr,
        //
        huff_data.dptr,
        //
        nullptr, nullptr, in_len, chunk_size, 0, nullptr, nullptr, this->time.lossless);

    // --------------------------------------------------------------------------------
    huff_data.device2host();

    // TODO size_t -> MetadataT
    this->dataseg.nbyte.at(cuszSEG::HUFF_META) = sizeof(size_t) * (2 * nchunk);
    this->dataseg.nbyte.at(cuszSEG::HUFF_DATA) = sizeof(H) * num_uints;

    cudaFree(tmp_space);

    return *this;
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR::DefaultPathCompressor(cuszCTX* _ctx, Capsule<T>* _in_data)
{
    this->ctx     = _ctx;
    this->in_data = _in_data;
    this->timing  = cuszWHEN::COMPRESS;
    this->header  = new cusz_header();
    this->xyz     = dim3(this->ctx->x, this->ctx->y, this->ctx->z);

    this->prescan();  // internally change eb (regarding value range)
    ConfigHelper::set_eb_series(this->ctx->eb, this->config);

    if (this->ctx->on_off.autotune_huffchunk) this->ctx->huffman_chunk = tune_deflate_chunksize(this->ctx->data_len);

    predictor             = new Predictor(this->xyz, this->ctx->eb, this->ctx->radius, false);
    this->ctx->quant_len  = predictor->get_quant_len();
    this->ctx->anchor_len = predictor->get_anchor_len();

    csr = new SpReducer(BINDING::template get_spreducer_input_len<cuszCTX>(this->ctx));
    sp_use.set_len(SparseMethodSetup::get_init_csr_nbyte<T, int>(this->ctx->data_len))
        .template alloc<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();

    LOGGING(LOG_INFO, "compressing...");
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR::DefaultPathCompressor(cuszCTX* _ctx, Capsule<BYTE>* _in_dump)
{
    this->ctx     = _ctx;
    this->in_dump = _in_dump;
    this->timing  = cuszWHEN::DECOMPRESS;
    auto dump     = this->in_dump->hptr;

    this->header = reinterpret_cast<cusz_header*>(dump);
    this->unpack_metadata();
    this->xyz = dim3(this->header->x, this->header->y, this->header->z);

    csr = new SpReducer(BINDING::template get_spreducer_input_len<cuszCTX>(this->ctx), this->ctx->nnz_outlier);
    sp_use.set_len(csr->get_total_nbyte())
        .template from_existing_on<cuszLOC::HOST>(
            reinterpret_cast<BYTE*>(dump + this->dataseg.get_offset(cuszSEG::OUTLIER)))
        .template alloc<cuszDEV::DEV, cuszLOC::DEVICE>()
        .host2device();

    predictor = new Predictor(this->xyz, this->ctx->eb, this->ctx->radius, false);

    codec = new Encoder(
        BINDING::template get_encoder_input_len(this->ctx), dump, this->header->huffman_chunk,
        this->header->huffman_num_uints, this->header->dict_size);

    LOGGING(LOG_INFO, "decompressing...");
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR::~DefaultPathCompressor()
{
    if (this->timing == cuszWHEN::COMPRESS) {  // release small-size arrays

        this->quant.template free<cuszDEV::DEV, cuszLOC::DEVICE>();
        this->freq.template free<cuszDEV::DEV, cuszLOC::DEVICE>();
        huff_data.template free<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();
        huff_counts.template free<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();
        sp_use.template free<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();
        book.template free<cuszDEV::DEV, cuszLOC::DEVICE>();
        revbook.template free<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();

        delete this->header;
    }
    else {
        cudaFree(sp_use.dptr);
    }

    delete csr;
    delete predictor;
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR& DPCOMPRESSOR::compress()
{
    this->dryrun();  // TODO

    this->quant.set_len(this->ctx->quant_len).template alloc<cuszDEV::DEV, cuszLOC::DEVICE>();
    this->freq.set_len(this->ctx->dict_size).template alloc<cuszDEV::DEV, cuszLOC::DEVICE>();
    book.set_len(this->ctx->dict_size).template alloc<cuszDEV::DEV, cuszLOC::DEVICE>();
    revbook
        .set_len(  //
            HuffmanHelper::get_revbook_nbyte<E, H>(this->ctx->dict_size))
        .template alloc<cuszDEV::DEV, cuszLOC::HOST_DEVICE>();

    predictor->construct(this->in_data->dptr, nullptr, this->quant.dptr);
    csr->gather(this->in_data->dptr, sp_dump_nbyte, this->ctx->nnz_outlier);
    csr->template consolidate<cuszLOC::DEVICE, cuszLOC::HOST>(sp_use.hptr);

    this->time.lossy   = predictor->get_time_elapsed();
    this->time.outlier = csr->get_time_elapsed();

    this->dataseg.nbyte.at(cuszSEG::OUTLIER) = sp_dump_nbyte;  // do before consolidate

    LOGGING(
        LOG_INFO, "#outlier = ", this->ctx->nnz_outlier,
        StringHelper::nnz_percentage(this->ctx->nnz_outlier, this->ctx->data_len));

    try_skip_huffman();

    // release in_data; subject to change
    cudaFree(this->in_data->dptr);

    this->get_freq_codebook()  //
        .huffman_encode()
        .try_report_compress_time();
    this->pack_metadata();

    return *this;
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR& DPCOMPRESSOR::decompress(Capsule<T>* decomp_space)
{
    this->quant.set_len(this->ctx->quant_len).template alloc<cuszDEV::DEV, cuszLOC::DEVICE>();
    auto xdata = decomp_space->dptr, outlier = decomp_space->dptr;

    using Mtype = typename Encoder::Mtype;

    // TODO pass dump and this->dataseg description
    // problem statement:
    // Data are described in two ways:
    // 1) fields of singleton, which are found&accessed by offset, or
    // 2) scattered, which are access f&a by addr (in absolute value)
    // Therefore, codec::decode() should be
    // decode(WHERE, FROM_DUMP, dump, offset, output)

    codec->decode(
        cuszLOC::HOST,                                                                                 //
        reinterpret_cast<H*>(this->in_dump->hptr + this->dataseg.get_offset(cuszSEG::HUFF_DATA)),      //
        reinterpret_cast<Mtype*>(this->in_dump->hptr + this->dataseg.get_offset(cuszSEG::HUFF_META)),  //
        reinterpret_cast<BYTE*>(this->in_dump->hptr + this->dataseg.get_offset(cuszSEG::REVBOOK)),     //
        this->quant.dptr);

    csr->scatter(sp_use.dptr, outlier);
    predictor->reconstruct(nullptr, this->quant.dptr, xdata);

    return *this;
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR& DPCOMPRESSOR::backmatter(Capsule<T>* decomp_space)
{
    decomp_space->device2host();

    this->time.lossless = codec->get_time_elapsed();
    this->time.outlier  = csr->get_time_elapsed();
    this->time.lossy    = predictor->get_time_elapsed();
    this->try_report_decompress_time();

    this->try_compare_with_origin(decomp_space->hptr);
    this->try_write2disk(decomp_space->hptr);

    return *this;
}

DPCOMPRESSOR_TYPE
template <cuszLOC FROM, cuszLOC TO>
DPCOMPRESSOR& DPCOMPRESSOR::consolidate(BYTE** dump_ptr)
{
    constexpr auto        DIRECTION = CopyDirection<FROM, TO>::direction;
    std::vector<uint32_t> offsets   = {0};

    auto REINTERP = [](auto* ptr) { return reinterpret_cast<BYTE*>(ptr); };
    auto ADDR     = [&](int seg_id) { return *dump_ptr + offsets.at(seg_id); };
    auto COPY     = [&](cuszSEG seg, auto src) {
        auto dst      = ADDR(this->dataseg.name2order.at(seg));
        auto src_byte = REINTERP(src);
        auto len      = this->dataseg.nbyte.at(seg);
        if (len != 0) cudaMemcpy(dst, src_byte, len, DIRECTION);
    };

    DatasegHelper::compress_time_conslidate_report(this->dataseg, offsets);
    auto total_nbyte = offsets.back();
    printf("\ncompression ratio:\t%.4f\n", this->ctx->data_len * sizeof(T) * 1.0 / total_nbyte);

    if CONSTEXPR (TO == cuszLOC::HOST)
        cudaMallocHost(dump_ptr, total_nbyte);
    else if (TO == cuszLOC::DEVICE)
        cudaMalloc(dump_ptr, total_nbyte);
    else
        throw std::runtime_error("[COMPRESSOR::consolidate] undefined behavior");

    COPY(cuszSEG::HEADER, this->header);
    COPY(cuszSEG::ANCHOR, this->anchor.template get<FROM>());
    COPY(cuszSEG::REVBOOK, revbook.template get<FROM>());
    COPY(cuszSEG::OUTLIER, sp_use.template get<FROM>());
    COPY(cuszSEG::HUFF_META, huff_counts.template get<FROM>() + this->ctx->nchunk);
    COPY(cuszSEG::HUFF_DATA, huff_data.template get<FROM>());

    return *this;
}

#define DPC_DC DefaultPathCompressor<DefaultPath::DefaultBinding>

template class DPC_DC;

template DPC_DC& DPC_DC::consolidate<cuszLOC::HOST, cuszLOC::HOST>(BYTE**);
template DPC_DC& DPC_DC::consolidate<cuszLOC::HOST, cuszLOC::DEVICE>(BYTE**);
template DPC_DC& DPC_DC::consolidate<cuszLOC::DEVICE, cuszLOC::HOST>(BYTE**);
template DPC_DC& DPC_DC::consolidate<cuszLOC::DEVICE, cuszLOC::DEVICE>(BYTE**);

#define DPC_FC DefaultPathCompressor<DefaultPath::FallbackBinding>

template class DPC_FC;

template DPC_FC& DPC_FC::consolidate<cuszLOC::HOST, cuszLOC::HOST>(BYTE**);
template DPC_FC& DPC_FC::consolidate<cuszLOC::HOST, cuszLOC::DEVICE>(BYTE**);
template DPC_FC& DPC_FC::consolidate<cuszLOC::DEVICE, cuszLOC::HOST>(BYTE**);
template DPC_FC& DPC_FC::consolidate<cuszLOC::DEVICE, cuszLOC::DEVICE>(BYTE**);
