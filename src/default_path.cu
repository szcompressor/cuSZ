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
DPCOMPRESSOR& DPCOMPRESSOR::analyze_compressibility()
{
    if (this->ctx->report.compressibility) {
        this->freq.template alloc<kHOST>().device2host();
        book.template alloc<kHOST>().device2host();

        Analyzer analyzer{};
        analyzer  //
            .estimate_compressibility_from_histogram(this->freq.hptr, this->dict_size)
            .template get_stat_from_huffman_book<H>(this->freq.hptr, book.hptr, this->ctx->data_len, this->dict_size)
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
        cudaMallocHost(&book.hptr, this->dict_size * sizeof(decltype(book.hptr)));
        book.device2host();

        std::stringstream s;
        s << this->ctx->fname.path_basename + "-" << this->dict_size << "-ui" << sizeof(H) << ".lean-book";

        // TODO as part of dump
        io::write_array_to_binary(s.str(), book.hptr, this->dict_size);

        cudaFreeHost(book.hptr);
        book.hptr = nullptr;

        LOGGING(LOG_INFO, "exporting codebook as binary; suffix: \".lean-book\"");

        this->dataseg.nbyte.at(cusz::SEG::BOOK) = this->dict_size * sizeof(H);
    }
    return *this;
}

DPCOMPRESSOR_TYPE DPCOMPRESSOR& DPCOMPRESSOR::internal_eval_try_export_quant()
{
    // internal_eval
    if (this->ctx->export_raw.quant) {  //
        this->quant.template alloc<kHOST>();
        this->quant.device2host();

        this->dataseg.nbyte.at(cusz::SEG::QUANT) = this->quant.nbyte();

        // TODO as part of dump
        io::write_array_to_binary(
            this->ctx->fname.path_basename + ".lean-this->quant", this->quant.hptr,
            BINDING::template get_uncompressed_len(predictor, codec));
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
            .template alloc<kHOST>()
            .device2host();

        // TODO: as part of cusza
        io::write_array_to_binary(
            this->ctx->fname.path_basename + ".this->quant", this->quant.hptr,
            BINDING::template get_uncompressed_len(predictor, codec));
        LOGGING(LOG_INFO, "to store this->quant.code directly (Huffman enc skipped)");
        exit(0);
    }

    return *this;
}

// TODO the experiments left out
// this->analyze_compressibility()  //
//     .internal_eval_try_export_book()
//     .internal_eval_try_export_quant();

DPCOMPRESSOR_TYPE
DPCOMPRESSOR::DefaultPathCompressor(cuszCTX* _ctx, Capsule<T>* _in_data, uint3 xyz, int dict_size)
{
    static_assert(not std::is_same<BYTE, T>::value, "[DefaultPathCompressor constructor] T must not be BYTE.");

    this->ctx       = _ctx;
    this->original  = _in_data;
    this->timing    = cusz::WHEN::COMPRESS;
    this->header    = new cuszHEADER();
    this->xyz       = xyz;
    this->dict_size = dict_size;

    this->prescan();  // internally change eb (regarding value range)
    // ConfigHelper::set_eb_series(this->ctx->eb, this->config);

    predictor             = new Predictor(this->xyz, false);
    this->ctx->quant_len  = predictor->get_quant_len();
    this->ctx->anchor_len = predictor->get_anchor_len();

    // -----------------------------------------------------------------------------

    // TODO 21-12-17 toward static method
    codec = new Codec;

    // TODO change to codec-input-len (1)
    cudaMalloc(&huff_workspace, codec->get_workspace_nbyte(BINDING::template get_uncompressed_len(predictor, codec)));

    huff_data.set_len(codec->get_max_output_nbyte(BINDING::template get_uncompressed_len(predictor, codec)))
        .template alloc<kHOST_DEVICE>();

    // gather metadata (without write) before gathering huff as sp on GPU
    huff_counts.set_len(this->ctx->nchunk * 3).template alloc<kHOST_DEVICE>();

    // -----------------------------------------------------------------------------

    uint32_t init_nnz = this->ctx->data_len * this->ctx->nz_density;

    auto m = Reinterpret1DTo2D::get_square_size(BINDING::template get_uncompressed_len(predictor, spreducer));
    ext_rowptr.set_len(m + 1).template alloc<kDEVICE>();
    ext_colidx.set_len(init_nnz).template alloc<kDEVICE>();
    ext_values.set_len(init_nnz).template alloc<kDEVICE>();

    spreducer = new SpReducer;

    sp_use.set_len(SparseMethodSetup::get_csr_nbyte<T, int>(this->ctx->data_len, init_nnz))
        .template alloc<kHOST_DEVICE>();

    LOGGING(LOG_INFO, "compressing...");
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR::DefaultPathCompressor(cuszCTX* _ctx, Capsule<BYTE>* _in_dump)
{
    this->ctx        = _ctx;
    this->compressed = _in_dump;
    this->timing     = cusz::WHEN::DECOMPRESS;
    auto dump        = this->compressed->hptr;

    this->header = reinterpret_cast<cuszHEADER*>(dump);
    this->unpack_metadata();
    this->xyz = dim3(this->header->x, this->header->y, this->header->z);

    spreducer = new SpReducer;  // TODO resume SpReducer::constructor(uncompressed_len)

    // predictor = new Predictor(this->xyz, this->ctx->eb, this->ctx->radius, false);
    predictor = new Predictor(this->xyz, false);

    // TODO use a compressor method instead of spreducer's
    sp_use
        .set_len(spreducer->get_total_nbyte(                               //
            BINDING::template get_uncompressed_len(predictor, spreducer),  //
            this->ctx->nnz_outlier))
        .template shallow_copy<kHOST>(reinterpret_cast<BYTE*>(dump + this->dataseg.get_offset(cusz::SEG::SPFMT)))
        .template alloc<kDEVICE>()
        .host2device();

    codec = new Codec;
    {
        auto nchunk = ConfigHelper::get_npart(
            BINDING::template get_uncompressed_len(predictor, codec), this->header->huffman_chunksize);

        auto _h_data = reinterpret_cast<H*>(this->compressed->hptr + this->dataseg.get_offset(cusz::SEG::HUFF_DATA));
        auto _h_meta = reinterpret_cast<M*>(this->compressed->hptr + this->dataseg.get_offset(cusz::SEG::HUFF_META));
        auto _h_rev  = reinterpret_cast<BYTE*>(this->compressed->hptr + this->dataseg.get_offset(cusz::SEG::REVBOOK));

        // max possible size instead of the fixed size, TODO check again
        cudaMalloc(&xhuff.in.dptr, sizeof(H) * this->header->quant_len / 2);
        (xhuff.in)  //
            .set_len(this->header->huffman_num_uints)
            .template shallow_copy<kHOST>(_h_data)
            .host2device();
        (xhuff.meta)
            .set_len(nchunk * 2)  //
            .template shallow_copy<kHOST>(_h_meta)
            .template alloc<kDEVICE>()
            .host2device();
        (xhuff.revbook)  //
            .set_len(Codec::get_revbook_nbyte(this->header->dict_size))
            .template shallow_copy<kHOST>(_h_rev)
            .template alloc<kDEVICE>()
            .host2device();
    }

    LOGGING(LOG_INFO, "decompressing...");
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR& DPCOMPRESSOR::compress(bool optional_release_input)
{
    auto& nnz = this->ctx->nnz_outlier;

    this->quant.set_len(predictor->get_quant_len()).template alloc<kDEVICE>();
    this->freq.set_len(this->dict_size).template alloc<kDEVICE>();
    book.set_len(this->dict_size).template alloc<kDEVICE>();
    revbook.set_len(Codec::get_revbook_nbyte(this->dict_size)).template alloc<kHOST_DEVICE>();

    {
        cudaStream_t stream_predictor;
        cudaStreamCreate(&stream_predictor);
        predictor->construct(
            this->original->dptr, nullptr, this->quant.dptr, this->ctx->eb, this->ctx->radius, stream_predictor);
        cudaStreamDestroy(stream_predictor);
    }

    {
        cudaStream_t stream_spreducer;
        CHECK_CUDA(cudaStreamCreate(&stream_spreducer));

        spreducer->gather(
            this->original->dptr,                                          // in data
            BINDING::template get_uncompressed_len(predictor, spreducer),  //
            ext_rowptr.dptr,                                               // space 1
            ext_colidx.dptr,                                               // space 2
            ext_values.dptr,                                               // space 3
            nnz,                                                           // out 1
            sp_dump_nbyte,                                                 // out 2
            stream_spreducer);

        spreducer->template consolidate<kDEVICE, kHOST>(sp_use.hptr);
        if (stream_spreducer) cudaStreamDestroy(stream_spreducer);
    }

    this->time.lossy    = predictor->get_time_elapsed();
    this->time.sparsity = spreducer->get_time_elapsed();

    this->dataseg.nbyte.at(cusz::SEG::SPFMT) = sp_dump_nbyte;  // do before consolidate

    // TODO runtime memory config
    LOGGING(LOG_INFO, "#outlier = ", nnz, StringHelper::nnz_percentage(nnz, this->ctx->data_len));

    try_skip_huffman();

    // release original; subject to change
    if (optional_release_input) this->original->template free<kDEVICE>();

    auto const chunk_size = this->ctx->huffman_chunksize;
    auto const nchunk     = this->ctx->nchunk;

    auto& num_bits  = this->ctx->huffman_num_bits;
    auto& num_uints = this->ctx->huffman_num_uints;

    {
        cudaStream_t stream_codec;
        CHECK_CUDA(cudaStreamCreate(&stream_codec));

        auto in_len = BINDING::template get_uncompressed_len(predictor, codec);

        codec->encode_integrated(
            /* space  */ this->freq.dptr, book.dptr, this->huff_workspace,
            /* input  */ this->quant.dptr, in_len,
            /* config */ this->dict_size, chunk_size,
            /* output */ revbook.dptr, huff_data, huff_counts, num_bits, num_uints,
            /* stream */ stream_codec);

        if (stream_codec) cudaStreamDestroy(stream_codec);
    }

    this->time.hist     = codec->get_time_hist();
    this->time.book     = codec->get_time_book();
    this->time.lossless = codec->get_time_lossless();

    revbook.device2host();  // need processing on CPU
    this->dataseg.nbyte.at(cusz::SEG::REVBOOK) = Codec::get_revbook_nbyte(this->dict_size);

    huff_data.device2host();
    this->dataseg.nbyte.at(cusz::SEG::HUFF_META) = sizeof(M) * (2 * nchunk);
    this->dataseg.nbyte.at(cusz::SEG::HUFF_DATA) = sizeof(H) * num_uints;

    this->noncritical__optional__report_compress_time();
    this->pack_metadata();

    return *this;
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR& DPCOMPRESSOR::decompress(Capsule<T>* decomp_space)
{
    this->quant.set_len(BINDING::template get_uncompressed_len(predictor, codec)).template alloc<kDEVICE>();
    auto xdata = decomp_space->dptr, outlier = decomp_space->dptr;

    // TODO pass dump and this->dataseg description
    // problem statement:
    // Data are described in two ways:
    // 1) fields of singleton, which are found&accessed by offset, or
    // 2) scattered, which are access f&a by addr (in absolute value)
    // Therefore, codec::decode() should be
    // decode(WHERE, FROM_DUMP, dump, offset, output)

    auto dump = this->compressed->hptr;

    auto _h_data = reinterpret_cast<H*>(this->compressed->hptr + this->dataseg.get_offset(cusz::SEG::HUFF_DATA));
    auto _h_meta = reinterpret_cast<M*>(this->compressed->hptr + this->dataseg.get_offset(cusz::SEG::HUFF_META));
    auto _h_rev  = reinterpret_cast<BYTE*>(this->compressed->hptr + this->dataseg.get_offset(cusz::SEG::REVBOOK));

    auto nchunk = ConfigHelper::get_npart(
        BINDING::template get_uncompressed_len(predictor, codec), this->header->huffman_chunksize);

    {
        cudaStream_t stream_codec;
        CHECK_CUDA(cudaStreamCreate(&stream_codec));

        auto uncompressed_len = BINDING::template get_uncompressed_len(predictor, codec);

        codec->decode(
            /* in  */ xhuff.in.dptr, xhuff.meta.dptr, xhuff.revbook.dptr, uncompressed_len,
            /* cfg */ this->header->dict_size, this->header->huffman_chunksize,
            /* out */ this->quant.dptr,
            /* stream */ stream_codec);

        if (stream_codec) cudaStreamDestroy(stream_codec);
    }

    {
        cudaStream_t stream_spreducer;
        CHECK_CUDA(cudaStreamCreate(&stream_spreducer));

        spreducer->scatter(
            sp_use.dptr,                                                   //
            this->ctx->nnz_outlier,                                        //
            outlier,                                                       //
            BINDING::template get_uncompressed_len(predictor, spreducer),  //
            stream_spreducer);

        if (stream_spreducer) cudaStreamDestroy(stream_spreducer);
    }

    {
        cudaStream_t stream_predictor;
        CHECK_CUDA(cudaStreamCreate(&stream_predictor));
        predictor->reconstruct(nullptr, this->quant.dptr, xdata, this->ctx->eb, this->ctx->radius, stream_predictor);
        if (stream_predictor) cudaStreamDestroy(stream_predictor);
    }

    return *this;
}

DPCOMPRESSOR_TYPE
DPCOMPRESSOR& DPCOMPRESSOR::backmatter(Capsule<T>* decomp_space)
{
    decomp_space->device2host();

    this->time.lossless = codec->get_time_elapsed();
    this->time.sparsity = spreducer->get_time_elapsed();
    this->time.lossy    = predictor->get_time_elapsed();
    this->noncritical__optional__report_decompress_time();

    this->noncritical__optional__compare_with_original(decomp_space->hptr, this->ctx->on_off.use_gpu_verify);
    this->noncritical__optional__write2disk(decomp_space->hptr);

    return *this;
}

DPCOMPRESSOR_TYPE
template <cusz::LOC FROM, cusz::LOC TO>
DPCOMPRESSOR& DPCOMPRESSOR::consolidate(BYTE** dump_ptr)
{
    constexpr auto        DIRECTION = CopyDirection<FROM, TO>::direction;
    std::vector<uint32_t> offsets   = {0};

    auto REINTERP = [](auto* ptr) { return reinterpret_cast<BYTE*>(ptr); };
    auto ADDR     = [&](int seg_id) { return *dump_ptr + offsets.at(seg_id); };
    auto COPY     = [&](cusz::SEG seg, auto src) {
        auto dst      = ADDR(this->dataseg.name2order.at(seg));
        auto src_byte = REINTERP(src);
        auto len      = this->dataseg.nbyte.at(seg);
        if (len != 0) cudaMemcpy(dst, src_byte, len, DIRECTION);
    };

    DatasegHelper::compress_time_conslidate_report(this->dataseg, offsets);
    auto total_nbyte = offsets.back();
    printf("\ncompression ratio:\t%.4f\n", this->ctx->data_len * sizeof(T) * 1.0 / total_nbyte);

    if CONSTEXPR (TO == kHOST)
        cudaMallocHost(dump_ptr, total_nbyte);
    else if (TO == kDEVICE)
        cudaMalloc(dump_ptr, total_nbyte);
    else
        throw std::runtime_error("[COMPRESSOR::consolidate] undefined behavior");

    COPY(cusz::SEG::HEADER, this->header);
    COPY(cusz::SEG::ANCHOR, this->anchor.template get<FROM>());
    COPY(cusz::SEG::REVBOOK, revbook.template get<FROM>());
    COPY(cusz::SEG::SPFMT, sp_use.template get<FROM>());
    COPY(cusz::SEG::HUFF_META, huff_counts.template get<FROM>() + this->ctx->nchunk);
    COPY(cusz::SEG::HUFF_DATA, huff_data.template get<FROM>());

    return *this;
}

#define DPC_DC DefaultPathCompressor<DefaultPath::DefaultBinding>

template class DPC_DC;

template DPC_DC& DPC_DC::consolidate<kHOST, kHOST>(BYTE**);
template DPC_DC& DPC_DC::consolidate<kHOST, kDEVICE>(BYTE**);
template DPC_DC& DPC_DC::consolidate<kDEVICE, kHOST>(BYTE**);
template DPC_DC& DPC_DC::consolidate<kDEVICE, kDEVICE>(BYTE**);

#define DPC_FC DefaultPathCompressor<DefaultPath::FallbackBinding>

template class DPC_FC;

template DPC_FC& DPC_FC::consolidate<kHOST, kHOST>(BYTE**);
template DPC_FC& DPC_FC::consolidate<kHOST, kDEVICE>(BYTE**);
template DPC_FC& DPC_FC::consolidate<kDEVICE, kHOST>(BYTE**);
template DPC_FC& DPC_FC::consolidate<kDEVICE, kDEVICE>(BYTE**);
