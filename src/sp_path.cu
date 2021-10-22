/**
 * @file sp_path.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-09-29
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "sp_path.cuh"

#define SPCOMPRESSOR_TYPE template <class BINDING>
#define SPCOMPRESSOR SpPathCompressor<BINDING>

SPCOMPRESSOR_TYPE
SPCOMPRESSOR::SpPathCompressor(cuszCTX* _ctx, Capsule<T>* _in_data)
{
    this->ctx     = _ctx;
    this->in_data = _in_data;
    this->timing  = cuszWHEN::COMPRESS;
    this->header  = new cusz_header();
    this->xyz     = dim3(this->ctx->x, this->ctx->y, this->ctx->z);

    this->prescan();  // internally change eb (regarding value range)
    ConfigHelper::set_eb_series(this->ctx->eb, this->config);

    predictor             = new Predictor(this->xyz, this->ctx->eb, this->ctx->radius, false);
    this->ctx->quant_len  = predictor->get_quant_len();
    this->ctx->anchor_len = predictor->get_anchor_len();

    this->anchor  //
        .set_len(this->ctx->anchor_len)
        .template alloc<cuszLOC::HOST_DEVICE>();

    auto sp_inlen = BINDING::template get_spreducer_input_len<cuszCTX>(this->ctx);
    spreducer     = new SpReducer(sp_inlen);
    sp_use  //
        .set_len(sp_inlen)
        .template alloc<cuszLOC::HOST_DEVICE>();

    LOGGING(LOG_INFO, "compressing...");
}

SPCOMPRESSOR_TYPE
SPCOMPRESSOR::SpPathCompressor(cuszCTX* _ctx, Capsule<BYTE>* _in_dump)
{
    this->ctx     = _ctx;
    this->in_dump = _in_dump;
    this->timing  = cuszWHEN::DECOMPRESS;
    auto dump     = this->in_dump->hptr;

    this->header = reinterpret_cast<cusz_header*>(dump);

    this->unpack_metadata();

    m = Reinterpret1DTo2D::get_square_size(this->ctx->data_len), mxm = m * m;

    this->xyz = dim3(this->header->x, this->header->y, this->header->z);

    // predictor comes first to determine sp-input size
    predictor = new Predictor(this->xyz, this->ctx->eb, this->ctx->radius, false);
    this->anchor  //
        .set_len(this->ctx->anchor_len)
        .template from_existing_on<cuszLOC::HOST>(  //
            reinterpret_cast<E*>(dump + this->dataseg.get_offset(cuszSEG::ANCHOR)))
        .template alloc<cuszLOC::DEVICE>()
        .host2device();

    spreducer = new SpReducer(BINDING::template get_spreducer_input_len<cuszCTX>(this->ctx), this->ctx->nnz_outlier);

    sp_use  //
        .set_len(spreducer->get_total_nbyte())
        .template from_existing_on<cuszLOC::HOST>(  //
            reinterpret_cast<BYTE*>(dump + this->dataseg.get_offset(cuszSEG::OUTLIER)))
        .template alloc<cuszLOC::DEVICE>()
        .host2device();

    LOGGING(LOG_INFO, "decompressing...");
}

SPCOMPRESSOR_TYPE
SPCOMPRESSOR::~SpPathCompressor()
{
    if (this->timing == cuszWHEN::COMPRESS) {  // release small-size arrays

        this->quant.template free<cuszLOC::DEVICE>();
        this->freq.template free<cuszLOC::DEVICE>();
        this->anchor.template free<cuszLOC::HOST_DEVICE>();

        delete this->header;
    }
    else {
        cudaFree(sp_use.dptr);
    }

    delete spreducer;
    delete predictor;
}

SPCOMPRESSOR_TYPE
SPCOMPRESSOR& SPCOMPRESSOR::compress()
{
    // this->dryrun();  // TODO

    this->quant  //
        .set_len(this->ctx->quant_len)
        .template alloc<cuszLOC::DEVICE, ALIGNDATA::SQUARE_MATRIX>();

    predictor->construct(this->in_data->dptr, this->anchor.dptr, this->quant.dptr);
    this->anchor.device2host();

    spreducer->gather(this->in_data->dptr, sp_dump_nbyte, this->ctx->nnz_outlier);
    spreducer->template consolidate<cuszLOC::DEVICE, cuszLOC::HOST>(sp_use.hptr);

    this->time.lossy = predictor->get_time_elapsed();
    cout << this->time.lossy << endl;
    this->time.outlier = spreducer->get_time_elapsed();

    this->dataseg.nbyte.at(cuszSEG::OUTLIER) = sp_dump_nbyte;  // do before consolidate

    LOGGING(
        LOG_INFO, "#outlier = ", this->ctx->nnz_outlier,
        StringHelper::nnz_percentage(this->ctx->nnz_outlier, this->ctx->data_len));

    // release in_data; subject to change
    cudaFree(this->in_data->dptr);

    this->try_report_compress_time();
    this->pack_metadata();

    return *this;
}

SPCOMPRESSOR_TYPE
SPCOMPRESSOR& SPCOMPRESSOR::decompress(Capsule<T>* decomp_space)
{
    this->quant.set_len(this->ctx->quant_len).template alloc<cuszLOC::DEVICE>();
    auto xdata = decomp_space->dptr, outlier = decomp_space->dptr;

    spreducer->scatter(sp_use.dptr, outlier);
    predictor->reconstruct(this->anchor.dptr, this->quant.dptr, xdata);

    return *this;
}

SPCOMPRESSOR_TYPE
SPCOMPRESSOR& SPCOMPRESSOR::backmatter(Capsule<T>* decomp_space)
{
    decomp_space->device2host();

    this->time.outlier = spreducer->get_time_elapsed();
    this->time.lossy   = predictor->get_time_elapsed();
    this->try_report_decompress_time();

    this->try_compare_with_origin(decomp_space->hptr);
    this->try_write2disk(decomp_space->hptr);

    return *this;
}

SPCOMPRESSOR_TYPE
template <cuszLOC FROM, cuszLOC TO>
SPCOMPRESSOR& SPCOMPRESSOR::consolidate(BYTE** dump_ptr)
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
    // COPY(cuszSEG::REVBOOK, revbook.template get<FROM>());
    COPY(cuszSEG::OUTLIER, sp_use.template get<FROM>());
    // COPY(cuszSEG::HUFF_META, huff_counts.template get<FROM>() + this->ctx->nchunk);
    // COPY(cuszSEG::HUFF_DATA, huff_data.template get<FROM>());

    return *this;
}

#define SP_DC SpPathCompressor<SparsityAwarePath::DefaultBinding>

template class SP_DC;

template SP_DC& SP_DC::consolidate<cuszLOC::HOST, cuszLOC::HOST>(BYTE**);
template SP_DC& SP_DC::consolidate<cuszLOC::HOST, cuszLOC::DEVICE>(BYTE**);
template SP_DC& SP_DC::consolidate<cuszLOC::DEVICE, cuszLOC::HOST>(BYTE**);
template SP_DC& SP_DC::consolidate<cuszLOC::DEVICE, cuszLOC::DEVICE>(BYTE**);

#define SP_FC SpPathCompressor<SparsityAwarePath::FallbackBinding>

template class SP_FC;

template SP_FC& SP_FC::consolidate<cuszLOC::HOST, cuszLOC::HOST>(BYTE**);
template SP_FC& SP_FC::consolidate<cuszLOC::HOST, cuszLOC::DEVICE>(BYTE**);
template SP_FC& SP_FC::consolidate<cuszLOC::DEVICE, cuszLOC::HOST>(BYTE**);
template SP_FC& SP_FC::consolidate<cuszLOC::DEVICE, cuszLOC::DEVICE>(BYTE**);
