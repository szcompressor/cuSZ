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
#define SPCOMPRESSOR SpPathCompressorOld<BINDING>

SPCOMPRESSOR_TYPE
SPCOMPRESSOR::SpPathCompressorOld(cuszCTX* _ctx, Capsule<T>* _in_data)
{
    this->ctx     = _ctx;
    this->in_data = _in_data;
    this->timing  = cusz::WHEN::COMPRESS;
    this->header  = new cuszHEADER();
    this->xyz     = dim3(this->ctx->x, this->ctx->y, this->ctx->z);

    this->prescan();  // internally change eb (regarding value range)
    ConfigHelper::set_eb_series(this->ctx->eb, this->config);

    predictor             = new Predictor(this->xyz, this->ctx->eb, 0 /* pseudo radius */, false);
    this->ctx->quant_len  = predictor->get_quant_len();
    this->ctx->anchor_len = predictor->get_anchor_len();

    this->quant  //
        .set_name("quant")
        .set_len(this->ctx->quant_len)
        .template alloc<EXEC_SPACE, cusz::ALIGNDATA::SQUARE_MATRIX>();

    this->anchor  //
        .set_name("anchor")
        .set_len(this->ctx->anchor_len)
        .template alloc<cusz::LOC::HOST_DEVICE>();

    auto sp_inlen = BINDING::template get_spreducer_input_len<cuszCTX>(this->ctx);
    spreducer     = new SpReducer(sp_inlen);
    sp_use  //
        .set_name("sp_use")
        .set_len(sp_inlen)
        .template alloc<cusz::LOC::HOST_DEVICE>();

    LOGGING(LOG_INFO, "compressing...");
}

SPCOMPRESSOR_TYPE
SPCOMPRESSOR& SPCOMPRESSOR::compress()
{
    // this->dryrun();  // TODO

    predictor->construct(this->in_data->dptr, this->anchor.dptr, this->quant.dptr);

    spreducer->gather(this->quant.dptr, sp_dump_nbyte, this->ctx->nnz_outlier);
    sp_use.set_len(sp_dump_nbyte).template alloc<FALLBACK_SPACE>();
    spreducer->template consolidate<EXEC_SPACE, FALLBACK_SPACE>(sp_use.template get<FALLBACK_SPACE>());

    // time
    this->time.lossy    = predictor->get_time_elapsed();
    this->time.sparsity = spreducer->get_time_elapsed();

    // record metadata
    this->dataseg.nbyte.at(cusz::SEG::ANCHOR) = this->anchor.device2host().nbyte();
    this->dataseg.nbyte.at(cusz::SEG::SPFMT)  = sp_dump_nbyte;  // do before consolidate

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
SPCOMPRESSOR::SpPathCompressorOld(cuszCTX* _ctx, Capsule<BYTE>* _in_dump)
{
    this->ctx     = _ctx;
    this->in_dump = _in_dump;
    this->timing  = cusz::WHEN::DECOMPRESS;
    auto dump     = this->in_dump->hptr;

    this->header = reinterpret_cast<cuszHEADER*>(dump);

    this->unpack_metadata();

    m = Reinterpret1DTo2D::get_square_size(this->ctx->data_len), mxm = m * m;

    this->xyz = dim3(this->header->x, this->header->y, this->header->z);

    // predictor comes first to determine sp-input size
    predictor = new Predictor(this->xyz, this->ctx->eb, 0 /* pseudo radius*/, false);
    this->anchor  //
        .set_len(predictor->get_anchor_len())
        .template shallow_copy<cusz::LOC::HOST>(  //
            reinterpret_cast<E*>(dump + this->dataseg.get_offset(cusz::SEG::ANCHOR)))
        .template alloc<cusz::LOC::DEVICE>()
        .host2device();

    spreducer = new SpReducer(BINDING::template get_spreducer_input_len<cuszCTX>(this->ctx), this->ctx->nnz_outlier);

    sp_use  //
        .set_len(spreducer->get_total_nbyte())
        .template shallow_copy<cusz::LOC::HOST>(  //
            reinterpret_cast<BYTE*>(dump + this->dataseg.get_offset(cusz::SEG::SPFMT)))
        .template alloc<cusz::LOC::DEVICE>()
        .host2device();

    LOGGING(LOG_INFO, "decompressing...");
}

SPCOMPRESSOR_TYPE
SPCOMPRESSOR& SPCOMPRESSOR::decompress(Capsule<T>* decomp_space)
{
    this->quant  //
        .set_len(this->ctx->quant_len)
        .template alloc<cusz::LOC::DEVICE>();
    auto xdata = decomp_space;

    spreducer->scatter(sp_use.dptr, this->quant.dptr);
    predictor->reconstruct(this->anchor.dptr, this->quant.dptr, xdata->dptr);

    return *this;
}

SPCOMPRESSOR_TYPE
SPCOMPRESSOR& SPCOMPRESSOR::backmatter(Capsule<T>* decomp_space)
{
    decomp_space->device2host();

    this->time.sparsity = spreducer->get_time_elapsed();
    this->time.lossy    = predictor->get_time_elapsed();
    this->try_report_decompress_time();

    this->try_compare_with_origin(decomp_space->hptr);
    this->try_write2disk(decomp_space->hptr);

    return *this;
}

SPCOMPRESSOR_TYPE
template <cusz::LOC FROM, cusz::LOC TO>
SPCOMPRESSOR& SPCOMPRESSOR::consolidate(BYTE** dump_ptr)
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

    if CONSTEXPR (TO == cusz::LOC::HOST)
        cudaMallocHost(dump_ptr, total_nbyte);
    else if (TO == cusz::LOC::DEVICE)
        cudaMalloc(dump_ptr, total_nbyte);
    else
        throw std::runtime_error("[COMPRESSOR::consolidate] undefined behavior");

    COPY(cusz::SEG::HEADER, this->header);
    COPY(cusz::SEG::ANCHOR, this->anchor.template get<FROM>());
    COPY(cusz::SEG::SPFMT, sp_use.template get<FROM>());

    return *this;
}

SPCOMPRESSOR_TYPE
SPCOMPRESSOR::~SpPathCompressorOld()
{
    if (this->timing == cusz::WHEN::COMPRESS) {  // release small-size arrays

        this->quant.template free<cusz::LOC::DEVICE>();
        this->freq.template free<cusz::LOC::DEVICE>();
        this->anchor.template free<cusz::LOC::HOST_DEVICE>();

        delete this->header;
    }
    else {
        cudaFree(sp_use.dptr);
    }

    delete spreducer;
    delete predictor;
}

#define SP_DC SpPathCompressorOld<SparsityAwarePath::DefaultBinding>

template class SP_DC;

template SP_DC& SP_DC::consolidate<cusz::LOC::HOST, cusz::LOC::HOST>(BYTE**);
template SP_DC& SP_DC::consolidate<cusz::LOC::HOST, cusz::LOC::DEVICE>(BYTE**);
template SP_DC& SP_DC::consolidate<cusz::LOC::DEVICE, cusz::LOC::HOST>(BYTE**);
template SP_DC& SP_DC::consolidate<cusz::LOC::DEVICE, cusz::LOC::DEVICE>(BYTE**);

#define SP_FC SpPathCompressorOld<SparsityAwarePath::FallbackBinding>

template class SP_FC;

template SP_FC& SP_FC::consolidate<cusz::LOC::HOST, cusz::LOC::HOST>(BYTE**);
template SP_FC& SP_FC::consolidate<cusz::LOC::HOST, cusz::LOC::DEVICE>(BYTE**);
template SP_FC& SP_FC::consolidate<cusz::LOC::DEVICE, cusz::LOC::HOST>(BYTE**);
template SP_FC& SP_FC::consolidate<cusz::LOC::DEVICE, cusz::LOC::DEVICE>(BYTE**);
