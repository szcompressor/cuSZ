/**
 * @file v2_compressor_impl.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-01-23
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef F4D645B7_B2E3_41AB_BCFD_DCF919C4C56D
#define F4D645B7_B2E3_41AB_BCFD_DCF919C4C56D

#include <iostream>

#include "component.hh"
#include "header.h"
#include "pipeline/v2_compressor.hh"
// #include "kernel/cpplaunch_cuda.hh"
#include "kernel/v2_lorenzo.hh"
#include "stat/stat_g.hh"
#include "utils/cuda_err.cuh"

#include "../detail/spv_gpu.inl"
#include "../kernel/detail/lorenzo23.inl"

#define TEMPLATE_TYPE template <class CONFIG>
#define IMPL v2_Compressor<CONFIG>::impl

#define ARCHIVE(VAR, FIELD)                                                                                  \
    if (segments[v2_header::FIELD] != 0 and VAR != nullptr) {                                                \
        auto dst = var_archive() + header.entry[v2_header::FIELD];                                           \
        auto src = reinterpret_cast<BYTE*>(VAR);                                                             \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, segments[v2_header::FIELD], cudaMemcpyDeviceToDevice, stream)); \
    }

#define ACCESS_VAR(SYM, TYPE) reinterpret_cast<TYPE*>(in_compressed + header->entry[v2_header::SYM])

namespace psz {

TEMPLATE_TYPE
IMPL::impl()
{
    codec = new Codec;
    // TODO re-enable fallback codec
    // fb_codec  = new FallbackCodec;
}

TEMPLATE_TYPE
void IMPL::destroy()
{
    if (codec) delete codec;
    // if (fb_codec) delete codec;

    // also deallocate buffer
}

TEMPLATE_TYPE
void IMPL::init(Context* config) { __init(config); }

TEMPLATE_TYPE
void IMPL::init(v2_header* config) { __init(config); }

TEMPLATE_TYPE
template <class ContextOrHeader>
void IMPL::__init(ContextOrHeader* c)
{
    static_assert(
        std::is_same<ContextOrHeader, Context>::value or  //
            std::is_same<ContextOrHeader, v2_header>::value,
        "[v2_Compressor::impl::init] not a valid comrpessor config type.");

    auto len = c->x * c->y * c->z;
    // TODO allocate anchor

    // allocate eq
    cudaMalloc(&d_errctrl, len * sizeof(EQ));  // to overlap with one of vle/hf buffers

    // allocate outlier
    outlier.allocate(len / sp_factor, true);

    // allocate vle/hf
    codec->init(len, c->radius * 2, c->vle_pardeg);
    // TODO disable fallback codec for now
}

TEMPLATE_TYPE
void IMPL::compress(
    Context*     c,
    T*           uncompressed,
    BYTE*&       compressed,
    size_t&      compressed_len,
    cudaStream_t stream,
    bool         dbg_print)
{
    auto const eb     = c->eb;
    auto const radius = c->radius;
    auto const pardeg = c->vle_pardeg;

    if (dbg_print) {
        printf("[dbg] eb: %lf\n", eb);
        printf("[dbg] radius: %d\n", radius);
        printf("[dbg] pardeg: %d\n", pardeg);
        // printf("[dbg] codecs_in_use: %d\n", codecs_in_use);
        printf("[dbg] sp_factor: %d\n", sp_factor);
    }

    data_len3 = dim3(c->x, c->y, c->z);
    data_len  = c->x * c->y * c->z;

    header.sp.factor = sp_factor;

    BYTE*  d_codec_out{nullptr};
    size_t codec_outlen{0};

    // size_t sublen;
    auto booklen = radius * 2;

    /******************************************************************************/

    // TODO version clarification
    // with compaction
    v2_compress_predict_lorenzo_i<T, EQ, FP>(
        uncompressed, data_len3, eb, radius, d_errctrl, dim3(1, 1, 1), d_anchor, dim3(1, 1, 1), outlier,
        &comp_time.construct, stream);

    outlier.make_count_host_accessible(stream);

    asz::stat::histogram<E>(d_errctrl, data_len, d_freq, booklen, &comp_time.hist, stream);

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // TODO overlapping memory
    codec->encode(d_errctrl, data_len, d_codec_out, codec_outlen, stream);

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // update header
    {
        header.x = c->x, header.y = c->y, header.z = c->z, header.w = 1;
        header.sp.count = outlier.access_count_on_host();
        // TODO the new
        {
            // header.config.radius = radius, header.config.eb = eb;
            // header.hf.pardeg = pardeg;
        }

        // the compat
        {
            header.radius = radius, header.eb = eb;
            header.vle_pardeg = pardeg;
        }

        // header.byte_vle  = 4;  // regardless of fallback codec
    };

    size_t segments[v2_header::END] = {0};

    // gather archive
    {
        // calculate offsets
        segments[v2_header::HEADER] = sizeof(v2_header);
        segments[v2_header::ANCHOR] = 0;  // placeholder
        segments[v2_header::SP_IDX] = outlier.access_count_on_host() * sizeof(IDX);
        segments[v2_header::SP_VAL] = outlier.access_count_on_host() * sizeof(T);
        segments[v2_header::HF]     = codec_outlen;

        header.entry[0] = 0;
        for (auto i = 1; i < v2_header::END + 1; i++) { header.entry[i] = segments[i - 1]; }
        for (auto i = 1; i < v2_header::END + 1; i++) { header.entry[i] += header.entry[i - 1]; }

        CHECK_CUDA(cudaStreamSynchronize(stream));

        // memcpy
        ARCHIVE(d_anchor, ANCHOR);
        ARCHIVE(outlier.idx, SP_IDX);
        ARCHIVE(outlier.val, SP_VAL);
        ARCHIVE(d_codec_out, HF);

        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // output
    compressed_len = header.entry[v2_header::END];
    compressed     = var_archive();

    // collect_compress_timerecord();
}

TEMPLATE_TYPE
void IMPL::decompress(v2_header* header, BYTE* in_compressed, T* out_decompressed, cudaStream_t stream, bool dbg_print)
{
    // TODO host having copy of header when compressing
    if (not header) {
        header = new v2_header;
        CHECK_CUDA(cudaMemcpyAsync(header, in_compressed, sizeof(v2_header), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    data_len3 = dim3(header->x, header->y, header->z);

    // use_fallback_codec      = header->byte_vle == 8;
    // auto const vle_pardeg = header->hf.pardeg;

    // The inputs of components are from `compressed`.
    // auto d_anchor = ACCESS_VAR(ANCHOR, T);
    auto d_vle   = ACCESS_VAR(HF, BYTE);
    auto d_spidx = ACCESS_VAR(SP_IDX, IDX);
    auto d_spval = ACCESS_VAR(SP_VAL, T);

    // wire and aliasing
    auto d_outlier = out_decompressed;
    auto d_xdata   = out_decompressed;

    psz::detail::spv_scatter<T, IDX>(d_spval, d_spidx, header->sp.count, d_outlier, &decomp_time.scatter, stream);

    codec->decode(d_vle, d_errctrl);

    decompress_predict_lorenzo_i<T, EQ, FP>(
        d_errctrl, data_len3,  //
        d_outlier,             //
        nullptr, 0,            // TODO remove
        header->eb, header->radius,
        d_xdata,  // output
        &decomp_time.reconstruct, stream);

    // collect_decompress_timerecord();

    // clear state for the next decompression after reporting
    // use_fallback_codec = false;
}

}  // namespace psz

#undef TEMPLATE_TYPE
#undef IMPL

#endif /* F4D645B7_B2E3_41AB_BCFD_DCF919C4C56D */
