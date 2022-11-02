/**
 * @file compressor_impl.cuh
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

#ifndef CUSZ_DEFAULT_PATH_CUH
#define CUSZ_DEFAULT_PATH_CUH

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <iostream>

#include "component.hh"
#include "compressor.hh"
#include "header.h"
#include "kernel/cpplaunch_cuda.hh"
#include "stat/stat_g.hh"
#include "utils/cuda_err.cuh"

#define DEFINE_DEV(VAR, TYPE) TYPE* d_##VAR{nullptr};
#define DEFINE_HOST(VAR, TYPE) TYPE* h_##VAR{nullptr};
#define FREEDEV(VAR) CHECK_CUDA(cudaFree(d_##VAR));
#define FREEHOST(VAR) CHECK_CUDA(cudaFreeHost(h_##VAR));

#define PRINT_ENTRY(VAR) printf("%d %-*s:  %'10u\n", (int)Header::VAR, 14, #VAR, header.entry[Header::VAR]);

#define DEVICE2DEVICE_COPY(VAR, FIELD)                                                                 \
    if (nbyte[Header::FIELD] != 0 and VAR != nullptr) {                                                \
        auto dst = d_reserved_compressed + header.entry[Header::FIELD];                                \
        auto src = reinterpret_cast<BYTE*>(VAR);                                                       \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, nbyte[Header::FIELD], cudaMemcpyDeviceToDevice, stream)); \
    }

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_compressed + header->entry[Header::SYM])

namespace cusz {

constexpr auto kHOST        = cusz::LOC::HOST;
constexpr auto kDEVICE      = cusz::LOC::DEVICE;
constexpr auto kHOST_DEVICE = cusz::LOC::HOST_DEVICE;

#define TEMPLATE_TYPE template <class BINDING>
#define IMPL Compressor<BINDING>::impl

TEMPLATE_TYPE
uint32_t IMPL::get_len_data() { return data_len3.x * data_len3.y * data_len3.z; }

TEMPLATE_TYPE
IMPL::impl()
{
    predictor = new Predictor;

    spcodec  = new Spcodec;
    codec    = new Codec;
    fb_codec = new FallbackCodec;
}

TEMPLATE_TYPE
void IMPL::destroy()
{
    if (spcodec) delete spcodec;
    if (codec) delete codec;
    if (fb_codec) delete codec;
    if (predictor) delete predictor;
}

TEMPLATE_TYPE
IMPL::~impl() { destroy(); }

//------------------------------------------------------------------------------

// TODO
TEMPLATE_TYPE
void IMPL::init(Context* config, bool dbg_print) { init_detail(config, dbg_print); }

TEMPLATE_TYPE
void IMPL::init(Header* config, bool dbg_print) { init_detail(config, dbg_print); }

template <class T>
void peek_devdata(T* d_arr, size_t num = 20)
{
    thrust::for_each(thrust::device, d_arr, d_arr + num, [=] __device__ __host__(const T i) { printf("%u\t", i); });
    printf("\n");
}

TEMPLATE_TYPE
void IMPL::compress(
    Context*     config,
    T*           uncompressed,
    BYTE*&       compressed,
    size_t&      compressed_len,
    cudaStream_t stream,
    bool         dbg_print)
{
    auto const eb                = (*config).eb;
    auto const radius            = (*config).radius;
    auto const pardeg            = (*config).vle_pardeg;
    auto const codecs_in_use     = (*config).codecs_in_use;
    auto const nz_density_factor = (*config).nz_density_factor;

    if (dbg_print) {
        std::cout << "eb\t" << eb << endl;
        std::cout << "radius\t" << radius << endl;
        std::cout << "pardeg\t" << pardeg << endl;
        std::cout << "codecs_in_use\t" << codecs_in_use << endl;
        std::cout << "nz_density_factor\t" << nz_density_factor << endl;
    }

    data_len3                 = dim3((*config).x, (*config).y, (*config).z);
    auto codec_force_fallback = (*config).codec_force_fallback();

    header.codecs_in_use     = codecs_in_use;
    header.nz_density_factor = nz_density_factor;

    T*     d_anchor{nullptr};   // predictor out1
    E*     d_errctrl{nullptr};  // predictor out2
    BYTE*  d_spfmt{nullptr};
    size_t spfmt_outlen{0};

    BYTE*  d_codec_out{nullptr};
    size_t codec_outlen{0};

    size_t data_len, errctrl_len, sublen, spcodec_inlen;
    auto   booklen = radius * 2;

    auto derive_lengths_after_prediction = [&]() {
        data_len    = (*predictor).get_len_data();
        errctrl_len = (*predictor).get_len_quant();

        // data_len    = (*prediction).get_len_data();
        // errctrl_len = (*prediction).get_len_quant();

        auto m        = Reinterpret1DTo2D::get_square_size(data_len);
        spcodec_inlen = m * m;
        sublen        = ConfigHelper::get_npart(data_len, pardeg);

        // std::cout << "datalen\t" << data_len << '\n';
        // std::cout << "errctrl_len\t" << errctrl_len << '\n';
        // std::cout << "spcodec_inlen\t" << spcodec_inlen << '\n';
        // std::cout << "sublen\t" << sublen << '\n';
    };

    auto update_header = [&]() {
        header.x          = data_len3.x;
        header.y          = data_len3.y;
        header.z          = data_len3.z;
        header.radius     = radius;
        header.vle_pardeg = pardeg;
        header.eb         = eb;
        header.byte_vle   = use_fallback_codec ? 8 : 4;
    };

    /******************************************************************************/

    // Prediction is the dependency of the rest procedures.
    predictor->construct(LorenzoI, data_len3, uncompressed, &d_anchor, &d_errctrl, eb, radius, stream);
    // peek_devdata(d_errctrl);

    derive_lengths_after_prediction();
    /******************************************************************************/

    asz::stat::histogram<E>(d_errctrl, errctrl_len, d_freq, booklen, &time_hist, stream);

    /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

    // TODO remove duplicate get_frequency inside encode_with_exception()
    encode_with_exception(
        d_errctrl, errctrl_len,                                 // input
        d_freq, booklen, sublen, pardeg, codec_force_fallback,  // config
        d_codec_out, codec_outlen,                              // output
        stream, dbg_print);

    (*spcodec).encode(uncompressed, spcodec_inlen, d_spfmt, spfmt_outlen, stream, dbg_print);

    /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

    /******************************************************************************/

    update_header();
    subfile_collect(
        d_anchor, (*predictor).get_len_anchor(),  //
        d_codec_out, codec_outlen,                //
        d_spfmt, spfmt_outlen,                    //
        stream, dbg_print);

    // output
    compressed_len = ConfigHelper::get_filesize(&header);
    compressed     = d_reserved_compressed;

    collect_compress_timerecord();

    // considering that codec can be consecutively in use, and can compress data of different huff-byte
    use_fallback_codec = false;
}

TEMPLATE_TYPE
void IMPL::clear_buffer()
{  //
    (*predictor).clear_buffer();
    (*codec).clear_buffer();
    (*spcodec).clear_buffer();
}

TEMPLATE_TYPE
void IMPL::decompress(Header* header, BYTE* in_compressed, T* out_decompressed, cudaStream_t stream, bool dbg_print)
{
    // TODO host having copy of header when compressing
    if (not header) {
        header = new Header;
        CHECK_CUDA(cudaMemcpyAsync(header, in_compressed, sizeof(Header), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    data_len3 = dim3(header->x, header->y, header->z);

    use_fallback_codec      = header->byte_vle == 8;
    double const eb         = header->eb;
    int const    radius     = header->radius;
    auto const   vle_pardeg = header->vle_pardeg;

    // The inputs of components are from `compressed`.
    auto d_anchor = ACCESSOR(ANCHOR, T);
    auto d_vle    = ACCESSOR(VLE, BYTE);
    auto d_sp     = ACCESSOR(SPFMT, BYTE);

    // wire the workspace
    auto d_errctrl = (*predictor).expose_quant();  // reuse space

    // wire and aliasing
    auto d_outlier       = out_decompressed;
    auto d_outlier_xdata = out_decompressed;

    auto spcodec_do            = [&]() { (*spcodec).decode(d_sp, d_outlier, stream); };
    auto decode_with_exception = [&]() {
        if (not use_fallback_codec) {  //
            (*codec).decode(d_vle, d_errctrl);
        }
        else {
            if (not fallback_codec_allocated) {
                (*fb_codec).init((*predictor).get_len_quant(), radius * 2, vle_pardeg, /*dbg print*/ false);
                fallback_codec_allocated = true;
            }
            (*fb_codec).decode(d_vle, d_errctrl);
        }
    };
    auto predictor_do = [&]() {
        (*predictor).reconstruct(LorenzoI, data_len3, d_outlier_xdata, d_anchor, d_errctrl, eb, radius, stream);
    };

    // process
    spcodec_do(), decode_with_exception(), predictor_do();

    collect_decompress_timerecord();

    // clear state for the next decompression after reporting
    use_fallback_codec = false;
}

// public getter
TEMPLATE_TYPE
void IMPL::export_header(Header& ext_header) { ext_header = header; }

TEMPLATE_TYPE
void IMPL::export_header(Header* ext_header) { *ext_header = header; }

TEMPLATE_TYPE
void IMPL::export_timerecord(TimeRecord* ext_timerecord)
{
    if (ext_timerecord) *ext_timerecord = timerecord;
}

// helper
TEMPLATE_TYPE
void IMPL::init_codec(size_t codec_in_len, unsigned int codec_config, int max_booklen, int pardeg, bool dbg_print)
{
    if (codec_config == 0b00) throw std::runtime_error("Argument codec_config must have set bit(s).");
    if (codec_config bitand 0b01) {
        if (dbg_print) LOGGING(LOG_INFO, "allocated 4-byte codec");
        (*codec).init(codec_in_len, max_booklen, pardeg, dbg_print);
    }
    if (codec_config bitand 0b10) {
        if (dbg_print) LOGGING(LOG_INFO, "allocated 8-byte (fallback) codec");
        (*fb_codec).init(codec_in_len, max_booklen, pardeg, dbg_print);
        fallback_codec_allocated = true;
    }
};

TEMPLATE_TYPE
template <class CONFIG>
void IMPL::init_detail(CONFIG* config, bool dbg_print)
{
    const auto cfg_radius      = (*config).radius;
    const auto cfg_pardeg      = (*config).vle_pardeg;
    const auto density_factor  = (*config).nz_density_factor;
    const auto codec_config    = (*config).codecs_in_use;
    const auto cfg_max_booklen = cfg_radius * 2;
    const auto x               = (*config).x;
    const auto y               = (*config).y;
    const auto z               = (*config).z;

    size_t spcodec_in_len, codec_in_len;

    (*predictor).init(LorenzoI, x, y, z, dbg_print);

    spcodec_in_len = (*predictor).get_alloclen_data();
    codec_in_len   = (*predictor).get_alloclen_quant();

    (*spcodec).init(spcodec_in_len, density_factor, dbg_print);

    {
        auto bytes = sizeof(cusz::FREQ) * cfg_max_booklen;
        cudaMalloc(&d_freq, bytes);
        cudaMemset(d_freq, 0x0, bytes);

        // cudaMalloc(&d_freq_another, bytes);
        // cudaMemset(d_freq_another, 0x0, bytes);
    }

    init_codec(codec_in_len, codec_config, cfg_max_booklen, cfg_pardeg, dbg_print);

    CHECK_CUDA(cudaMalloc(&d_reserved_compressed, (*predictor).get_alloclen_data() * sizeof(T) / 2));
}

TEMPLATE_TYPE
void IMPL::collect_compress_timerecord()
{
#define COLLECT_TIME(NAME, TIME) timerecord.push_back({const_cast<const char*>(NAME), TIME});

    if (not timerecord.empty()) timerecord.clear();

    COLLECT_TIME("predict", (*predictor).get_time_elapsed());
    COLLECT_TIME("histogram", time_hist);

    if (not use_fallback_codec) {
        COLLECT_TIME("book", (*codec).get_time_book());
        COLLECT_TIME("huff-enc", (*codec).get_time_lossless());
    }
    else {
        COLLECT_TIME("book", (*fb_codec).get_time_book());
        COLLECT_TIME("huff-enc", (*fb_codec).get_time_lossless());
    }

    COLLECT_TIME("outlier", (*spcodec).get_time_elapsed());
}

TEMPLATE_TYPE
void IMPL::collect_decompress_timerecord()
{
    if (not timerecord.empty()) timerecord.clear();

    COLLECT_TIME("outlier", (*spcodec).get_time_elapsed());

    if (not use_fallback_codec) {  //
        COLLECT_TIME("huff-dec", (*codec).get_time_lossless());
    }
    else {  //
        COLLECT_TIME("huff-dec", (*fb_codec).get_time_lossless());
    }

    COLLECT_TIME("predict", (*predictor).get_time_elapsed());
}

TEMPLATE_TYPE
void IMPL::encode_with_exception(
    E*           d_in,
    size_t       inlen,
    cusz::FREQ*  d_freq,
    int          booklen,
    int          sublen,
    int          pardeg,
    bool         codec_force_fallback,
    BYTE*&       d_out,
    size_t&      outlen,
    cudaStream_t stream,
    bool         dbg_print)
{
    auto build_codebook_using = [&](auto encoder) { encoder->build_codebook(d_freq, booklen, stream); };
    auto encode_with          = [&](auto encoder) { encoder->encode(d_in, inlen, d_out, outlen, stream); };

    auto try_fallback_alloc = [&]() {
        use_fallback_codec = true;
        if (not fallback_codec_allocated) {
            LOGGING(LOG_EXCEPTION, "online allocate fallback (8-byte) codec");
            fb_codec->init(inlen, booklen, pardeg, dbg_print);
            fallback_codec_allocated = true;
        }
    };

    /******************************************************************************/
    if (not codec_force_fallback) {
        try {
            build_codebook_using(codec);
            encode_with(codec);
        }
        catch (const std::runtime_error& e) {
            LOGGING(LOG_EXCEPTION, "switch to fallback codec");
            try_fallback_alloc();

            build_codebook_using(fb_codec);
            encode_with(fb_codec);
        }
    }
    else {
        LOGGING(LOG_INFO, "force switch to fallback codec");
        try_fallback_alloc();

        build_codebook_using(fb_codec);
        encode_with(fb_codec);
    }
}

TEMPLATE_TYPE
void IMPL::subfile_collect(
    T*           d_anchor,
    size_t       anchor_len,
    BYTE*        d_codec_out,
    size_t       codec_outlen,
    BYTE*        d_spfmt_out,
    size_t       spfmt_outlen,
    cudaStream_t stream,
    bool         dbg_print)
{
    header.header_nbyte = sizeof(Header);
    uint32_t nbyte[Header::END];
    nbyte[Header::HEADER] = 128;
    nbyte[Header::ANCHOR] = sizeof(T) * anchor_len;
    nbyte[Header::VLE]    = sizeof(BYTE) * codec_outlen;
    nbyte[Header::SPFMT]  = sizeof(BYTE) * spfmt_outlen;

    header.entry[0] = 0;
    // *.END + 1; need to know the ending position
    for (auto i = 1; i < Header::END + 1; i++) { header.entry[i] = nbyte[i - 1]; }
    for (auto i = 1; i < Header::END + 1; i++) { header.entry[i] += header.entry[i - 1]; }

    auto debug_header_entry = [&]() {
        printf("\nsubfile collect in compressor:\n");
        printf("  ENTRIES\n");

        PRINT_ENTRY(HEADER);
        PRINT_ENTRY(ANCHOR);
        PRINT_ENTRY(VLE);
        PRINT_ENTRY(SPFMT);
        PRINT_ENTRY(END);
        printf("\n");
    };

    if (dbg_print) debug_header_entry();

    CHECK_CUDA(cudaMemcpyAsync(d_reserved_compressed, &header, sizeof(header), cudaMemcpyHostToDevice, stream));

    DEVICE2DEVICE_COPY(d_anchor, ANCHOR)
    DEVICE2DEVICE_COPY(d_codec_out, VLE)
    DEVICE2DEVICE_COPY(d_spfmt_out, SPFMT)

    /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));
}

}  // namespace cusz

#undef FREEDEV
#undef FREEHOST
#undef DEFINE_DEV
#undef DEFINE_HOST
#undef DEVICE2DEVICE_COPY
#undef PRINT_ENTRY
#undef ACCESSOR
#undef COLLECT_TIME

#undef TEMPLATE_TYPE
#undef IMPL

#endif
