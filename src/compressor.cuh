/**
 * @file compressor.cuh
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

#include "base_compressor.cuh"
#include "binding.hh"
#include "componments.hh"
#include "header.hh"

#define DEFINE_DEV(VAR, TYPE) TYPE* d_##VAR{nullptr};
#define DEFINE_HOST(VAR, TYPE) TYPE* h_##VAR{nullptr};
#define FREEDEV(VAR) CHECK_CUDA(cudaFree(d_##VAR));
#define FREEHOST(VAR) CHECK_CUDA(cudaFreeHost(h_##VAR));

#define D2D_CPY(VAR, FIELD)                                                                            \
    {                                                                                                  \
        auto dst = d_reserved_compressed + header.entry[HEADER::FIELD];                                \
        auto src = reinterpret_cast<BYTE*>(d_##VAR);                                                   \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, nbyte[HEADER::FIELD], cudaMemcpyDeviceToDevice, stream)); \
    }

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_compressed + header->entry[HEADER::SYM])

namespace cusz {

constexpr auto kHOST        = cusz::LOC::HOST;
constexpr auto kDEVICE      = cusz::LOC::DEVICE;
constexpr auto kHOST_DEVICE = cusz::LOC::HOST_DEVICE;

template <class BINDING>
class Compressor : public BaseCompressor<typename BINDING::PREDICTOR> {
   public:
    using Predictor     = typename BINDING::PREDICTOR;
    using SpCodec       = typename BINDING::SPCODEC;
    using Codec         = typename BINDING::CODEC;
    using FallbackCodec = typename BINDING::FALLBACK_CODEC;

    using BYTE = uint8_t;
    using T    = typename Predictor::Origin;
    using FP   = typename Predictor::Precision;
    using E    = typename Predictor::ErrCtrl;
    using H    = typename Codec::Encoded;
    using M    = typename Codec::MetadataT;
    using H_FB = typename FallbackCodec::Encoded;

    using TimeRecord   = std::vector<std::tuple<const char*, double>>;
    using timerecord_t = TimeRecord*;

   private:
    bool use_fallback_codec{false};
    bool fallback_codec_allocated{false};

    using HEADER = cuszHEADER;
    HEADER header;

    BYTE* d_reserved_compressed{nullptr};

    TimeRecord timerecord;

   private:
    Predictor*     predictor;
    SpCodec*       spcodec;
    Codec*         codec;
    FallbackCodec* fb_codec;

   private:
    dim3     data_len3;
    uint32_t get_len_data() { return data_len3.x * data_len3.y * data_len3.z; }

   public:
    // void export_header(HEADER*& ext_header) { ext_header = &header; }
    void export_header(HEADER& ext_header) { ext_header = header; }

    Compressor()
    {
        predictor = new Predictor;
        spcodec   = new SpCodec;
        codec     = new Codec;
        fb_codec  = new FallbackCodec;
    }

    void destroy()
    {
        if (spcodec) delete spcodec;
        if (codec) delete codec;
        if (fb_codec) delete codec;
        if (predictor) delete predictor;
    }

    ~Compressor() { destroy(); }

    /**
     * @brief Export internal Time Record list by deep copy.
     *
     * @param ext_timerecord nullable; pointer to external TimeRecord.
     */
    void export_timerecord(TimeRecord* ext_timerecord)
    {
        if (ext_timerecord) *ext_timerecord = timerecord;
    }

    template <class CONFIG>
    void init(CONFIG* config, bool dbg_print = false)
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

        auto allocate_codec = [&]() {
            if (codec_config == 0b00) throw std::runtime_error("Argument codec_config must have set bit(s).");
            if (codec_config bitand 0b01) {
                if (dbg_print) LOGGING(LOG_INFO, "allocated 4-byte codec");
                (*codec).init(codec_in_len, cfg_max_booklen, cfg_pardeg, dbg_print);
            }
            if (codec_config bitand 0b10) {
                if (dbg_print) LOGGING(LOG_INFO, "allocated 8-byte (fallback) codec");
                (*fb_codec).init(codec_in_len, cfg_max_booklen, cfg_pardeg, dbg_print);
                fallback_codec_allocated = true;
            }
        };

        (*predictor).init(x, y, z, dbg_print);

        spcodec_in_len = (*predictor).get_alloclen_data();
        codec_in_len   = (*predictor).get_alloclen_quant();

        (*spcodec).init(spcodec_in_len, density_factor, dbg_print);

        allocate_codec();

        CHECK_CUDA(cudaMalloc(&d_reserved_compressed, (*predictor).get_alloclen_data() * sizeof(T) / 2));
    }

    void collect_compress_timerecord()
    {
#define COLLECT_TIME(NAME, TIME) timerecord.push_back({const_cast<const char*>(NAME), TIME});

        if (not timerecord.empty()) timerecord.clear();

        COLLECT_TIME("predict", (*predictor).get_time_elapsed());

        if (not use_fallback_codec) {
            COLLECT_TIME("histogram", (*codec).get_time_hist());
            COLLECT_TIME("book", (*codec).get_time_book());
            COLLECT_TIME("huff-enc", (*codec).get_time_lossless());
        }
        else {
            COLLECT_TIME("histogram", (*fb_codec).get_time_hist());
            COLLECT_TIME("book", (*fb_codec).get_time_book());
            COLLECT_TIME("huff-enc", (*fb_codec).get_time_lossless());
        }

        COLLECT_TIME("outlier", (*spcodec).get_time_elapsed());
    }

    void collect_decompress_timerecord()
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

    template <class CONFIG>
    void compress(
        CONFIG*      config,
        T*           uncompressed,
        BYTE*&       compressed,
        size_t&      compressed_len,
        cudaStream_t stream    = nullptr,
        bool         dbg_print = false)
    {
        auto const eb                = (*config).eb;
        auto const radius            = (*config).radius;
        auto const pardeg            = (*config).vle_pardeg;
        auto const codecs_in_use     = (*config).codecs_in_use;
        auto const nz_density_factor = (*config).nz_density_factor;

        data_len3 = dim3((*config).x, (*config).y, (*config).z);

        compress_detail(
            uncompressed, eb, radius, pardeg, codecs_in_use, nz_density_factor, compressed, compressed_len,
            (*config).codec_force_fallback(), stream, dbg_print);
    }

    void compress_detail(
        T*             uncompressed,
        double const   eb,
        int const      radius,
        int const      pardeg,
        uint32_t const codecs_in_use,
        int const      nz_density_factor,
        BYTE*&         compressed,
        size_t&        compressed_len,
        bool           codec_force_fallback,
        cudaStream_t   stream    = nullptr,
        bool           dbg_print = false)
    {
        header.codecs_in_use     = codecs_in_use;
        header.nz_density_factor = nz_density_factor;

        T*     d_anchor{nullptr};   // predictor out1
        E*     d_errctrl{nullptr};  // predictor out2
        BYTE*  d_spfmt{nullptr};
        size_t spfmt_out_len{0};

        BYTE*  d_codec_out{nullptr};
        size_t codec_out_len{0};

        size_t data_len, m, errctrl_len, sublen;
        // must precede the following derived lengths
        auto predictor_do = [&]() {
            (*predictor).construct(data_len3, uncompressed, d_anchor, d_errctrl, eb, radius, stream);
        };

        auto spcodec_do = [&]() { (*spcodec).encode(uncompressed, m * m, d_spfmt, spfmt_out_len, stream, dbg_print); };

        auto codec_do_with_exception = [&]() {
            auto encode_with_fallback_codec = [&]() {
                use_fallback_codec = true;
                if (not fallback_codec_allocated) {
                    if (dbg_print) LOGGING(LOG_EXCEPTION, "online allocate fallback (8-byte) codec");

                    (*fb_codec).init(errctrl_len, radius * 2, pardeg, /*dbg print*/ false);
                    fallback_codec_allocated = true;
                }
                (*fb_codec).encode(
                    d_errctrl, errctrl_len, radius * 2, sublen, pardeg, d_codec_out, codec_out_len, stream);
            };

            if (not codec_force_fallback) {
                try {
                    (*codec).encode(
                        d_errctrl, errctrl_len, radius * 2, sublen, pardeg, d_codec_out, codec_out_len, stream);
                }
                catch (const std::runtime_error& e) {
                    if (dbg_print) LOGGING(LOG_EXCEPTION, "switch to fallback codec");
                    encode_with_fallback_codec();
                }
            }
            else {
                if (dbg_print) LOGGING(LOG_INFO, "force switch to fallback codec");

                encode_with_fallback_codec();
            }
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

        auto subfile_collect = [&]() {
            header.header_nbyte = sizeof(HEADER);
            uint32_t nbyte[HEADER::END];
            nbyte[HEADER::HEADER] = 128;
            nbyte[HEADER::ANCHOR] = sizeof(T) * (*predictor).get_len_anchor();
            nbyte[HEADER::VLE]    = sizeof(BYTE) * codec_out_len;
            nbyte[HEADER::SPFMT]  = sizeof(BYTE) * spfmt_out_len;

            header.entry[0] = 0;
            // *.END + 1; need to know the ending position
            for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] = nbyte[i - 1]; }
            for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] += header.entry[i - 1]; }

            auto debug_header_entry = [&]() {
                printf("\nsubfile collect in compressor:\n");
                printf("  ENTRIES\n");

#define PRINT_ENTRY(VAR) printf("%d %-*s:  %'10u\n", (int)HEADER::VAR, 14, #VAR, header.entry[HEADER::VAR]);
                PRINT_ENTRY(HEADER);
                PRINT_ENTRY(ANCHOR);
                PRINT_ENTRY(VLE);
                PRINT_ENTRY(SPFMT);
                PRINT_ENTRY(END);
                printf("\n");
#undef PRINT_ENTRY
            };

            if (dbg_print) debug_header_entry();

            CHECK_CUDA(cudaMemcpyAsync(d_reserved_compressed, &header, sizeof(header), cudaMemcpyHostToDevice, stream));

            D2D_CPY(anchor, ANCHOR)
            D2D_CPY(codec_out, VLE)
            D2D_CPY(spfmt, SPFMT)

            /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));
        };

        // execution below
        // ---------------

        predictor_do();

        data_len    = (*predictor).get_len_data();
        m           = Reinterpret1DTo2D::get_square_size(data_len);
        errctrl_len = (*predictor).get_len_quant();
        sublen      = ConfigHelper::get_npart(data_len, pardeg);

        spcodec_do(), codec_do_with_exception();

        /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

        update_header(), subfile_collect();
        // output
        compressed_len = header.get_filesize();
        compressed     = d_reserved_compressed;

        collect_compress_timerecord();

        // considering that codec can be consecutively in use, and can compress data of different huff-byte
        use_fallback_codec = false;
    }

    void clear_buffer()
    {  //
        (*predictor).clear_buffer();
        (*codec).clear_buffer();
        (*spcodec).clear_buffer();
    }

    /**
     * @brief High-level decompress method for this compressor
     *
     * @param header header on host; if null, copy from device binary (from the beginning)
     * @param in_compressed device pointer, the cusz archive bianry
     * @param out_decompressed device pointer, output decompressed data
     * @param stream CUDA stream
     * @param rpt_print control over printing time
     */
    void decompress(
        cuszHEADER*  header,
        BYTE*        in_compressed,
        T*           out_decompressed,
        cudaStream_t stream    = nullptr,
        bool         dbg_print = true)
    {
        // TODO host having copy of header when compressing
        if (not header) {
            header = new HEADER;
            CHECK_CUDA(cudaMemcpyAsync(header, in_compressed, sizeof(HEADER), cudaMemcpyDeviceToHost, stream));
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

        auto spcodec_do              = [&]() { (*spcodec).decode(d_sp, d_outlier, stream); };
        auto codec_do_with_exception = [&]() {
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
            (*predictor).reconstruct(data_len3, d_outlier_xdata, d_anchor, d_errctrl, eb, radius, stream);
        };

        // process
        spcodec_do(), codec_do_with_exception(), predictor_do();

        collect_decompress_timerecord();

        // clear state for the next decompression after reporting
        use_fallback_codec = false;
    }
};

template <typename InputData = float>
struct Framework {
    using DATA    = InputData;                          // depend on template input
    using ERRCTRL = ErrCtrlTrait<2>::type;              // predefined
    using FP      = FastLowPrecisionTrait<true>::type;  // predefined

    /* Predictor */
    using PredictorLorenzo = cusz::PredictorLorenzo<DATA, ERRCTRL, FP>;
    using PredictorSpline3 = cusz::Spline3<DATA, ERRCTRL, FP>;

    /* Lossless SpCodec */
    using SpCodecCSR = cusz::CSR11<DATA>;

    /* Lossless Codec*/
    using CodecHuffman32 = cusz::HuffmanCoarse<ERRCTRL, HuffTrait<4>::type, MetadataTrait<4>::type>;
    using CodecHuffman64 = cusz::HuffmanCoarse<ERRCTRL, HuffTrait<8>::type, MetadataTrait<4>::type>;

    /* Predefined Combination */
    using LorenzoFeatured = CompressorTemplate<PredictorLorenzo, SpCodecCSR, CodecHuffman32, CodecHuffman64>;
    using Spline3Featured = CompressorTemplate<PredictorSpline3, SpCodecCSR, CodecHuffman32, CodecHuffman64>;

    /* Usable Compressor */
    using DefaultCompressor         = class Compressor<LorenzoFeatured>;
    using LorenzoFeaturedCompressor = class Compressor<LorenzoFeatured>;
    using Spline3FeaturedCompressor = class Compressor<Spline3Featured>; /* in progress */
};

}  // namespace cusz

#undef FREEDEV
#undef FREEHOST
#undef DEFINE_DEV
#undef DEFINE_HOST
#undef D2D_CPY
#undef ACCESSOR
#undef COLLECT_TIME

#endif
