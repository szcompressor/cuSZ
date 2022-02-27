/**
 * @file default_path.cuh
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
#include "wrapper.hh"
#include "wrapper/spgs.cuh"

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

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_compressed + header.entry[HEADER::SYM])

constexpr auto kHOST        = cusz::LOC::HOST;
constexpr auto kDEVICE      = cusz::LOC::DEVICE;
constexpr auto kHOST_DEVICE = cusz::LOC::HOST_DEVICE;

template <class BINDING>
class DefaultPathCompressor : public BaseCompressor<typename BINDING::PREDICTOR> {
   public:
    using Predictor     = typename BINDING::PREDICTOR;
    using SpReducer     = typename BINDING::SPREDUCER;
    using Codec         = typename BINDING::CODEC;
    using FallbackCodec = typename BINDING::FALLBACK_CODEC;

    using BYTE = uint8_t;
    using T    = typename Predictor::Origin;
    using FP   = typename Predictor::Precision;
    using E    = typename Predictor::ErrCtrl;
    using H    = typename Codec::Encoded;
    using M    = typename Codec::MetadataT;
    using H_FB = typename FallbackCodec::Encoded;

    bool use_fallback_codec{false};
    bool fallback_codec_allocated{false};

    struct header_t {
        static const int HEADER = 0;
        static const int ANCHOR = 1;
        static const int VLE    = 2;
        static const int SPFMT  = 3;
        static const int END    = 4;

        uint32_t header_nbyte : 8;
        uint32_t fp : 1;
        uint32_t byte_uncompressed : 4;  // T; 1, 2, 4, 8
        uint32_t byte_vle : 4;           // 4, 8
        uint32_t byte_errctrl : 3;       // 1, 2, 4
        uint32_t byte_meta : 4;          // 4, 8
        uint32_t vle_pardeg;
        uint32_t x;
        uint32_t y;
        uint32_t z;
        uint32_t w;
        uint32_t ndim : 3;  // 1,2,3,4
        double   eb;
        size_t   data_len;
        size_t   errctrl_len;
        uint32_t radius : 16;

        uint32_t entry[END + 1];

        uint32_t file_size() const { return entry[END]; }
    };
    using HEADER = struct header_t;

    struct runtime_helper {
    };
    using RT = runtime_helper;
    RT rt;

   private:
    BYTE* d_reserved_compressed{nullptr};

   private:
    Predictor*     predictor;
    SpReducer*     spreducer;
    Codec*         codec;
    FallbackCodec* fb_codec;

   private:
    dim3 const data_size;
    uint32_t   get_data_len() { return data_size.x * data_size.y * data_size.z; }

   private:
    // TODO better move to base compressor
    // DefaultPathCompressor& analyze_compressibility();
    // DefaultPathCompressor& internal_eval_try_export_book();
    // DefaultPathCompressor& internal_eval_try_export_quant();
    // DefaultPathCompressor& try_skip_huffman();
    // DefaultPathCompressor& get_freq_codebook();
    // DefaultPathCompressor& old_huffman_encode();

   public:
    DefaultPathCompressor(cuszCTX* _ctx, Capsule<T>* _in_data, uint3 xyz, int dict_size);
    DefaultPathCompressor(cuszCTX* _ctx, Capsule<BYTE>* _in_dump);

    ~DefaultPathCompressor()
    {
        if (spreducer) delete spreducer;
        if (codec) delete codec;
        if (fb_codec) delete codec;
        if (predictor) delete predictor;
    }

    DefaultPathCompressor& compress(bool optional_release_input = false);

    template <cusz::LOC SRC, cusz::LOC DST>
    DefaultPathCompressor& consolidate(BYTE** dump);

    DefaultPathCompressor& decompress(Capsule<T>* out_xdata);
    DefaultPathCompressor& backmatter(Capsule<T>* out_xdata);

    // new
    /**
     * @brief Construct a new Default Path Compressor object
     *
     * @param xyz
     */
    DefaultPathCompressor(dim3 xyz) : data_size(xyz)
    {
        predictor = new Predictor(xyz);
        spreducer = new SpReducer;
        codec     = new Codec;
        fb_codec  = new FallbackCodec;
    }

    /**
     * @brief Allocate workspace accordingly.
     *
     * @param cfg_radius
     * @param cfg_pardeg
     * @param sp_factor
     * @param codec_config
     * @param dbg_print
     */
    void allocate_workspace(
        int  cfg_radius,
        int  cfg_pardeg,
        int  sp_factor    = 4,
        int  codec_config = 0b01,
        bool dbg_print    = false)
    {
        const auto cfg_max_booklen  = cfg_radius * 2;
        const auto spreducer_in_len = (*predictor).get_data_len();
        const auto codec_in_len     = (*predictor).get_quant_len();

        auto allocate_predictor = [&]() { (*predictor).allocate_workspace(dbg_print); };
        auto allocate_spreducer = [&]() { (*spreducer).allocate_workspace(spreducer_in_len, sp_factor, dbg_print); };
        auto allocate_codec     = [&]() {
            if (codec_config == 0b00)
                throw std::runtime_error("[DefaultPathCompressor::allocate_workspace] must have set bit(s).");
            if (codec_config bitand 0b01) {
                (*codec).allocate_workspace(codec_in_len, cfg_max_booklen, cfg_pardeg, dbg_print);
            if (codec_config bitand 0b10) {
                (*fb_codec).allocate_workspace(codec_in_len, cfg_max_booklen, cfg_pardeg, dbg_print);
        };

        allocate_predictor(), allocate_spreducer(), allocate_codec();
        CHECK_CUDA(cudaMalloc(&d_reserved_compressed, (*predictor).get_data_len() * sizeof(T) / 2));
    }

    void try_report_compression(size_t compressed_len)
    {
        auto get_cr        = [&]() { return get_data_len() * sizeof(T) * 1.0 / compressed_len; };
        auto byte_to_gbyte = [&](double bytes) { return bytes / 1024 / 1024 / 1024; };
        auto ms_to_s       = [&](double ms) { return ms / 1000; };

        auto bytes = get_data_len() * sizeof(T);

        auto time_p = (*predictor).get_time_elapsed();
        auto tp_p   = byte_to_gbyte(bytes) / ms_to_s(time_p);

        float time_h, time_b, time_c;
        float tp_h, tp_b, tp_c;
        if (not use_fallback_codec) {
            time_h = (*codec).get_time_hist();
            tp_h   = byte_to_gbyte(bytes) / ms_to_s(time_h);
            time_b = (*codec).get_time_book();
            tp_b   = byte_to_gbyte(bytes) / ms_to_s(time_b);
            time_c = (*codec).get_time_lossless();
            tp_c   = byte_to_gbyte(bytes) / ms_to_s(time_c);
        }
        else {
            time_h = (*fb_codec).get_time_hist();
            tp_h   = byte_to_gbyte(bytes) / ms_to_s(time_h);
            time_b = (*fb_codec).get_time_book();
            tp_b   = byte_to_gbyte(bytes) / ms_to_s(time_b);
            time_c = (*fb_codec).get_time_lossless();
            tp_c   = byte_to_gbyte(bytes) / ms_to_s(time_c);
        }

        auto time_s = (*spreducer).get_time_elapsed();
        auto tp_s   = byte_to_gbyte(bytes) / ms_to_s(time_s);

        auto time_subtotal = time_p + time_h + time_c + time_s;
        auto tp_subtotal   = byte_to_gbyte(bytes) / ms_to_s(time_subtotal);

        auto time_total = time_subtotal + time_b;
        auto tp_total   = byte_to_gbyte(bytes) / ms_to_s(time_total);

        printf("\n(c) COMPRESSION REPORT\n");
        printf("%-*s: %.2f\n", 20, "compression ratio", get_cr());
        printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "predictor time", time_p, tp_p);
        printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "hist time", time_h, tp_h);
        printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "codec time", time_c, tp_c);
        printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "spreducer time", time_s, tp_s);
        printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "-- subtotal time", time_subtotal, tp_subtotal);
        printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "book time", time_b, tp_b);
        printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "-- total time", time_total, tp_total);
        printf("\n");
    }

    void try_report_decompression()
    {
        auto byte_to_gbyte = [&](double bytes) { return bytes / 1024 / 1024 / 1024; };
        auto ms_to_s       = [&](double ms) { return ms / 1000; };

        auto bytes = get_data_len() * sizeof(T);

        auto time_p = (*predictor).get_time_elapsed();
        auto tp_p   = byte_to_gbyte(bytes) / ms_to_s(time_p);

        float time_c, tp_c;
        if (not use_fallback_codec) {
            time_c = (*codec).get_time_lossless();
            tp_c   = byte_to_gbyte(bytes) / ms_to_s(time_c);
        }
        else {
            time_c = (*fb_codec).get_time_lossless();
            tp_c   = byte_to_gbyte(bytes) / ms_to_s(time_c);
        }

        auto time_s = (*spreducer).get_time_elapsed();
        auto tp_s   = byte_to_gbyte(bytes) / ms_to_s(time_s);

        auto time_total = time_p + time_s + time_c;
        auto tp_total   = byte_to_gbyte(bytes) / ms_to_s(time_total);

        printf("\n(d) deCOMPRESSION REPORT\n");
        printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "spreducer time", time_s, tp_s);
        printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "codec time", time_c, tp_c);
        printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "predictor time", time_p, tp_p);
        printf("%-*s: %4.3f ms\tthroughput  : %4.2f GiB/s\n", 20, "-- total time", time_total, tp_total);
        printf("\n");
    }

    /**
     * @brief
     *
     * @param uncompressed
     * @param eb
     * @param radius
     * @param compressed
     * @param compressed_len
     * @param stream
     */
    void compress(
        T*           uncompressed,
        double const eb,
        int const    radius,
        int const    pardeg,
        BYTE*&       compressed,
        size_t&      compressed_len,
        bool         force_use_fallback_codec,
        cudaStream_t stream    = nullptr,
        bool         rpt_print = true,
        bool         dbg_print = false)
    {
        T*     d_anchor{nullptr};   // predictor out1
        E*     d_errctrl{nullptr};  // predictor out2
        BYTE*  d_spfmt{nullptr};
        size_t spfmt_out_len{0};

        BYTE*  d_codec_out{nullptr};
        size_t codec_out_len{0};

        HEADER header;

        auto data_len    = (*predictor).get_data_len();
        auto m           = Reinterpret1DTo2D::get_square_size(data_len);
        auto errctrl_len = (*predictor).get_quant_len();
        auto sublen      = ConfigHelper::get_npart(data_len, pardeg);

        auto predictor_do = [&]() { (*predictor).construct(uncompressed, eb, radius, d_anchor, d_errctrl, stream); };
        auto spreducer_do = [&]() {
            (*spreducer).gather(uncompressed, m * m, d_spfmt, spfmt_out_len, stream, dbg_print);
        };

        auto codec_do_with_exception = [&]() {
            auto encode_with_fallback_codec = [&]() {
                use_fallback_codec = true;
                if (not fallback_codec_allocated) {
                    (*fb_codec).allocate_workspace(errctrl_len, radius * 2, pardeg, /*dbg print*/ false);
                    fallback_codec_allocated = true;
                }
                (*fb_codec).encode(
                    d_errctrl, errctrl_len, radius * 2, sublen, pardeg, d_codec_out, codec_out_len, stream);
            };

            if (not force_use_fallback_codec) {
                try {
                    (*codec).encode(
                        d_errctrl, errctrl_len, radius * 2, sublen, pardeg, d_codec_out, codec_out_len, stream);
                }
                catch (const std::runtime_error& e) {
                    cout << "exception switch to fallback codec\n";
                    encode_with_fallback_codec();
                }
            }
            else {
                cout << "force switch to fallback codec\n";
                encode_with_fallback_codec();
            }
        };

        auto update_header = [&]() {
            header.x          = data_size.x;
            header.y          = data_size.y;
            header.z          = data_size.z;
            header.radius     = radius;
            header.vle_pardeg = pardeg;
            header.eb         = eb;
            header.byte_vle   = use_fallback_codec ? 8 : 4;
        };

        auto subfile_collect = [&]() {
            header.header_nbyte = sizeof(HEADER);
            uint32_t nbyte[HEADER::END];
            nbyte[HEADER::HEADER] = 128;
            nbyte[HEADER::ANCHOR] = sizeof(T) * (*predictor).get_anchor_len();
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

        predictor_do(), spreducer_do(), codec_do_with_exception();

        /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

        update_header(), subfile_collect();
        compressed_len = header.file_size();
        compressed     = d_reserved_compressed;

        if (rpt_print) try_report_compression(compressed_len);

        // considering that codec can be consecutively in use, and can compress data of different huff-byte
        use_fallback_codec = false;
    }

    /**
     * @brief
     *
     */
    void clear_buffer()
    {  //
        (*predictor).clear_buffer();
        (*codec).clear_buffer();
        (*spreducer).clear_buffer();
    }

    /**
     * @brief High-level decompress method for this compressor
     *
     * @param in_compressed device pointer, the cusz archive bianry
     * @param eb host variable, error bound
     * @param radius host variable, in this case, it is 0
     * @param out_decompressed device pointer, output decompressed data
     * @param stream CUDA stream
     */
    void decompress(
        BYTE*        in_compressed,
        double const eb,
        int const    radius,
        T*           out_decompressed,
        cudaStream_t stream    = nullptr,
        bool         rpt_print = true)
    {
        // TODO host having copy of header
        HEADER header;
        CHECK_CUDA(cudaMemcpyAsync(&header, in_compressed, sizeof(header), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        use_fallback_codec = header.byte_vle == 8;

        // The inputs of components are from `compressed`.
        auto d_anchor       = ACCESSOR(ANCHOR, T);
        auto d_decoder_in   = ACCESSOR(VLE, BYTE);
        auto d_spreducer_in = ACCESSOR(SPFMT, BYTE);

        // wire the workspace
        auto d_errctrl = (*predictor).expose_quant();  // reuse space

        // aliasing
        auto d_decoder_out  = d_errctrl;
        auto d_predictor_in = d_errctrl;

        // wire and aliasing
        auto d_spreducer_out = out_decompressed;
        auto d_predictor_out = out_decompressed;

        auto spreducer_do            = [&]() { (*spreducer).scatter(d_spreducer_in, d_spreducer_out, stream); };
        auto codec_do_with_exception = [&]() {
            if (not use_fallback_codec) { (*codec).decode(d_decoder_in, d_decoder_out); }
            else {
                if (not fallback_codec_allocated) {
                    (*fb_codec).allocate_workspace(
                        (*predictor).get_quant_len(), header.radius * 2, header.vle_pardeg, /*dbg print*/ false);
                    fallback_codec_allocated = true;
                }

                (*fb_codec).decode(d_decoder_in, d_decoder_out);
            }
        };
        auto predictor_do = [&]() {
            (*predictor).reconstruct(d_anchor, d_predictor_in, eb, radius, d_predictor_out, stream);
        };

        spreducer_do(), codec_do_with_exception(), predictor_do();

        try_report_decompression();

        // clear state for the next decompression after reporting
        use_fallback_codec = false;
    }
};

struct DefaultPath {
    using DATA    = DataTrait<4>::type;
    using ERRCTRL = ErrCtrlTrait<2>::type;
    using FP      = FastLowPrecisionTrait<true>::type;

    using LorenzoBasedBinding = PredictorReducerCodecBinding<
        cusz::PredictorLorenzo<DATA, ERRCTRL, FP>,
        cusz::CSR11<DATA>,
        // cusz::spGS<DATA>,  //  not woking for CUDA 10.2 on ppc
        cusz::HuffmanCoarse<ERRCTRL, HuffTrait<4>::type, MetadataTrait<4>::type>,
        cusz::HuffmanCoarse<ERRCTRL, HuffTrait<8>::type, MetadataTrait<4>::type>  //
        >;
    using DefaultBinding = LorenzoBasedBinding;

    using DefaultCompressor      = class DefaultPathCompressor<DefaultBinding>;
    using LorenzoBasedCompressor = class DefaultPathCompressor<DefaultBinding>;
};

#undef FREEDEV
#undef FREEHOST
#undef DEFINE_DEV
#undef DEFINE_HOST
#undef D2D_CPY
#undef ACCESSOR

#endif
