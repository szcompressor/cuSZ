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
    using Predictor = typename BINDING::PREDICTOR;
    using SpReducer = typename BINDING::SPREDUCER;
    using Codec     = typename BINDING::CODEC;

    using BYTE = uint8_t;
    using T    = typename Predictor::Origin;
    using FP   = typename Predictor::Precision;
    using E    = typename Predictor::ErrCtrl;
    using H    = typename Codec::Encoded;
    using M    = typename Codec::MetadataT;

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
        // uint32_t byte_meta : 4;       // 4, 8
        uint32_t vle_pardeg;
        uint32_t x : 16;
        uint32_t y : 16;
        uint32_t z : 16;
        uint32_t w : 16;
        uint32_t ndim : 3;  // 1,2,3,4
        double   eb;
        size_t   data_len;
        size_t   errctrl_len;
        uint32_t radius : 16;

        uint32_t entry[END];

        uint32_t file_size() const { return entry[END]; }
    };
    using HEADER = struct header_t;

    struct runtime_helper {
    };
    using RT = runtime_helper;
    RT rt;

   private:
    ////// new
    BYTE* d_reserved_compressed{nullptr};

    ////// (end of) new

   private:
    // --------------------
    // not in base class
    // --------------------
    Capsule<H>    book;
    Capsule<H>    huff_data;
    Capsule<M>    huff_counts;
    Capsule<BYTE> revbook;
    Capsule<BYTE> sp_use;

    // tmp, device only
    Capsule<int> ext_rowptr;
    Capsule<int> ext_colidx;
    Capsule<T>   ext_values;

    H* huff_workspace;  // compress

    struct {
        Capsule<H>    in;
        Capsule<M>    meta;
        Capsule<BYTE> revbook;
    } xhuff;

    Predictor* predictor;
    SpReducer* spreducer;
    Codec*     codec;

    size_t   m, mxm;
    uint32_t sp_dump_nbyte;

   private:
    dim3     data_size;
    uint32_t get_data_len() { return data_size.x * data_size.y * data_size.z; }

   private:
    // TODO better move to base compressor
    DefaultPathCompressor& analyze_compressibility();
    DefaultPathCompressor& internal_eval_try_export_book();
    DefaultPathCompressor& internal_eval_try_export_quant();
    DefaultPathCompressor& try_skip_huffman();
    // DefaultPathCompressor& get_freq_codebook();
    // DefaultPathCompressor& old_huffman_encode();

   public:
    DefaultPathCompressor(cuszCTX* _ctx, Capsule<T>* _in_data, uint3 xyz, int dict_size);
    DefaultPathCompressor(cuszCTX* _ctx, Capsule<BYTE>* _in_dump);

    /**
     * @deprecated
     */
    ~DefaultPathCompressor()
    {
        if (this->timing == cusz::WHEN::COMPRESS) {  // release small-size arrays

            this->quant.template free<kDEVICE>();
            this->freq.template free<kDEVICE>();
            huff_data.template free<kHOST_DEVICE>();
            huff_counts.template free<kHOST_DEVICE>();
            sp_use.template free<kHOST_DEVICE>();
            book.template free<kDEVICE>();
            revbook.template free<kHOST_DEVICE>();

            cudaFree(huff_workspace);

            ext_rowptr.template free<kDEVICE>();
            ext_colidx.template free<kDEVICE>();
            ext_values.template free<kDEVICE>();

            delete this->header;
        }
        else {
            cudaFree(sp_use.dptr);

            xhuff.in.template free<kDEVICE>();
            xhuff.meta.template free<kDEVICE>();
            xhuff.revbook.template free<kDEVICE>();
        }

        if (spreducer) delete spreducer;
        if (codec) delete codec;
        if (predictor) delete predictor;
    }

    /*
    ~DefaultPathCompressor()
    {
        if (predictor) delete predictor;
        if (codec) delete codec;
        if (spreducer) delete spreducer;
    };
    */

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
    DefaultPathCompressor(uint3 xyz) : data_size(xyz)
    {
        predictor = new Predictor(xyz);
        spreducer = new SpReducer;
        codec     = new Codec;
    }

    /**
     * @brief Allocate workspace accordingly.
     *
     * @param cfg_max_booklen
     * @param cfg_pardeg
     * @param dbg_print
     */
    void allocate_workspace(int cfg_max_booklen, int cfg_pardeg, bool dbg_print = false)
    {
        (*predictor).allocate_workspace(dbg_print);

        auto spreducer_in_len = (*predictor).get_data_len();
        (*spreducer).allocate_workspace(spreducer_in_len, dbg_print);

        auto codec_in_len = (*predictor).get_quant_len();
        (*codec).allocate_workspace(codec_in_len, cfg_max_booklen, cfg_pardeg, dbg_print);

        CHECK_CUDA(cudaMalloc(&d_reserved_compressed, (*predictor).get_data_len() * sizeof(T) / 2));
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
        cudaStream_t stream    = nullptr,
        bool         dbg_print = false)
    {
        T*     d_anchor{nullptr};   // predictor out1
        E*     d_errctrl{nullptr};  // predictor out2
        BYTE*  d_spfmt{nullptr};
        size_t spfmt_out_len{0};

        BYTE*  d_codec_out{nullptr};
        size_t codec_out_len{0};

        HEADER header;

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
                // for (auto i = 0; i < HEADER::END + 1; i++) printf("%u, header entry: %u\n", i, header.entry[i]);
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

        (*predictor).construct(uncompressed, eb, radius, d_anchor, d_errctrl, stream);

        auto data_len = (*predictor).get_data_len();
        auto m        = Reinterpret1DTo2D::get_square_size(data_len);

        (*spreducer).gather_new(uncompressed, m * m, d_spfmt, spfmt_out_len, stream, dbg_print);

        auto errctrl_len = (*predictor).get_quant_len();
        (*codec).encode_new(d_errctrl, errctrl_len, radius * 2, pardeg, d_codec_out, codec_out_len, stream);

        /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

        subfile_collect();
        compressed_len = header.file_size();
        compressed     = d_reserved_compressed;

        // cout << "(c) predictor time  : " << (*predictor).get_time_elapsed() << " ms\n";
        // cout << "(c) codec time      : " << (*codec).get_time_elapsed() << " ms\n";
        // cout << "(c) spreducer time  : " << (*spreducer).get_time_elapsed() << " ms\n";

        auto compress_report = [&]() {
            auto get_cr        = [&]() { return get_data_len() * sizeof(T) * 1.0 / compressed_len; };
            auto byte_to_gbyte = [&](double bytes) { return bytes / 1024 / 1024 / 1024; };
            auto ms_to_s       = [&](double ms) { return ms / 1000; };

            auto bytes = get_data_len() * sizeof(T);

            auto time_p = (*predictor).get_time_elapsed();
            auto tp_p   = byte_to_gbyte(bytes) / ms_to_s(time_p);

            auto time_h = (*codec).get_time_hist();
            auto tp_h   = byte_to_gbyte(bytes) / ms_to_s(time_h);
            auto time_b = (*codec).get_time_book();
            auto tp_b   = byte_to_gbyte(bytes) / ms_to_s(time_b);
            auto time_c = (*codec).get_time_lossless();
            auto tp_c   = byte_to_gbyte(bytes) / ms_to_s(time_c);

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
        };

        compress_report();
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
        cudaStream_t stream = nullptr)
    {
        HEADER header;
        CHECK_CUDA(cudaMemcpyAsync(&header, in_compressed, sizeof(header), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

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

        (*spreducer).scatter_new(d_spreducer_in, d_spreducer_out, stream);
        (*codec).decode_new(d_decoder_in, d_decoder_out);
        (*predictor).reconstruct(d_anchor, d_predictor_in, eb, radius, d_predictor_out, stream);

        auto decompress_report = [&]() {
            auto byte_to_gbyte = [&](double bytes) { return bytes / 1024 / 1024 / 1024; };
            auto ms_to_s       = [&](double ms) { return ms / 1000; };

            auto bytes = get_data_len() * sizeof(T);

            auto time_p = (*predictor).get_time_elapsed();
            auto tp_p   = byte_to_gbyte(bytes) / ms_to_s(time_p);

            auto time_c = (*codec).get_time_lossless();
            auto tp_c   = byte_to_gbyte(bytes) / ms_to_s(time_c);

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
        };

        decompress_report();
    }
};

struct DefaultPath {
    using DATA    = DataTrait<4>::type;
    using ERRCTRL = ErrCtrlTrait<2>::type;
    using FP      = FastLowPrecisionTrait<true>::type;

    using DefaultBinding = PredictorReducerCodecBinding<
        cusz::PredictorLorenzo<DATA, ERRCTRL, FP>,
        cusz::CSR11<DATA>,
        // cusz::spGS<DATA>,  //  not woking for CUDA 10.2 on ppc
        cusz::HuffmanCoarse<ERRCTRL, HuffTrait<4>::type, MetadataTrait<4>::type>  //
        >;

    using DefaultCompressor = class DefaultPathCompressor<DefaultBinding>;

    using FallbackBinding = PredictorReducerCodecBinding<
        cusz::PredictorLorenzo<DATA, ERRCTRL, FP>,
        cusz::CSR11<DATA>,
        // cusz::spGS<DATA>,  //  not woking for CUDA 10.2 ppc
        cusz::HuffmanCoarse<ERRCTRL, HuffTrait<8>::type, MetadataTrait<4>::type>  //
        >;

    using FallbackCompressor = class DefaultPathCompressor<FallbackBinding>;
};

#undef FREEDEV
#undef FREEHOST
#undef DEFINE_DEV
#undef DEFINE_HOST
#undef D2D_CPY
#undef ACCESSOR

#endif
