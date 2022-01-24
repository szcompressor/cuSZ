/**
 * @file huffman_coarse.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-12-17
 * (created) 2020-04-24 (rev1) 2021-09-05 (rev2) 2021-12-29
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * @copyright (C) 2021 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_WRAPPER_HUFFMAN_COARSE_CUH
#define CUSZ_WRAPPER_HUFFMAN_COARSE_CUH

#include <clocale>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <type_traits>
using std::cout;

#include "../../include/reducer.hh"
#include "../common/capsule.hh"
#include "../common/definition.hh"
#include "../common/type_traits.hh"
#include "../header.hh"
#include "../kernel/codec_huffman.cuh"
#include "../kernel/hist.cuh"
#include "../utils.hh"
#include "huffman_coarse.cuh"
#include "huffman_parbook.cuh"

/******************************************************************************
                            macros for shorthand writing
 ******************************************************************************/

#define EXPORT_NBYTE(FIELD) nbyte[HEADER::FIELD] = rte.nbyte[RTE::FIELD];

#define DEVICE2DEVICE_COPY(VAR, FIELD)                                            \
    {                                                                             \
        constexpr auto D2D = cudaMemcpyDeviceToDevice;                            \
        auto           dst = d_compressed + header.entry[HEADER::FIELD];          \
        auto           src = reinterpret_cast<BYTE*>(d_##VAR);                    \
        CHECK_CUDA(cudaMemcpyAsync(dst, src, nbyte[HEADER::FIELD], D2D, stream)); \
    }

#define DEFINE_HC_ARRAY(VAR, TYPE) \
    TYPE* d_##VAR{nullptr};        \
    TYPE* h_##VAR{nullptr};

#define ACCESSOR(SYM, TYPE) reinterpret_cast<TYPE*>(in_compressed + header.entry[HEADER::SYM])

#define HC_ALLOCHOST(VAR, SYM)                     \
    cudaMallocHost(&h_##VAR, rte.nbyte[RTE::SYM]); \
    memset(h_##VAR, 0x0, rte.nbyte[RTE::SYM]);

#define HC_ALLOCDEV(VAR, SYM)                  \
    cudaMalloc(&d_##VAR, rte.nbyte[RTE::SYM]); \
    cudaMemset(d_##VAR, 0x0, rte.nbyte[RTE::SYM]);

#define HC_FREEHOST(VAR)       \
    if (h_##VAR) {             \
        cudaFreeHost(h_##VAR); \
        h_##VAR = nullptr;     \
    }

#define HC_FREEDEV(VAR)    \
    if (d_##VAR) {         \
        cudaFree(d_##VAR); \
        d_##VAR = nullptr; \
    }

/******************************************************************************
                                class definition
 ******************************************************************************/

namespace cusz {

template <typename Huff, typename Meta>
__global__ void
huffman_coarse_concatenate(Huff* gapped, Meta* par_entry, Meta* par_ncell, int const cfg_sublen, Huff* non_gapped)
{
    auto n   = par_ncell[blockIdx.x];
    auto src = gapped + cfg_sublen * blockIdx.x;
    auto dst = non_gapped + par_entry[blockIdx.x];

    for (auto i = threadIdx.x; i < n; i += blockDim.x) {  // block-stride
        dst[i] = src[i];
    }
}

}  // namespace cusz

namespace cusz {

template <typename T, typename H, typename M = uint32_t>
class HuffmanCoarse : public cusz::VariableRate {
   public:
    using Origin    = T;
    using Encoded   = H;
    using MetadataT = M;
    using FreqT     = cusz::FREQ;
    using BYTE      = uint8_t;

   private:
    DEFINE_HC_ARRAY(tmp, H);
    DEFINE_HC_ARRAY(compressed, BYTE);  // alias in address
    DEFINE_HC_ARRAY(freq, FreqT);
    DEFINE_HC_ARRAY(book, H);
    DEFINE_HC_ARRAY(revbook, BYTE);
    DEFINE_HC_ARRAY(par_nbit, M);
    DEFINE_HC_ARRAY(par_ncell, M);
    DEFINE_HC_ARRAY(par_entry, M);
    DEFINE_HC_ARRAY(bitstream, H);

   public:
    /**
     * @brief on host; separate from (binary) data fields
     * otherwise, aligning to 128B can be unwanted
     *
     */
    struct header_t {
        static const int HEADER    = 0;
        static const int REVBOOK   = 1;
        static const int PAR_NBIT  = 2;
        static const int PAR_ENTRY = 3;
        static const int BITSTREAM = 4;
        static const int END       = 5;

        int       header_nbyte : 16;
        int       booklen : 16;
        int       sublen;
        size_t    uncompressed_len;
        size_t    total_nbit;
        size_t    total_ncell;  // TODO change to uint32_t
        MetadataT entry[END + 1];

        MetadataT subfile_size() const { return entry[END]; }
    };
    using HEADER = header_t;

    struct runtime_encode_helper {
        static const int TMP       = 0;
        static const int FREQ      = 1;
        static const int BOOK      = 2;
        static const int REVBOOK   = 3;
        static const int PAR_NBIT  = 4;
        static const int PAR_NCELL = 5;
        static const int PAR_ENTRY = 6;
        static const int BITSTREAM = 7;
        static const int END       = 8;

        uint32_t nbyte[END];
    };
    using RTE = runtime_encode_helper;
    RTE rte;

    /**
     * @brief Allocate workspace according to the input size & configurations.
     *
     * @param in_uncompressed_len uncompressed length
     * @param cfg_booklen codebook length
     * @param cfg_pardeg degree of parallelism
     * @param dbg_print print for debugging
     */
    void allocate_workspace(size_t const in_uncompressed_len, int cfg_booklen, int cfg_pardeg, bool dbg_print = false)
    {
        auto max_compressed_bytes = [&]() { return in_uncompressed_len / 2 * sizeof(H); };
        auto debug                = [&]() {
            setlocale(LC_NUMERIC, "");
#define PRINT_DBG(VAR) printf("nbyte-%-*s:  %'10u\n", 10, #VAR, rte.nbyte[RTE::VAR]);
            printf("\nHuffmanCoarse::allocate_workspace() debugging:\n");
            PRINT_DBG(TMP);
            PRINT_DBG(FREQ);
            PRINT_DBG(BOOK);
            PRINT_DBG(REVBOOK);
            PRINT_DBG(PAR_NBIT);
            PRINT_DBG(PAR_NCELL);
            PRINT_DBG(BITSTREAM);
            printf("\n");
#undef PRINT_DBG
        };

        memset(rte.nbyte, 0, sizeof(uint32_t) * RTE::END);
        // memset(rte.entry, 0, sizeof(uint32_t) * (RTE::END + 1));

        rte.nbyte[RTE::TMP]       = sizeof(H) * in_uncompressed_len;
        rte.nbyte[RTE::FREQ]      = sizeof(FreqT) * cfg_booklen;
        rte.nbyte[RTE::BOOK]      = sizeof(H) * cfg_booklen;
        rte.nbyte[RTE::REVBOOK]   = get_revbook_nbyte(cfg_booklen);
        rte.nbyte[RTE::PAR_NBIT]  = sizeof(M) * cfg_pardeg;
        rte.nbyte[RTE::PAR_NCELL] = sizeof(M) * cfg_pardeg;
        rte.nbyte[RTE::PAR_ENTRY] = sizeof(M) * cfg_pardeg;
        rte.nbyte[RTE::BITSTREAM] = max_compressed_bytes();

        HC_ALLOCDEV(tmp, TMP);
        HC_ALLOCDEV(freq, FREQ);
        HC_ALLOCDEV(book, BOOK);
        HC_ALLOCDEV(revbook, REVBOOK);
        HC_ALLOCDEV(par_nbit, PAR_NBIT);
        HC_ALLOCDEV(par_ncell, PAR_NCELL);
        HC_ALLOCDEV(par_entry, PAR_ENTRY);
        HC_ALLOCDEV(bitstream, BITSTREAM);

        // standalone definition for output
        d_compressed = reinterpret_cast<BYTE*>(d_tmp);

        HC_ALLOCHOST(freq, FREQ);
        HC_ALLOCHOST(book, BOOK);
        HC_ALLOCHOST(revbook, REVBOOK);
        HC_ALLOCHOST(par_nbit, PAR_NBIT);
        HC_ALLOCHOST(par_ncell, PAR_NCELL);
        HC_ALLOCHOST(par_entry, PAR_ENTRY);

        if (dbg_print) debug();
    }

   private:
    using BOOK = H;
    using SYM  = T;

    static const int CELL_BITWIDTH = sizeof(H) * 8;

    float milliseconds{0.0};
    float time_hist{0.0}, time_book{0.0}, time_lossless{0.0};

   public:
    //
    float get_time_elapsed() const { return milliseconds; }
    float get_time_hist() const { return time_hist; }
    float get_time_book() const { return time_book; }
    float get_time_lossless() const { return time_lossless; }

    // TODO this kind of space will be overlapping with quant-codes
    size_t get_workspace_nbyte(size_t len) const { return sizeof(H) * len; }
    size_t get_max_output_nbyte(size_t len) const { return sizeof(H) * len / 2; }

    static uint32_t get_revbook_nbyte(int dict_size)
    {
        return sizeof(BOOK) * (2 * CELL_BITWIDTH) + sizeof(SYM) * dict_size;
    }

    constexpr bool can_overlap_input_and_firstphase_encode() { return sizeof(T) == sizeof(H); }

    static size_t tune_coarse_huffman_sublen(size_t len)
    {
        int current_dev = 0;
        cudaSetDevice(current_dev);
        cudaDeviceProp dev_prop{};
        cudaGetDeviceProperties(&dev_prop, current_dev);

        auto nSM               = dev_prop.multiProcessorCount;
        auto allowed_block_dim = dev_prop.maxThreadsPerBlock;
        auto deflate_nthread   = allowed_block_dim * nSM / HuffmanHelper::DEFLATE_CONSTANT;
        auto optimal_sublen    = ConfigHelper::get_npart(len, deflate_nthread);
        optimal_sublen         = ConfigHelper::get_npart(optimal_sublen, HuffmanHelper::BLOCK_DIM_DEFLATE) *
                         HuffmanHelper::BLOCK_DIM_DEFLATE;

        return optimal_sublen;
    }

    static void get_coarse_pardeg(size_t len, int& sublen, int& pardeg)
    {
        sublen = HuffmanCoarse::tune_coarse_huffman_sublen(len);
        pardeg = ConfigHelper::get_npart(len, sublen);
    }

   public:
    // 21-12-17 toward static method
    HuffmanCoarse() = default;

    ~HuffmanCoarse()
    {
        HC_FREEDEV(tmp);
        HC_FREEDEV(freq);
        HC_FREEDEV(book);
        HC_FREEDEV(revbook);
        HC_FREEDEV(par_nbit);
        HC_FREEDEV(par_ncell);
        HC_FREEDEV(par_entry);
        HC_FREEDEV(bitstream);

        HC_FREEHOST(freq);
        HC_FREEHOST(book);
        HC_FREEHOST(revbook);
        HC_FREEHOST(par_nbit);
        HC_FREEHOST(par_ncell);
        HC_FREEHOST(par_entry);
    }

   public:
    /**
     * @brief Inspect the input data; generate histogram, codebook (for encoding), reversed codebook (for decoding).
     *
     * @param tmp_freq (device array) If called by other class methods, use class-private d_freq; otherwise, use
     * external array.
     * @param tmp_book (device array) If called by other class methods, use class-private d_book.
     * @param in_uncompressed (device array) input data
     * @param in_uncompressed_len (host variable) input data length
     * @param cfg_booklen (host variable) configuration, book size
     * @param out_revbook (device array) If called by other class methods, use class-private d_revbook.
     * @param stream CUDA stream
     */
    void inspect(
        cusz::FREQ*  tmp_freq,
        H*           tmp_book,
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        BYTE*        out_revbook,
        cudaStream_t stream = nullptr)
    {
        kernel_wrapper::get_frequency<T>(
            in_uncompressed, in_uncompressed_len, tmp_freq, cfg_booklen, time_hist, stream);

        // This is end-to-end time for parbook.
        cuda_timer_t t;
        t.timer_start(stream);
        kernel_wrapper::par_get_codebook<T, H>(tmp_freq, cfg_booklen, tmp_book, out_revbook, stream);
        t.timer_end(stream);
        cudaStreamSynchronize(stream);

        time_book = t.get_time_elapsed();
    }

   public:
    /**
     * @brief
     * @deprecated use `encode_new` instead
     *
     * @param d_freq
     * @param d_book
     * @param d_tmp
     * @param in_uncompressed
     * @param in_uncompressed_len
     * @param cfg_booklen
     * @param cfg_sublen
     * @param d_revbook
     * @param out_compressed
     * @param out_compressed_meta
     * @param out_total_nbit
     * @param out_total_ncell
     * @param stream
     */
    void encode_integrated(
        FreqT*       d_freq,
        H*           d_book,
        H*           d_tmp,
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        int const    cfg_sublen,
        BYTE*        d_revbook,
        Capsule<H>&  out_compressed,
        Capsule<M>&  out_compressed_meta,
        size_t&      out_total_nbit,
        size_t&      out_total_ncell,
        cudaStream_t stream = nullptr)
    {
        auto const cfg_pardeg = ConfigHelper::get_npart(in_uncompressed_len, cfg_sublen);

        auto d_par_nbit  = out_compressed_meta.dptr;
        auto d_par_ncell = out_compressed_meta.dptr + cfg_pardeg;
        auto d_par_entry = out_compressed_meta.dptr + cfg_pardeg * 2;
        // TODO change order
        auto h_par_ncell = out_compressed_meta.hptr;
        auto h_par_nbit  = out_compressed_meta.hptr + cfg_pardeg;
        auto h_par_entry = out_compressed_meta.hptr + cfg_pardeg * 2;

        auto encode_phase1 = [&]() {
            auto block_dim = HuffmanHelper::BLOCK_DIM_ENCODE;
            auto grid_dim  = ConfigHelper::get_npart(in_uncompressed_len, block_dim);

            int numSMs;
            cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

            cuda_timer_t t;
            t.timer_start(stream);

            cusz::coarse_par::detail::kernel::huffman_encode_fixedlen_gridstride<T, H>
                <<<8 * numSMs, 256, sizeof(H) * cfg_booklen, stream>>>  //
                (in_uncompressed, in_uncompressed_len, d_book, cfg_booklen, d_tmp);

            t.timer_end(stream);
            time_lossless += t.get_time_elapsed();
            cudaStreamSynchronize(stream);
        };

        auto encode_phase2 = [&]() {
            auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
            auto grid_dim  = ConfigHelper::get_npart(cfg_pardeg, block_dim);

            cuda_timer_t t;
            t.timer_start(stream);

            cusz::coarse_par::detail::kernel::huffman_encode_deflate<H><<<grid_dim, block_dim, 0, stream>>>  //
                (d_tmp, in_uncompressed_len, d_par_nbit, d_par_ncell, cfg_sublen, cfg_pardeg);

            t.timer_end(stream);
            time_lossless += t.get_time_elapsed();
            cudaStreamSynchronize(stream);
        };

        auto encode_phase3 = [&]() {
            cudaMemcpyAsync(h_par_nbit, d_par_nbit, cfg_pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_par_ncell, d_par_ncell, cfg_pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            memcpy(h_par_entry + 1, h_par_ncell, (cfg_pardeg - 1) * sizeof(M));
            for (auto i = 1; i < cfg_pardeg; i++) h_par_entry[i] += h_par_entry[i - 1];  // inclusive scan

            out_total_nbit  = std::accumulate(h_par_nbit, h_par_nbit + cfg_pardeg, (size_t)0);
            out_total_ncell = std::accumulate(h_par_ncell, h_par_ncell + cfg_pardeg, (size_t)0);

            cudaMemcpyAsync(d_par_entry, h_par_entry, cfg_pardeg * sizeof(M), cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
        };

        auto encode_phase4 = [&]() {
            cuda_timer_t t;
            t.timer_start(stream);
            cusz::huffman_coarse_concatenate<H, M><<<cfg_pardeg, 128, 0, stream>>>  //
                (d_tmp, d_par_entry, d_par_ncell, cfg_sublen, out_compressed.dptr);
            t.timer_end(stream);
            time_lossless += t.get_time_elapsed();
            cudaStreamSynchronize(stream);
        };

        // -----------------------------------------------------------------------------

        inspect(d_freq, d_book, in_uncompressed, in_uncompressed_len, cfg_booklen, d_revbook, stream);

        encode_phase1();
        encode_phase2();
        encode_phase3();

        // update with the exact length
        out_compressed.set_len(out_total_ncell);

        encode_phase4();
    }

    /**
     * @brief
     * @deprecated use `decode_new` instead
     *
     * @param in_compressed
     * @param in_compressed_meta
     * @param in_revbook
     * @param in_uncompressed_len
     * @param cfg_booklen
     * @param cfg_sublen
     * @param out_decompressed
     * @param stream
     */
    void decode(
        H*           in_compressed,
        M*           in_compressed_meta,
        BYTE*        in_revbook,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        int const    cfg_sublen,
        T*           out_decompressed,
        cudaStream_t stream = nullptr)
    {
        auto const pardeg        = ConfigHelper::get_npart(in_uncompressed_len, cfg_sublen);
        auto       revbook_nbyte = get_revbook_nbyte(cfg_booklen);

        auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
        auto grid_dim  = ConfigHelper::get_npart(pardeg, block_dim);

        cuda_timer_t t;
        t.timer_start(stream);
        cusz::coarse_par::detail::kernel::huffman_decode<T, H, M><<<grid_dim, block_dim, revbook_nbyte, stream>>>(
            in_compressed, in_compressed_meta, in_revbook, revbook_nbyte, cfg_sublen, pardeg, out_decompressed);
        t.timer_end(stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        milliseconds = t.get_time_elapsed();
    }

   private:
    /**
     * @brief Collect fragmented field with repurposing TMP space.
     *
     * @param header (host variable)
     * @param stream CUDA stream
     */
    void subfile_collect(
        HEADER&      header,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        int const    cfg_sublen,
        cudaStream_t stream = nullptr)
    {
        auto BARRIER = [&]() {
            if (stream)
                CHECK_CUDA(cudaStreamSynchronize(stream));
            else
                CHECK_CUDA(cudaDeviceSynchronize());
        };

        header.header_nbyte     = sizeof(struct header_t);
        header.booklen          = cfg_booklen;
        header.sublen           = cfg_sublen;
        header.uncompressed_len = in_uncompressed_len;

        MetadataT nbyte[HEADER::END];
        nbyte[HEADER::HEADER] = 128;

        EXPORT_NBYTE(REVBOOK)
        EXPORT_NBYTE(PAR_NBIT)
        EXPORT_NBYTE(PAR_ENTRY)
        EXPORT_NBYTE(BITSTREAM)

        header.entry[0] = 0;
        // *.END + 1: need to know the ending position
        for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] = nbyte[i - 1]; }
        for (auto i = 1; i < HEADER::END + 1; i++) { header.entry[i] += header.entry[i - 1]; }

        auto debug_header_entry = [&]() {
            for (auto i = 0; i < HEADER::END + 1; i++) printf("%d, header entry: %d\n", i, header.entry[i]);
        };
        // debug_header_entry();

        CHECK_CUDA(cudaMemcpyAsync(d_compressed, &header, sizeof(header), cudaMemcpyHostToDevice, stream));

        /* debug */ BARRIER();

        DEVICE2DEVICE_COPY(revbook, REVBOOK)
        DEVICE2DEVICE_COPY(par_nbit, PAR_NBIT)
        DEVICE2DEVICE_COPY(par_entry, PAR_ENTRY)
        DEVICE2DEVICE_COPY(bitstream, BITSTREAM)
    }

   public:
    /**
     * @brief Public encode interface.
     *
     * @param in_uncompressed (device array)
     * @param in_uncompressed_len (host variable)
     * @param cfg_booklen (host variable)
     * @param cfg_sublen (host variable)
     * @param out_compressed (device array) reference
     * @param out_compressed_len (host variable) reference output
     * @param stream CUDA stream
     */
    void encode_new(
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        int const    cfg_sublen,
        BYTE*&       out_compressed,
        size_t&      out_compressed_len,
        cudaStream_t stream = nullptr)
    {
        auto const cfg_pardeg = ConfigHelper::get_npart(in_uncompressed_len, cfg_sublen);

        cuda_timer_t t;
        time_lossless = 0;

        auto BARRIER = [&]() {
            if (stream)
                CHECK_CUDA(cudaStreamSynchronize(stream));
            else
                CHECK_CUDA(cudaDeviceSynchronize());
        };

        struct header_t header;

        auto encode_phase1_new = [&]() {
            auto block_dim = HuffmanHelper::BLOCK_DIM_ENCODE;
            auto grid_dim  = ConfigHelper::get_npart(in_uncompressed_len, block_dim);

            int numSMs;
            cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

            // cuda_timer_t t;
            t.timer_start(stream);

            cusz::coarse_par::detail::kernel::huffman_encode_fixedlen_gridstride<T, H>
                <<<8 * numSMs, 256, sizeof(H) * cfg_booklen, stream>>>  //
                (in_uncompressed, in_uncompressed_len, d_book, cfg_booklen, d_tmp);

            t.timer_end(stream);
            BARRIER();

            time_lossless += t.get_time_elapsed();
        };

        auto encode_phase2_new = [&]() {
            auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;
            auto grid_dim  = ConfigHelper::get_npart(cfg_pardeg, block_dim);

            // cuda_timer_t t;
            t.timer_start(stream);

            cusz::coarse_par::detail::kernel::huffman_encode_deflate<H><<<grid_dim, block_dim, 0, stream>>>  //
                (d_tmp, in_uncompressed_len, d_par_nbit, d_par_ncell, cfg_sublen, cfg_pardeg);

            t.timer_end(stream);
            BARRIER();

            time_lossless += t.get_time_elapsed();
        };

        auto encode_phase3_new = [&]() {
            CHECK_CUDA(cudaMemcpyAsync(h_par_nbit, d_par_nbit, cfg_pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(
                cudaMemcpyAsync(h_par_ncell, d_par_ncell, cfg_pardeg * sizeof(M), cudaMemcpyDeviceToHost, stream));
            BARRIER();

            memcpy(h_par_entry + 1, h_par_ncell, (cfg_pardeg - 1) * sizeof(M));
            for (auto i = 1; i < cfg_pardeg; i++) h_par_entry[i] += h_par_entry[i - 1];  // inclusive scan

            header.total_nbit  = std::accumulate(h_par_nbit, h_par_nbit + cfg_pardeg, (size_t)0);
            header.total_ncell = std::accumulate(h_par_ncell, h_par_ncell + cfg_pardeg, (size_t)0);

            CHECK_CUDA(
                cudaMemcpyAsync(d_par_entry, h_par_entry, cfg_pardeg * sizeof(M), cudaMemcpyHostToDevice, stream));
            BARRIER();

            // update with the precise BITSTREAM nbyte
            rte.nbyte[RTE::BITSTREAM] = sizeof(H) * header.total_ncell;
        };

        auto encode_phase4_new = [&]() {
            // cuda_timer_t t;
            t.timer_start(stream);
            {
                cusz::huffman_coarse_concatenate<H, M><<<cfg_pardeg, 128, 0, stream>>>  //
                    (d_tmp, d_par_entry, d_par_ncell, cfg_sublen, d_bitstream);
            }
            t.timer_end(stream);
            BARRIER();

            time_lossless += t.get_time_elapsed();
        };

        // -----------------------------------------------------------------------------

        inspect(d_freq, d_book, in_uncompressed, in_uncompressed_len, cfg_booklen, d_revbook, stream);

        encode_phase1_new();
        CHECK_CUDA(cudaDeviceSynchronize());
        encode_phase2_new();
        encode_phase3_new();
        encode_phase4_new();

        subfile_collect(header, in_uncompressed_len, cfg_booklen, cfg_sublen, stream);

        out_compressed     = d_compressed;
        out_compressed_len = header.subfile_size();
    }

    /**
     * @brief Public decode interface.
     *
     * @param in_compressed (device array) input
     * @param out_decompressed (device array output) output
     * @param stream CUDA stream
     * @param header_on_device If true, copy header from device binary to host.
     */
    void decode_new(
        BYTE*        in_compressed,  //
        T*           out_decompressed,
        cudaStream_t stream           = nullptr,
        bool         header_on_device = true)
    {
        header_t header;
        if (header_on_device)
            CHECK_CUDA(cudaMemcpyAsync(&header, in_compressed, sizeof(header), cudaMemcpyDeviceToHost, stream));

        auto d_revbook   = ACCESSOR(REVBOOK, BYTE);
        auto d_par_nbit  = ACCESSOR(PAR_NBIT, M);
        auto d_par_entry = ACCESSOR(PAR_ENTRY, M);
        auto d_bitstream = ACCESSOR(BITSTREAM, H);

        auto const revbook_nbyte = get_revbook_nbyte(header.booklen);
        auto const pardeg        = ConfigHelper::get_npart(header.uncompressed_len, header.sublen);
        auto const block_dim     = HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
        auto const grid_dim      = ConfigHelper::get_npart(pardeg, block_dim);

        cuda_timer_t t;
        t.timer_start(stream);
        cusz::coarse_par::detail::kernel::huffman_decode_new<T, H, M><<<grid_dim, block_dim, revbook_nbyte, stream>>>(
            d_bitstream, d_revbook, d_par_nbit, d_par_entry, revbook_nbyte, header.sublen, pardeg, out_decompressed);
        t.timer_end(stream);
        cudaStreamSynchronize(stream);

        time_lossless = t.get_time_elapsed();
    }

    // end of class definition
};

}  // namespace cusz

#undef HC_ALLOCDEV
#undef HC_ALLOCHOST
#undef HC_FREEDEV
#undef HC_FREEHOST
#undef EXPORT_NBYTE
#undef ACCESSOR
#undef DEVICE2DEVICE_COPY

#endif
