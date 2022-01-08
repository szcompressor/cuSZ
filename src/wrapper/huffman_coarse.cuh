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
#include "../utils/cuda_err.cuh"
#include "huffman_parbook.cuh"

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
#define DEFINE_HC_ARRAY(VAR, TYPE) \
    TYPE* d_##VAR{nullptr};        \
    TYPE* h_##VAR{nullptr};

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

    void allocate_workspace(size_t const in_uncompressed_len, int cfg_booklen, int cfg_pardeg)
    {
        auto max_compressed_bytes = [&]() { return in_uncompressed_len / 2 * sizeof(H); };

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

#define HC_ALLOCHOST(VAR, SYM)                     \
    cudaMallocHost(&h_##VAR, rte.nbyte[RTE::SYM]); \
    memset(h_##VAR, 0x0, rte.nbyte[RTE::SYM]);

#define HC_ALLOCDEV(VAR, SYM)                  \
    cudaMalloc(&d_##VAR, rte.nbyte[RTE::SYM]); \
    cudaMemset(d_##VAR, 0x0, rte.nbyte[RTE::SYM]);

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

#undef HC_ALLOCDEV
#undef HC_ALLOCHOST
    }

   private:
    using BOOK = H;
    using SYM  = T;

    static const int CELL_BITWIDTH = sizeof(H) * 8;

    float milliseconds;
    float time_hist, time_book, time_lossless;

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
#define HC_FREEHOST(VAR, SYM)  \
    if (h_##VAR) {             \
        cudaFreeHost(h_##VAR); \
        h_##VAR = nullptr;     \
    }

#define HC_FREEDEV(VAR, SYM) \
    if (d_##VAR) {           \
        cudaFree(d_##VAR);   \
        d_##VAR = nullptr;   \
    }

        HC_FREEDEV(tmp, TMP);
        HC_FREEDEV(freq, FREQ);
        HC_FREEDEV(book, BOOK);
        HC_FREEDEV(revbook, REVBOOK);
        HC_FREEDEV(par_nbit, PAR_NBIT);
        HC_FREEDEV(par_ncell, PAR_NCELL);
        HC_FREEDEV(par_entry, PAR_ENTRY);
        HC_FREEDEV(bitstream, BITSTREAM);

        HC_FREEHOST(freq, FREQ);
        HC_FREEHOST(book, BOOK);
        HC_FREEHOST(revbook, REVBOOK);
        HC_FREEHOST(par_nbit, PAR_NBIT);
        HC_FREEHOST(par_ncell, PAR_NCELL);
        HC_FREEHOST(par_entry, PAR_ENTRY);

#undef HC_FREEDEV
#undef HC_FREEHOST
    }

   private:
    void inspect(
        cusz::FREQ*  tmp_freq,
        H*           tmp_book,
        T*           in_data,
        size_t const in_len,
        int const    cfg_booklen,
        BYTE*        out_revbook,
        cudaStream_t = nullptr);

    void encode_phase1(
        H*           tmp_book,
        H*           tmp_encspace,
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        cudaStream_t stream = nullptr);

    void encode_phase2(
        H*           tmp_encspace,
        size_t const in_uncompressed_len,
        int const    cfg_sublen,
        int const    cfg_pardeg,
        M*           par_nbit,
        M*           par_ncell,
        cudaStream_t stream = nullptr);

    void encode_phase3(
        M*           in_meta_deviceview,
        int const    cfg_pardeg,
        M*           out_meta_hostview,
        size_t&      out_total_nbit,
        size_t&      out_total_ncell,
        cudaStream_t stream = nullptr);

    void encode_phase4(
        H*           tmp_encspace,
        size_t const in_uncompressed_len,
        int const    cfg_sublen,
        int const    cfg_pardeg,
        M*           out_compressed_meta,
        H*           out_compressed,
        cudaStream_t stream = nullptr);

   public:
    void encode(
        cusz::FREQ*  tmp_freq,
        H*           tmp_book,
        H*           tmp_encspace,
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        int const    cfg_sublen,
        BYTE*        out_revbook,
        Capsule<H>&  out_compressed,
        Capsule<M>&  out_compressed_meta,
        size_t&      out_total_nbit,
        size_t&      out_total_ncell,
        cudaStream_t = nullptr);

    void encode_integrated(
        cusz::FREQ*  tmp_freq,
        H*           tmp_book,
        H*           tmp_encspace,
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        int const    cfg_sublen,
        BYTE*        out_revbook,
        Capsule<H>&  out_compressed,
        Capsule<M>&  out_compressed_meta,
        size_t&      out_total_nbit,
        size_t&      out_total_ncell,
        cudaStream_t = nullptr);

    void decode(
        H*           in_compressed,
        M*           in_compressed_meta,
        BYTE*        in_revbook,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        int const    cfg_sublen,
        T*           out_uncompressed,
        cudaStream_t = nullptr);

    // new encoding
    void encode_new(
        T*           in_uncompressed,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        int const    cfg_sublen,
        BYTE*&       out_compressed,
        size_t&      out_compressed_len,
        cudaStream_t stream = nullptr);

    void decode_new(
        BYTE*        in_compressed,  //
        T*           out_uncompressed,
        cudaStream_t stream           = nullptr,
        bool         header_on_device = true);

    // end of class definition
};

}  // namespace cusz

#endif
