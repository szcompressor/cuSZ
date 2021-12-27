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

#include "../../include/reducer.hh"
#include "../common/definition.hh"
#include "../common/type_traits.hh"
#include "../header.hh"
#include "huffman_parbook.cuh"
// #include "huffman_coarse.cuh"

namespace cusz {

template <typename T, typename H, typename M = uint32_t>
class HuffmanCoarse : public cusz::VariableRate {
   public:
    using Origin    = T;
    using Encoded   = H;
    using MetadataT = M;

    /**
     * @brief on host; separate from data fields
     * otherwise, aligning to 128B can be unwanted
     *
     */
    struct header_t {
        // length determined on start
        uint32_t revbook;
        uint32_t par_meta;
        // varying length
        uint32_t par_data;
    } header;

    Capsule<uint8_t> tmpspace;
    Capsule<uint8_t> datafield;

   private:
    using BYTE      = uint8_t;
    using ADDR_OFST = uint32_t;
    using BOOK      = H;
    using SYM       = T;

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

    ~HuffmanCoarse() {}

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

    void decode(
        H*           in_compressed,
        M*           in_compressed_meta,
        BYTE*        in_revbook,
        size_t const in_uncompressed_len,
        int const    cfg_booklen,
        int const    cfg_sublen,
        T*           out_uncompressed,
        cudaStream_t = nullptr);

    // end of class definition
};

}  // namespace cusz

#endif