/**
 * @file huffman_coarse.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-12-17
 * (created) 2020-04-24 (rev1) 2021-09-05 (rev2) 2021-12-17
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

template <typename Huff>
void huffman_process_metadata(size_t* _counts, size_t* dev_bits, size_t nchunk, size_t& num_bits, size_t& num_uints);

template <typename Huff>
__global__ void huffman_enc_concatenate(
    Huff*   in_enc_space,
    Huff*   out_bitstream,
    size_t* sp_entries,
    size_t* sp_uints,
    size_t  chunk_size);

}  // namespace cusz
namespace cusz {

template <typename T, typename H, typename M = uint64_t>
class HuffmanCoarse : public cusz::VariableRate {
   public:
    using Origin  = T;
    using Encoded = H;
    using Mtype   = M;

   private:
    using BYTE      = uint8_t;
    using ADDR_OFST = uint32_t;

    cuszHEADER* header;
    BYTE*       dump;
    BYTE*       h_revbook;
    H*          h_bitstream;
    M*          h_bits_entries;

    float milliseconds;

    // messy
    float time_hist, time_book, time_lossless;

    uint32_t orilen;
    uint32_t nchunk, chunk_size, num_uints, revbook_nbyte;

   public:
    //
    float get_time_elapsed() const { return milliseconds; }
    float get_time_hist() const { return time_hist; }
    float get_time_book() const { return time_book; }
    float get_time_lossless() const { return time_lossless; }

    // TODO this kind of space will be overlapping with quant-codes
    size_t get_workspace_nbyte(size_t comp_in_len) const { return sizeof(H) * comp_in_len; }
    size_t get_max_output_nbyte(size_t comp_in_len) const { return sizeof(H) * comp_in_len / 2; }

    static uint32_t get_revbook_nbyte(int dict_size)
    {
        using BOOK = H;
        using SYM  = T;

        constexpr auto TYPE_BITCOUNT = sizeof(BOOK) * 8;
        return sizeof(BOOK) * (2 * TYPE_BITCOUNT) + sizeof(SYM) * dict_size;
    }

    static size_t tune_coarse_huffman_chunksize(size_t len)
    {
        int current_dev = 0;
        cudaSetDevice(current_dev);
        cudaDeviceProp dev_prop{};
        cudaGetDeviceProperties(&dev_prop, current_dev);

        auto nSM                = dev_prop.multiProcessorCount;
        auto allowed_block_dim  = dev_prop.maxThreadsPerBlock;
        auto deflate_nthread    = allowed_block_dim * nSM / HuffmanHelper::DEFLATE_CONSTANT;
        auto optimal_chunk_size = ConfigHelper::get_npart(len, deflate_nthread);
        optimal_chunk_size      = ConfigHelper::get_npart(optimal_chunk_size, HuffmanHelper::BLOCK_DIM_DEFLATE) *
                             HuffmanHelper::BLOCK_DIM_DEFLATE;

        return optimal_chunk_size;
    }

    static void get_coarse_parallelism(size_t len, int& chunksize, int& nchunk)
    {
        chunksize = HuffmanCoarse::tune_coarse_huffman_chunksize(len);
        nchunk    = ConfigHelper::get_npart(len, chunksize);
    }

   public:
    // 21-12-17 toward static method
    HuffmanCoarse() = default;

    ~HuffmanCoarse() {}

   private:
    void
    huffman_process_metadata(size_t* _counts, size_t* dev_bits, size_t nchunk, size_t& num_bits, size_t& num_uints);

    void huffman_encode_proxy1(
        H*      dev_enc_space,
        size_t* dev_bits,
        size_t* dev_uints,
        size_t* dev_entries,
        size_t* host_counts,
        T*      dev_input,
        H*      dev_book,
        size_t  len,
        int     chunk_size,
        int     dict_size,
        size_t* ptr_num_bits,
        size_t* ptr_num_uints,
        float&  milliseconds);

    void huffman_encode_proxy2(
        H*      dev_enc_space,
        size_t* dev_uints,
        size_t* dev_entries,
        H*      dev_out_bitstream,
        size_t  len,
        int     chunk_size,
        float&  milliseconds);

   public:
    void decode(
        uint32_t _orilen,
        BYTE*    _dump,
        uint32_t _chunk_size,
        uint32_t _num_uints,
        uint32_t _dict_size,
        H*       bitstream,
        M*       bits_entries,
        BYTE*    revbook,
        T*       out);

    void encode(
        H*               workspace,    // intermediate
        T*               in,           // input 1
        size_t           in_len,       // input 1 size
        uint32_t*        freq,         // input 2
        H*               book,         // input 3
        int              dict_size,    // input 2&2 size
        BYTE*            revbook,      // output 1
        Capsule<H>&      huff_data,    // output 2
        Capsule<size_t>& huff_counts,  // output 3
        int              chunk_size,   // related
        size_t&          num_bits,     // output
        size_t&          num_uints     // output
    );
};

}  // namespace cusz

#endif