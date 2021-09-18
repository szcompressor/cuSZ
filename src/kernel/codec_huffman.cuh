/**
 * @file codec_huffman.cuh
 * @author Jiannan Tian
 * @brief Wrapper of Huffman codec.
 * @version 0.2
 * @date 2020-02-13
 * (created) 2020-02-02, (rev1) 2021-02-13
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_KERNEL_CODEC_HUFFMAN_CUH
#define CUSZ_KERNEL_CODEC_HUFFMAN_CUH

#include <stddef.h>
#include <stdint.h>
#include <cstdio>
#include <limits>

#include "../type_aliasing.hh"
#include "../type_trait.hh"

#define TIX threadIdx.x
#define BIX blockIdx.x
#define BDX blockDim.x

#if CUDART_VERSION >= 11000
// #pragma message __FILE__ ": (CUDA 11 onward), cub from system path"
#include <cub/cub.cuh>
#else
// #pragma message __FILE__ ": (CUDA 10 or earlier), cub from git submodule"
#include "../../external/cub/cub/cub.cuh"
#endif

using BYTE = uint8_t;

namespace {

template <typename T>
struct PackedWord;

template <>
struct PackedWord<UI4> {
    UI4 word : 24;
    UI4 bits : 8;
};

template <>
struct PackedWord<UI8> {
    UI8 word : 56;
    UI8 bits : 8;
};

// TODO change size_t to unsigned int
template <typename Huff, typename Output>
__device__ void InflateChunkwise(Huff* input, Output* out, size_t total_bw, BYTE* singleton)
{
    static const auto dtype_width = sizeof(Huff) * 8;

    uint8_t next_bit;
    auto    idx_bit  = 0;
    auto    idx_byte = 0;
    auto    idx_out  = 0;

    Huff bufr = input[idx_byte];

    auto first = reinterpret_cast<Huff*>(singleton);
    auto entry = first + dtype_width;
    auto keys  = reinterpret_cast<Output*>(singleton + sizeof(Huff) * (2 * dtype_width));
    Huff v     = (bufr >> (dtype_width - 1)) & 0x1;  // get the first bit
    auto l     = 1;
    auto i     = 0;
    while (i < total_bw) {
        while (v < first[l]) {  // append next i_cb bit
            ++i;
            idx_byte = i / dtype_width;  // [1:exclusive]
            idx_bit  = i % dtype_width;
            if (idx_bit == 0) {
                // idx_byte += 1; // [1:exclusive]
                bufr = input[idx_byte];
            }

            next_bit = ((bufr >> (dtype_width - 1 - idx_bit)) & 0x1);
            v        = (v << 1) | next_bit;
            ++l;
        }
        out[idx_out++] = keys[entry[l] + v - first[l]];
        {
            ++i;
            idx_byte = i / dtype_width;  // [2:exclusive]
            idx_bit  = i % dtype_width;
            if (idx_bit == 0) {
                // idx_byte += 1; // [2:exclusive]
                bufr = input[idx_byte];
            }

            next_bit = ((bufr >> (dtype_width - 1 - idx_bit)) & 0x1);
            v        = 0x0 | next_bit;
        }
        l = 1;
    }
}

}  // namespace

namespace cusz {

template <typename Input, typename Huff>
__global__ void EncodeFixedLen(Input*, Huff*, size_t, Huff*, int offset = 0);

template <typename Input, typename Huff, int SEQ = HuffmanHelper::ENC_SEQUENTIALITY>
__global__ void encode_fixedlen_space_cub(Input*, Huff*, size_t, Huff*, int offset = 0);

template <typename Huff>
__global__ void encode_deflate(Huff*, size_t, size_t*, int);

template <typename Quant, typename Huff>
__global__ void Decode(Huff*, size_t*, Quant*, size_t, int, int, BYTE*, size_t);

}  // namespace cusz

template <typename Input, typename Huff>
__global__ void cusz::EncodeFixedLen(Input* data, Huff* huff, size_t len, Huff* codebook, int offset)
{
    size_t gid = BDX * BIX + TIX;
    if (gid >= len) return;
    huff[gid] = codebook[data[gid] + offset];  // try to exploit cache?
    __syncthreads();
}

template <typename Input, typename Huff, int SEQ>
__global__ void cusz::encode_fixedlen_space_cub(Input* data, Huff* huff, size_t len, Huff* codebook, int offset)
{
    static const auto block_dim = HuffmanHelper::BLOCK_DIM_ENCODE;
    // coalesce-load (warp-striped) and transpose in shmem (similar for store)
    typedef cub::BlockLoad<Input, block_dim, SEQ, cub::BLOCK_LOAD_WARP_TRANSPOSE>  BlockLoadT_input;
    typedef cub::BlockStore<Huff, block_dim, SEQ, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT_huff;

    __shared__ union TempStorage {  // overlap shared memory space
        typename BlockLoadT_input::TempStorage load_input;
        typename BlockStoreT_huff::TempStorage store_huff;
    } temp_storage;

    Input thread_scope_data[SEQ];
    Huff  thread_scope_huff[SEQ];

    // TODO pad for potential out-of-range access
    // (BIX * BDX * SEQ) denotes the start of the data chunk that belongs to this thread block
    BlockLoadT_input(temp_storage.load_input).Load(data + (BIX * BDX) * SEQ, thread_scope_data);
    __syncthreads();  // barrier for shmem reuse

#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
        auto id              = (BIX * BDX + TIX) * SEQ + i;
        thread_scope_huff[i] = id < len ? codebook[thread_scope_data[i] + offset] : 0;
    }
    __syncthreads();

    BlockStoreT_huff(temp_storage.store_huff).Store(huff + (BIX * BDX) * SEQ, thread_scope_huff);
}

template <typename Huff>
__global__ void cusz::encode_deflate(Huff* huff, size_t len, size_t* sp_meta, int chunk_size)
{
    // TODO static check with Huff and UI4/8
    static const auto dtype_width = sizeof(Huff) * 8;

    auto gid = BIX * BDX + TIX;
    if (gid >= ((len + chunk_size - 1) / chunk_size)) return;

    size_t  residue_bits = sizeof(Huff) * 8, total_bits = 0;
    Huff*   ptr = huff + gid * chunk_size;
    Huff    packed_word, bufr;
    uint8_t word_width;

    for (auto i = 0; i < chunk_size; i++) {
        packed_word    = huff[gid * chunk_size + i];
        auto word_ptr  = reinterpret_cast<struct PackedWord<Huff>*>(&packed_word);
        word_width     = word_ptr->bits;
        word_ptr->bits = (uint8_t)0x0;

        if (residue_bits == dtype_width) {  // a new unit of compact format
            bufr = 0x0;
        }
        ////////////////////////////////////////////////////////////////

        if (word_width <= residue_bits) {
            residue_bits -= word_width;
            bufr |= packed_word << residue_bits;

            if (residue_bits == 0) {
                residue_bits = dtype_width;
                *(ptr++)     = bufr;
            }
        }
        else {
            // example: we have 5-bit code 11111 but 3 bits available in (*ptr)
            // 11111 for the residue 3 bits in (*ptr); 11111 for 2 bits of (*(++ptr)), starting with MSB
            // ^^^                                        ^^
            auto l_bits = word_width - residue_bits;
            auto r_bits = dtype_width - l_bits;

            bufr |= packed_word >> l_bits;
            *(ptr++) = bufr;
            bufr     = packed_word << r_bits;

            residue_bits = r_bits;
        }
        total_bits += word_width;
    }
    *ptr = bufr;  // manage the last unit

    *(sp_meta + gid) = total_bits;
}

template <typename Quant, typename Huff>
__global__ void cusz::Decode(
    Huff*   sp_huff,
    size_t* sp_meta,
    Quant*  quant_out,
    size_t  len,
    int     chunk_size,
    int     n_chunk,
    BYTE*   singleton,
    size_t  singleton_size)
{
    extern __shared__ BYTE _s_singleton[];
    static const auto      block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;

    for (auto i = 0; i < (singleton_size - 1 + block_dim) / block_dim; i++) {
        if (TIX + i * block_dim < singleton_size) _s_singleton[TIX + i * block_dim] = singleton[TIX + i * block_dim];
    }
    __syncthreads();

    auto bits         = sp_meta;
    auto UInt_entries = sp_meta + n_chunk;

    auto chunk_id = BIX * BDX + TIX;

    if (chunk_id >= n_chunk) return;

    InflateChunkwise(sp_huff + UInt_entries[chunk_id], quant_out + chunk_size * chunk_id, bits[chunk_id], _s_singleton);
    __syncthreads();
};

template <typename Output, typename Huff, typename MetadataT>
__global__ void huffman_decode_kernel(
    Huff*        bitstream,
    MetadataT*   seg_entries,
    MetadataT*   seg_bits,
    Output*      output,
    unsigned int chunk_size,
    unsigned int nchunk,
    BYTE*        revbook,
    unsigned int revbook_len)
{
    extern __shared__ BYTE shmem[];
    constexpr auto         dim_block = 256;

    auto n = (revbook_len - 1 + dim_block) / dim_block;
    for (auto i = 0; i < n; i++) {
        if (TIX + i * dim_block < revbook_len) shmem[TIX + i * dim_block] = revbook[TIX + i * dim_block];
    }
    __syncthreads();

    auto id = BIX * BDX + TIX;
    if (id < nchunk) {
        InflateChunkwise(bitstream + seg_entries[id], output + chunk_size * id, seg_bits[id], shmem);
        __syncthreads();
    }
};

#endif