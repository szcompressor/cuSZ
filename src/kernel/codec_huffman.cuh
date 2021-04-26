/**
 * @file codec_huffman.cuh
 * @author Jiannan Tian
 * @brief Wrapper of Huffman codec.
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-02-02
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef KERNEL_HUFF_CODEC
#define KERNEL_HUFF_CODEC

#include <stddef.h>
#include <stdint.h>
#include <cstdio>
#include <limits>

#include "../type_aliasing.hh"
#include "../type_trait.hh"

#define tix threadIdx.x
#define tiy threadIdx.y
#define tiz threadIdx.z
#define bix blockIdx.x
#define biy blockIdx.y
#define biz blockIdx.z
#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

#if CUDART_VERSION >= 11000
#pragma message(__FILE__ ": (CUDA 11 onward), cub from system path")
#include <cub/cub.cuh>
#else
#pragma message(__FILE__ ": (CUDA 10 or earlier), cub from git submodule")
#include "../../external/cub/cub/cub.cuh"
#endif

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

namespace kernel {

template <typename Input, typename Huff>
__global__ void EncodeFixedLen(Input*, Huff*, size_t, Huff*, int offset = 0);

template <typename Input, typename Huff, int Sequentiality = HuffConfig::enc_sequentiality>
__global__ void EncodeFixedLen_cub(Input*, Huff*, size_t, Huff*, int offset = 0);

template <typename Huff>
__global__ void Deflate(Huff*, size_t, size_t*, int);

template <typename Huff, typename Output>
__device__ void InflateChunkwise(Huff*, Output*, size_t, uint8_t*);

template <typename Quant, typename Huff>
__global__ void Decode(Huff*, size_t*, Quant*, size_t, int, int, uint8_t*, size_t);

}  // namespace kernel

template <typename Input, typename Huff>
__global__ void kernel::EncodeFixedLen(Input* data, Huff* huff, size_t len, Huff* codebook, int offset)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= len) return;
    huff[gid] = codebook[data[gid] + offset];  // try to exploit cache?
    __syncthreads();
}

template <typename Input, typename Huff, int Sequentiality>
__global__ void kernel::EncodeFixedLen_cub(Input* data, Huff* huff, size_t len, Huff* codebook, int offset)
{
    static const auto Db = HuffConfig::Db_encode;
    // coalesce-load (warp-striped) and transpose in shmem (similar for store)
    typedef cub::BlockLoad<Input, Db, Sequentiality, cub::BLOCK_LOAD_WARP_TRANSPOSE>  BlockLoadT_input;
    typedef cub::BlockStore<Huff, Db, Sequentiality, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT_huff;

    __shared__ union TempStorage {  // overlap shared memory space
        typename BlockLoadT_input::TempStorage load_input;
        typename BlockStoreT_huff::TempStorage store_huff;
    } temp_storage;

    Input thread_scope_data[Sequentiality];
    Huff  thread_scope_huff[Sequentiality];

    // TODO pad for potential out-of-range access
    // (bix * bdx * Sequentiality) denotes the start of the data chunk that belongs to this thread block
    BlockLoadT_input(temp_storage.load_input).Load(data + (bix * bdx) * Sequentiality, thread_scope_data);
    __syncthreads();  // barrier for shmem reuse

#pragma unroll
    for (auto i = 0; i < Sequentiality; i++) {
        auto id              = (bix * bdx + tix) * Sequentiality + i;
        thread_scope_huff[i] = id < len ? codebook[thread_scope_data[i] + offset] : 0;
    }
    __syncthreads();

    BlockStoreT_huff(temp_storage.store_huff).Store(huff + (bix * bdx) * Sequentiality, thread_scope_huff);
}

template <typename Huff>
__global__ void kernel::Deflate(Huff* huff, size_t len, size_t* sp_meta, int chunk_size)
{
    // TODO static check with Huff and UI4/8
    static const auto dtype_width = sizeof(Huff) * 8;

    auto gid = blockIdx.x * blockDim.x + threadIdx.x;
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

// TODO change size_t to unsigned int
template <typename Huff, typename Output>
__device__ void kernel::InflateChunkwise(Huff* input, Output* out, size_t total_bw, uint8_t* singleton)
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

template <typename Quant, typename Huff>
__global__ void kernel::Decode(
    Huff*    sp_huff,
    size_t*  sp_meta,
    Quant*   quant_out,
    size_t   len,
    int      chunk_size,
    int      n_chunk,
    uint8_t* singleton,
    size_t   singleton_size)
{
    extern __shared__ uint8_t _s_singleton[];
    static const auto         Db = HuffConfig::Db_deflate;

    for (auto i = 0; i < (singleton_size - 1 + Db) / Db; i++) {
        if (tix + i * Db < singleton_size) _s_singleton[tix + i * Db] = singleton[tix + i * Db];
    }
    __syncthreads();

    auto bits         = sp_meta;
    auto UInt_entries = sp_meta + n_chunk;

    auto chunk_id = bix * bdx + tix;

    if (chunk_id >= n_chunk) return;

    InflateChunkwise(sp_huff + UInt_entries[chunk_id], quant_out + chunk_size * chunk_id, bits[chunk_id], _s_singleton);
    __syncthreads();
};

#endif