/**
 * @file huffman_codec.cu
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

#include <stddef.h>
#include <stdint.h>
#include <cstdio>
#include <limits>

#include "huff_codec.cuh"
#include "type_aliasing.hh"
#include "type_trait.hh"

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
#include "../external/cub/cub/cub.cuh"
#endif

template <typename Input, typename Huff>
__global__ void lossless::wrapper::EncodeFixedLen(Input* data, Huff* huff, size_t len, Huff* codebook, int offset)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= len) return;
    huff[gid] = codebook[data[gid] + offset];  // try to exploit cache?
    __syncthreads();
}

template <typename Input, typename Huff, int Sequentiality>
__global__ void lossless::wrapper::EncodeFixedLen_cub(Input* data, Huff* huff, size_t len, Huff* codebook, int offset)
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
__global__ void lossless::wrapper::Deflate(
    Huff*   h_in_out,  //
    size_t  len,
    size_t* densely_meta,
    int     PART_SIZE)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (len - 1) / PART_SIZE + 1) return;
    uint8_t bitwidth;
    size_t  densely_coded_lsb_pos = sizeof(Huff) * 8, total_bitwidth = 0;
    size_t  ending = (gid + 1) * PART_SIZE <= len ? PART_SIZE : len - gid * PART_SIZE;
    //    if ((gid + 1) * PART_SIZE > len) printf("\n\ngid %lu\tending %lu\n\n", gid, ending);
    Huff  msb_bw_word_lsb, _1, _2;
    Huff* current = h_in_out + gid * PART_SIZE;
    for (size_t i = 0; i < ending; i++) {
        msb_bw_word_lsb = h_in_out[gid * PART_SIZE + i];
        bitwidth        = *((uint8_t*)&msb_bw_word_lsb + (sizeof(Huff) - 1));

        *((uint8_t*)&msb_bw_word_lsb + sizeof(Huff) - 1) = 0x0;
        if (densely_coded_lsb_pos == sizeof(Huff) * 8) *current = 0x0;  // a new unit of data type
        if (bitwidth <= densely_coded_lsb_pos) {
            densely_coded_lsb_pos -= bitwidth;
            *current |= msb_bw_word_lsb << densely_coded_lsb_pos;
            if (densely_coded_lsb_pos == 0) {
                densely_coded_lsb_pos = sizeof(Huff) * 8;
                ++current;
            }
        }
        else {
            // example: we have 5-bit code 11111 but 3 bits left for (*current)
            // we put first 3 bits of 11111 to the last 3 bits of (*current)
            // and put last 2 bits from MSB of (*(++current))
            // the comment continues with the example
            _1 = msb_bw_word_lsb >> (bitwidth - densely_coded_lsb_pos);
            _2 = msb_bw_word_lsb << (sizeof(Huff) * 8 - (bitwidth - densely_coded_lsb_pos));
            *current |= _1;
            *(++current) = 0x0;
            *current |= _2;
            densely_coded_lsb_pos = sizeof(Huff) * 8 - (bitwidth - densely_coded_lsb_pos);
        }
        total_bitwidth += bitwidth;
    }
    *(densely_meta + gid) = total_bitwidth;
}

// TODO change size_t to unsigned int
template <typename Huff, typename Output>
__device__ void
lossless::wrapper::InflateChunkwise(Huff* in_huff, Output* out_quant, size_t total_bw, uint8_t* singleton)
{
    uint8_t next_bit;
    size_t  idx_bit;
    size_t  idx_byte   = 0;
    size_t  idx_bcoded = 0;
    auto    first      = reinterpret_cast<Huff*>(singleton);
    auto    entry      = first + sizeof(Huff) * 8;
    auto    keys       = reinterpret_cast<Output*>(singleton + sizeof(Huff) * (2 * sizeof(Huff) * 8));
    Huff    v          = (in_huff[idx_byte] >> (sizeof(Huff) * 8 - 1)) & 0x1;  // get the first bit
    size_t  l          = 1;
    size_t  i          = 0;
    while (i < total_bw) {
        while (v < first[l]) {  // append next i_cb bit
            ++i;
            idx_byte = i / (sizeof(Huff) * 8);
            idx_bit  = i % (sizeof(Huff) * 8);
            next_bit = ((in_huff[idx_byte] >> (sizeof(Huff) * 8 - 1 - idx_bit)) & 0x1);
            v        = (v << 1) | next_bit;
            ++l;
        }
        out_quant[idx_bcoded++] = keys[entry[l] + v - first[l]];
        {
            ++i;
            idx_byte = i / (sizeof(Huff) * 8);
            idx_bit  = i % (sizeof(Huff) * 8);
            next_bit = ((in_huff[idx_byte] >> (sizeof(Huff) * 8 - 1 - idx_bit)) & 0x1);
            v        = 0x0 | next_bit;
        }
        l = 1;
    }
}

template <typename Quant, typename Huff>
__global__ void lossless::wrapper::Decode(
    Huff*    densely,     //
    size_t*  dH_meta,     //
    Quant*   q_out,       //
    size_t   len,         //
    int      chunk_size,  //
    int      n_chunk,
    uint8_t* singleton,
    size_t   singleton_size)
{
    extern __shared__ uint8_t _s_singleton[];
    if (threadIdx.x == 0) memcpy(_s_singleton, singleton, singleton_size);
    __syncthreads();

    auto dH_bit_meta   = dH_meta;
    auto dH_uInt_entry = dH_meta + n_chunk;

    size_t chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    // if (chunk_id == 0) printf("n_chunk: %lu\n", n_chunk);
    if (chunk_id >= n_chunk) return;

    InflateChunkwise(                       //
        densely + dH_uInt_entry[chunk_id],  //
        q_out + chunk_size * chunk_id,      //
        dH_bit_meta[chunk_id],              //
        _s_singleton);
    __syncthreads();
};

template __global__ void lossless::wrapper::EncodeFixedLen<UI1, UI4>(UI1*, UI4*, size_t, UI4*, int);
template __global__ void lossless::wrapper::EncodeFixedLen<UI1, UI8>(UI1*, UI8*, size_t, UI8*, int);
template __global__ void lossless::wrapper::EncodeFixedLen<UI2, UI4>(UI2*, UI4*, size_t, UI4*, int);
template __global__ void lossless::wrapper::EncodeFixedLen<UI2, UI8>(UI2*, UI8*, size_t, UI8*, int);
template __global__ void lossless::wrapper::EncodeFixedLen<I1, UI4>(I1*, UI4*, size_t, UI4*, int);
template __global__ void lossless::wrapper::EncodeFixedLen<I1, UI8>(I1*, UI8*, size_t, UI8*, int);
template __global__ void lossless::wrapper::EncodeFixedLen<I2, UI4>(I2*, UI4*, size_t, UI4*, int);
template __global__ void lossless::wrapper::EncodeFixedLen<I2, UI8>(I2*, UI8*, size_t, UI8*, int);

template __global__ void lossless::wrapper::EncodeFixedLen_cub<UI1, UI4>(UI1*, UI4*, size_t, UI4*, int);
template __global__ void lossless::wrapper::EncodeFixedLen_cub<UI1, UI8>(UI1*, UI8*, size_t, UI8*, int);
template __global__ void lossless::wrapper::EncodeFixedLen_cub<UI2, UI4>(UI2*, UI4*, size_t, UI4*, int);
template __global__ void lossless::wrapper::EncodeFixedLen_cub<UI2, UI8>(UI2*, UI8*, size_t, UI8*, int);
template __global__ void lossless::wrapper::EncodeFixedLen_cub<I1, UI4>(I1*, UI4*, size_t, UI4*, int);
template __global__ void lossless::wrapper::EncodeFixedLen_cub<I1, UI8>(I1*, UI8*, size_t, UI8*, int);
template __global__ void lossless::wrapper::EncodeFixedLen_cub<I2, UI4>(I2*, UI4*, size_t, UI4*, int);
template __global__ void lossless::wrapper::EncodeFixedLen_cub<I2, UI8>(I2*, UI8*, size_t, UI8*, int);

template __global__ void lossless::wrapper::Deflate<UI4>(UI4*, size_t, size_t*, int);
template __global__ void lossless::wrapper::Deflate<UI8>(UI8*, size_t, size_t*, int);

template __device__ void lossless::wrapper::InflateChunkwise<UI4, UI1>(UI4*, UI1*, size_t, UI1*);
template __device__ void lossless::wrapper::InflateChunkwise<UI4, UI2>(UI4*, UI2*, size_t, UI1*);
template __device__ void lossless::wrapper::InflateChunkwise<UI8, UI1>(UI8*, UI1*, size_t, UI1*);
template __device__ void lossless::wrapper::InflateChunkwise<UI8, UI2>(UI8*, UI2*, size_t, UI1*);

template __global__ void lossless::wrapper::Decode<UI1, UI4>(UI4*, size_t*, UI1*, size_t, int, int, UI1*, size_t);
template __global__ void lossless::wrapper::Decode<UI1, UI8>(UI8*, size_t*, UI1*, size_t, int, int, UI1*, size_t);

template __global__ void lossless::wrapper::Decode<UI2, UI4>(UI4*, size_t*, UI2*, size_t, int, int, UI1*, size_t);
template __global__ void lossless::wrapper::Decode<UI2, UI8>(UI8*, size_t*, UI2*, size_t, int, int, UI1*, size_t);
