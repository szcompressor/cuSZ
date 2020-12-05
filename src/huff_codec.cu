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

using uint8__t = uint8_t;

template <typename Input, typename Huff>
__global__ void lossless::wrapper::EncodeFixedLen(Input* d, Huff* h, size_t dlen, Huff* codebook)
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= dlen) return;
    h[gid] = codebook[d[gid]];  // try to exploit cache?
    __syncthreads();
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

template __global__ void lossless::wrapper::EncodeFixedLen<UI1, UI4>(UI1*, UI4*, size_t, UI4*);
template __global__ void lossless::wrapper::EncodeFixedLen<UI1, UI8>(UI1*, UI8*, size_t, UI8*);
template __global__ void lossless::wrapper::EncodeFixedLen<UI1, UI8_2>(UI1*, UI8_2*, size_t, UI8_2*);

template __global__ void lossless::wrapper::EncodeFixedLen<UI2, UI4>(UI2*, UI4*, size_t, UI4*);
template __global__ void lossless::wrapper::EncodeFixedLen<UI2, UI8>(UI2*, UI8*, size_t, UI8*);
template __global__ void lossless::wrapper::EncodeFixedLen<UI2, UI8_2>(UI2*, UI8_2*, size_t, UI8_2*);

template __global__ void lossless::wrapper::EncodeFixedLen<UI4, UI4>(UI4*, UI4*, size_t, UI4*);
template __global__ void lossless::wrapper::EncodeFixedLen<UI4, UI8>(UI4*, UI8*, size_t, UI8*);
template __global__ void lossless::wrapper::EncodeFixedLen<UI4, UI8_2>(UI4*, UI8_2*, size_t, UI8_2*);

template __global__ void lossless::wrapper::Deflate<UI4>(UI4*, size_t, size_t*, int);
template __global__ void lossless::wrapper::Deflate<UI8>(UI8*, size_t, size_t*, int);
template __global__ void lossless::wrapper::Deflate<UI8_2>(UI8_2*, size_t, size_t*, int);

template __device__ void lossless::wrapper::InflateChunkwise<UI4, UI1>(UI4*, UI1*, size_t, UI1*);
template __device__ void lossless::wrapper::InflateChunkwise<UI4, UI2>(UI4*, UI2*, size_t, UI1*);
template __device__ void lossless::wrapper::InflateChunkwise<UI4, UI4>(UI4*, UI4*, size_t, UI1*);
template __device__ void lossless::wrapper::InflateChunkwise<UI8, UI1>(UI8*, UI1*, size_t, UI1*);
template __device__ void lossless::wrapper::InflateChunkwise<UI8, UI2>(UI8*, UI2*, size_t, UI1*);
template __device__ void lossless::wrapper::InflateChunkwise<UI8, UI4>(UI8*, UI4*, size_t, UI1*);
template __device__ void lossless::wrapper::InflateChunkwise<UI8_2, UI1>(UI8_2*, UI1*, size_t, UI1*);
template __device__ void lossless::wrapper::InflateChunkwise<UI8_2, UI2>(UI8_2*, UI2*, size_t, UI1*);
template __device__ void lossless::wrapper::InflateChunkwise<UI8_2, UI4>(UI8_2*, UI4*, size_t, UI1*);

template __global__ void lossless::wrapper::Decode<UI1, UI4>(UI4*, size_t*, UI1*, size_t, int, int, UI1*, size_t);
template __global__ void lossless::wrapper::Decode<UI1, UI8>(UI8*, size_t*, UI1*, size_t, int, int, UI1*, size_t);
template __global__ void lossless::wrapper::Decode<UI1, UI8_2>(UI8_2*, size_t*, UI1*, size_t, int, int, UI1*, size_t);

template __global__ void lossless::wrapper::Decode<UI2, UI4>(UI4*, size_t*, UI2*, size_t, int, int, UI1*, size_t);
template __global__ void lossless::wrapper::Decode<UI2, UI8>(UI8*, size_t*, UI2*, size_t, int, int, UI1*, size_t);
template __global__ void lossless::wrapper::Decode<UI2, UI8_2>(UI8_2*, size_t*, UI2*, size_t, int, int, UI1*, size_t);

template __global__ void lossless::wrapper::Decode<UI4, UI4>(UI4*, size_t*, UI4*, size_t, int, int, UI1*, size_t);
template __global__ void lossless::wrapper::Decode<UI4, UI8>(UI8*, size_t*, UI4*, size_t, int, int, UI1*, size_t);
template __global__ void lossless::wrapper::Decode<UI4, UI8_2>(UI8_2*, size_t*, UI4*, size_t, int, int, UI1*, size_t);
