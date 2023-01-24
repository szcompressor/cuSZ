/**
 * @file codec_huffman.cuh
 * @author Jiannan Tian
 * @brief Huffman kernel definitions
 * @version 0.2
 * @date 2020-02-13
 * (created) 2020-02-02, (rev1) 2021-02-13, (rev2) 2021-12-29
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_KERNEL_CODEC_HUFFMAN_CUH
#define CUSZ_KERNEL_CODEC_HUFFMAN_CUH

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "common.hh"
#include "hf/hf_bookg.hh"
#include "hf/hf_codecg.hh"
#include "hf/hf_struct.h"
#include "utils/cuda_err.cuh"
#include "utils/timer.h"

#define TIX threadIdx.x
#define BIX blockIdx.x
#define BDX blockDim.x

#if __has_include(<cub/cub.cuh>)
// #pragma message __FILE__ ": (CUDA 11 onward), cub from system path"
#include <cub/cub.cuh>
#else
// #pragma message __FILE__ ": (CUDA 10 or earlier), cub from git submodule"
#include "../../third_party/cub/cub/cub.cuh"
#endif

using BYTE = uint8_t;

extern __shared__ char __codec_huffman_uninitialized[];

struct __helper {
    __device__ __forceinline__ static unsigned int local_tid_1() { return threadIdx.x; }
    __device__ __forceinline__ static unsigned int global_tid_1() { return blockIdx.x * blockDim.x + threadIdx.x; }
    __device__ __forceinline__ static unsigned int block_stride_1() { return blockDim.x; }
    __device__ __forceinline__ static unsigned int grid_stride_1() { return blockDim.x * gridDim.x; }
    template <int SEQ>
    __device__ __forceinline__ static unsigned int global_tid()
    {
        return blockIdx.x * blockDim.x * SEQ + threadIdx.x;
    }
    template <int SEQ>
    __device__ __forceinline__ static unsigned int grid_stride()
    {
        return blockDim.x * gridDim.x * SEQ;
    }
};

template <typename UNCOMPRESSED, typename COMPRESSED, typename MetadataT>
__global__ void hf_decode_kernel(
    COMPRESSED*   compressed,
    uint8_t*      revbook,
    MetadataT*    par_nbit,
    MetadataT*    par_entry,
    int const     revbook_nbyte,
    int const     sublen,
    int const     pardeg,
    UNCOMPRESSED* out_uncompressed);

namespace asz {
namespace detail {

template <typename UNCOMPRESSED, typename ENCODED>
__global__ void hf_encode_phase1_fill(
    UNCOMPRESSED* in_uncompressed,
    size_t const  in_uncompressed_len,
    ENCODED*      in_book,
    int const     in_booklen,
    ENCODED*      out_encoded);

template <typename COMPRESSED, typename MetadataT>
__global__ void hf_encode_phase2_deflate(
    COMPRESSED*  inout_inplace,
    size_t const len,
    MetadataT*   par_nbit,
    MetadataT*   par_ncell,
    int const    sublen,
    int const    pardeg);

template <typename Huff, typename Meta>
__global__ void
hf_encode_phase4_concatenate(Huff* gapped, Meta* par_entry, Meta* par_ncell, int const cfg_sublen, Huff* non_gapped);

// TODO change size_t to unsigned int
template <typename COMPRESSED, typename UNCOMPRESSED>
__device__ void
hf_decode_single_thread_inflate(COMPRESSED* input, UNCOMPRESSED* out, int const total_bw, BYTE* revbook);

}  // namespace detail
}  // namespace asz

// TODO change size_t to unsigned int
template <typename COMPRESSED, typename UNCOMPRESSED>
__device__ void
asz::detail::hf_decode_single_thread_inflate(COMPRESSED* input, UNCOMPRESSED* out, int const total_bw, BYTE* revbook)
{
    static const auto DTYPE_WIDTH = sizeof(COMPRESSED) * 8;

    int  next_bit;
    auto idx_bit  = 0;
    auto idx_byte = 0;
    auto idx_out  = 0;

    COMPRESSED bufr = input[idx_byte];

    auto       first = reinterpret_cast<COMPRESSED*>(revbook);
    auto       entry = first + DTYPE_WIDTH;
    auto       keys  = reinterpret_cast<UNCOMPRESSED*>(revbook + sizeof(COMPRESSED) * (2 * DTYPE_WIDTH));
    COMPRESSED v     = (bufr >> (DTYPE_WIDTH - 1)) & 0x1;  // get the first bit
    auto       l     = 1;
    auto       i     = 0;

    while (i < total_bw) {
        while (v < first[l]) {  // append next i_cb bit
            ++i;
            idx_byte = i / DTYPE_WIDTH;  // [1:exclusive]
            idx_bit  = i % DTYPE_WIDTH;
            if (idx_bit == 0) {
                // idx_byte += 1; // [1:exclusive]
                bufr = input[idx_byte];
            }

            next_bit = ((bufr >> (DTYPE_WIDTH - 1 - idx_bit)) & 0x1);
            v        = (v << 1) | next_bit;
            ++l;
        }
        out[idx_out++] = keys[entry[l] + v - first[l]];
        {
            ++i;
            idx_byte = i / DTYPE_WIDTH;  // [2:exclusive]
            idx_bit  = i % DTYPE_WIDTH;
            if (idx_bit == 0) {
                // idx_byte += 1; // [2:exclusive]
                bufr = input[idx_byte];
            }

            next_bit = ((bufr >> (DTYPE_WIDTH - 1 - idx_bit)) & 0x1);
            v        = 0x0 | next_bit;
        }
        l = 1;
    }
}

template <typename UNCOMPRESSED, typename ENCODED>
__global__ void asz::detail::hf_encode_phase1_fill(
    UNCOMPRESSED* in_uncompressed,
    size_t const  in_uncompressed_len,
    ENCODED*      in_book,
    int const     in_booklen,
    ENCODED*      out_encoded)
{
    auto shmem_cb = reinterpret_cast<ENCODED*>(__codec_huffman_uninitialized);

    // load from global memory
    for (auto idx = __helper::local_tid_1();  //
         idx < in_booklen;                    //
         idx += __helper::block_stride_1())
        shmem_cb[idx] = in_book[idx];

    __syncthreads();

    for (auto idx = __helper::global_tid_1();  //
         idx < in_uncompressed_len;            //
         idx += __helper::grid_stride_1()      //
    )
        out_encoded[idx] = shmem_cb[(int)in_uncompressed[idx]];
}

template <typename COMPRESSED, typename MetadataT>
__global__ void asz::detail::hf_encode_phase2_deflate(
    COMPRESSED*  inout_inplace,
    size_t const len,
    MetadataT*   par_nbit,
    MetadataT*   par_ncell,
    int const    sublen,
    int const    pardeg)
{
    constexpr int CELL_BITWIDTH = sizeof(COMPRESSED) * 8;

    auto tid = BIX * BDX + TIX;

    if (tid * sublen < len) {
        int         residue_bits = CELL_BITWIDTH;
        int         total_bits   = 0;
        COMPRESSED* ptr          = inout_inplace + tid * sublen;
        COMPRESSED  bufr;
        uint8_t     word_width;

        auto did = tid * sublen;
        for (auto i = 0; i < sublen; i++, did++) {
            if (did == len) break;

            COMPRESSED packed_word = inout_inplace[tid * sublen + i];
            auto       word_ptr    = reinterpret_cast<struct PackedWordByWidth<sizeof(COMPRESSED)>*>(&packed_word);
            word_width             = word_ptr->bits;
            word_ptr->bits         = (uint8_t)0x0;

            if (residue_bits == CELL_BITWIDTH) {  // a new unit of compact format
                bufr = 0x0;
            }
            ////////////////////////////////////////////////////////////////

            if (word_width <= residue_bits) {
                residue_bits -= word_width;
                bufr |= packed_word << residue_bits;

                if (residue_bits == 0) {
                    residue_bits = CELL_BITWIDTH;
                    *(ptr++)     = bufr;
                }
            }
            else {
                // example: we have 5-bit code 11111 but 3 bits available in (*ptr)
                // 11111 for the residue 3 bits in (*ptr); 11111 for 2 bits of (*(++ptr)), starting with MSB
                // ^^^                                        ^^
                auto l_bits = word_width - residue_bits;
                auto r_bits = CELL_BITWIDTH - l_bits;

                bufr |= packed_word >> l_bits;
                *(ptr++) = bufr;
                bufr     = packed_word << r_bits;

                residue_bits = r_bits;
            }
            total_bits += word_width;
        }
        *ptr = bufr;  // manage the last unit

        par_nbit[tid]  = total_bits;
        par_ncell[tid] = (total_bits + CELL_BITWIDTH - 1) / CELL_BITWIDTH;
    }
}

template <typename Huff, typename Meta>
__global__ void asz::detail::hf_encode_phase4_concatenate(
    Huff*     gapped,
    Meta*     par_entry,
    Meta*     par_ncell,
    int const cfg_sublen,
    Huff*     non_gapped)
{
    auto n   = par_ncell[blockIdx.x];
    auto src = gapped + cfg_sublen * blockIdx.x;
    auto dst = non_gapped + par_entry[blockIdx.x];

    for (auto i = threadIdx.x; i < n; i += blockDim.x) {  // block-stride
        dst[i] = src[i];
    }
}

template <typename UNCOMPRESSED, typename COMPRESSED, typename MetadataT>
__global__ void hf_decode_kernel(
    COMPRESSED*   compressed,
    uint8_t*      revbook,
    MetadataT*    par_nbit,
    MetadataT*    par_entry,
    int const     revbook_nbyte,
    int const     sublen,
    int const     pardeg,
    UNCOMPRESSED* out_uncompressed)
{
    extern __shared__ uint8_t shmem[];
    constexpr auto            block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;

    auto R = (revbook_nbyte - 1 + block_dim) / block_dim;

    for (auto i = 0; i < R; i++) {
        if (TIX + i * block_dim < revbook_nbyte) shmem[TIX + i * block_dim] = revbook[TIX + i * block_dim];
    }
    __syncthreads();

    auto gid = BIX * BDX + TIX;

    if (gid < pardeg) {
        asz::detail::hf_decode_single_thread_inflate(
            compressed + par_entry[gid], out_uncompressed + sublen * gid, par_nbit[gid], shmem);
        __syncthreads();
    }
}

#endif
