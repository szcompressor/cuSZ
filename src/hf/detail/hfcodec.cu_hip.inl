/**
 * @file codec_huffman.cuh
 * @author Jiannan Tian
 * @brief Huffman kernel definitions
 * @version 0.2
 * @date 2020-02-13
 * (created) 2020-02-02, (rev1) 2021-02-13, (rev2) 2021-12-29
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_KERNEL_CODEC_HUFFMAN_CUH
#define CUSZ_KERNEL_CODEC_HUFFMAN_CUH

#include "busyheader.hh"
#include "hf/hfcodec.hh"
#include "hf/hfstruct.h"
#include "typing.hh"
#include "utils/config.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

#define TIX threadIdx.x
#define BIX blockIdx.x
#define BDX blockDim.x

using BYTE = uint8_t;

extern __shared__ char __codec_huffman_uninitialized[];

struct __helper {
  __device__ __forceinline__ static unsigned int local_tid_1()
  {
    return threadIdx.x;
  }
  __device__ __forceinline__ static unsigned int global_tid_1()
  {
    return blockIdx.x * blockDim.x + threadIdx.x;
  }
  __device__ __forceinline__ static unsigned int block_stride_1()
  {
    return blockDim.x;
  }
  __device__ __forceinline__ static unsigned int grid_stride_1()
  {
    return blockDim.x * gridDim.x;
  }
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

template <typename E, typename H, typename M>
__global__ void hf_decode_kernel(
    H* in, uint8_t* revbook, M* par_nbit, M* par_entry,
    int const revbook_nbyte, int const sublen, int const pardeg, E* out);

namespace psz {
namespace detail {

template <typename E, typename H>
__global__ void hf_encode_phase1_fill(
    E* in_uncompressed, size_t const in_uncompressed_len, H* in_book,
    int const in_booklen, H* out_encoded);

template <typename H, typename M>
__global__ void hf_encode_phase2_deflate(
    H* inout_inplace, size_t const len, M* par_nbit, M* par_ncell,
    int const sublen, int const pardeg);

template <typename H, typename M>
__global__ void hf_encode_phase4_concatenate(
    H* gapped, M* par_entry, M* par_ncell, int const cfg_sublen,
    H* non_gapped);

// TODO change size_t to unsigned int
template <typename H, typename E>
__device__ void hf_decode_single_thread_inflate(
    H* input, E* out, int const total_bw, BYTE* revbook);

}  // namespace detail
}  // namespace psz

// TODO change size_t to unsigned int
template <typename H, typename E>
__device__ void psz::detail::hf_decode_single_thread_inflate(
    H* input, E* out, int const total_bw, BYTE* revbook)
{
  constexpr auto CELL_BITWIDTH = sizeof(H) * 8;

  int next_bit;
  auto idx_bit = 0;
  auto idx_byte = 0;
  auto idx_out = 0;

  H bufr = input[idx_byte];

  auto first = reinterpret_cast<H*>(revbook);
  auto entry = first + CELL_BITWIDTH;
  auto keys = reinterpret_cast<E*>(revbook + sizeof(H) * (2 * CELL_BITWIDTH));
  H v = (bufr >> (CELL_BITWIDTH - 1)) & 0x1;  // get the first bit
  auto l = 1;
  auto i = 0;

  while (i < total_bw) {
    while (v < first[l]) {  // append next i_cb bit
      ++i;
      idx_byte = i / CELL_BITWIDTH;  // [1:exclusive]
      idx_bit = i % CELL_BITWIDTH;
      if (idx_bit == 0) {
        // idx_byte += 1; // [1:exclusive]
        bufr = input[idx_byte];
      }

      next_bit = ((bufr >> (CELL_BITWIDTH - 1 - idx_bit)) & 0x1);
      v = (v << 1) | next_bit;
      ++l;
    }
    out[idx_out++] = keys[entry[l] + v - first[l]];
    {
      ++i;
      idx_byte = i / CELL_BITWIDTH;  // [2:exclusive]
      idx_bit = i % CELL_BITWIDTH;
      if (idx_bit == 0) {
        // idx_byte += 1; // [2:exclusive]
        bufr = input[idx_byte];
      }

      next_bit = ((bufr >> (CELL_BITWIDTH - 1 - idx_bit)) & 0x1);
      v = 0x0 | next_bit;
    }
    l = 1;
  }
}

template <typename E, typename H>
__global__ void psz::detail::hf_encode_phase1_fill(
    E* in_uncompressed, size_t const in_uncompressed_len, H* in_book,
    int const in_booklen, H* out_encoded)
{
  auto shmem_cb = reinterpret_cast<H*>(__codec_huffman_uninitialized);

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

template <typename H, typename M>
__global__ void psz::detail::hf_encode_phase2_deflate(
    H* inout_inplace, size_t const len, M* par_nbit, M* par_ncell,
    int const sublen, int const pardeg)
{
  constexpr int CELL_BITWIDTH = sizeof(H) * 8;

  auto tid = BIX * BDX + TIX;

  if (tid * sublen < len) {
    int residue_bits = CELL_BITWIDTH;
    int total_bits = 0;
    H* ptr = inout_inplace + tid * sublen;
    H bufr;
    uint8_t word_width;

    auto did = tid * sublen;
    for (auto i = 0; i < sublen; i++, did++) {
      if (did == len) break;

      H packed_word = inout_inplace[tid * sublen + i];
      auto word_ptr =
          reinterpret_cast<struct PackedWordByWidth<sizeof(H)>*>(&packed_word);
      word_width = word_ptr->bits;
      word_ptr->bits = (uint8_t)0x0;

      if (residue_bits == CELL_BITWIDTH) {  // a new unit of compact format
        bufr = 0x0;
      }
      ////////////////////////////////////////////////////////////////

      if (word_width <= residue_bits) {
        residue_bits -= word_width;
        bufr |= packed_word << residue_bits;

        if (residue_bits == 0) {
          residue_bits = CELL_BITWIDTH;
          *(ptr++) = bufr;
        }
      }
      else {
        // example: we have 5-bit code 11111 but 3 bits available in (*ptr)
        // 11111 for the residue 3 bits in (*ptr); 11111 for 2 bits of
        // (*(++ptr)), starting with MSB
        // ^^^                                        ^^
        auto l_bits = word_width - residue_bits;
        auto r_bits = CELL_BITWIDTH - l_bits;

        bufr |= packed_word >> l_bits;
        *(ptr++) = bufr;
        bufr = packed_word << r_bits;

        residue_bits = r_bits;
      }
      total_bits += word_width;
    }
    *ptr = bufr;  // manage the last unit

    par_nbit[tid] = total_bits;
    par_ncell[tid] = (total_bits + CELL_BITWIDTH - 1) / CELL_BITWIDTH;
  }
}

template <typename H, typename M>
__global__ void psz::detail::hf_encode_phase4_concatenate(
    H* gapped, M* par_entry, M* par_ncell, int const cfg_sublen,
    H* non_gapped)
{
  auto n = par_ncell[blockIdx.x];
  auto src = gapped + cfg_sublen * blockIdx.x;
  auto dst = non_gapped + par_entry[blockIdx.x];

  for (auto i = threadIdx.x; i < n; i += blockDim.x) {  // block-stride
    dst[i] = src[i];
  }
}

template <typename E, typename H, typename M>
__global__ void hf_decode_kernel(
    H* in, uint8_t* revbook, M* par_nbit, M* par_entry,
    int const revbook_nbyte, int const sublen, int const pardeg, E* out)
{
  extern __shared__ uint8_t shmem[];
  constexpr auto block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;

  auto R = (revbook_nbyte - 1 + block_dim) / block_dim;

  for (auto i = 0; i < R; i++) {
    if (TIX + i * block_dim < revbook_nbyte)
      shmem[TIX + i * block_dim] = revbook[TIX + i * block_dim];
  }
  __syncthreads();

  auto gid = BIX * BDX + TIX;

  if (gid < pardeg) {
    psz::detail::hf_decode_single_thread_inflate(
        in + par_entry[gid], out + sublen * gid, par_nbit[gid], shmem);
    __syncthreads();
  }
}

#endif
