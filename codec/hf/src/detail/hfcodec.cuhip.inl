/**
 * @file hfcodec.cuhip.inl
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

#include <cstdio>

#include "hfclass.hh"  // contains HuffmanHelper; TODO put in another file
#include "hfword.hh"

#define TIX threadIdx.x
#define BIX blockIdx.x
#define BDX blockDim.x

using BYTE = uint8_t;

extern __shared__ char __codec_raw[];

namespace {
struct helper {
  __device__ __forceinline__ static unsigned int local_tid_1() { return threadIdx.x; }
  __device__ __forceinline__ static unsigned int global_tid_1()
  {
    return blockIdx.x * blockDim.x + threadIdx.x;
  }
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

}  // namespace

namespace phf::experimental {

// a duplicate from psz
template <typename T, typename M = u4>
__global__ void KERNEL_CUHIP_scatter(T* val, M* idx, int const n, T* out)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    int dst_idx = idx[tid];
    out[dst_idx] = val[tid];
  }
}

// TODO totally disable H (no need to be other than uint32_t)
// TODO kernel wrapper
template <typename E, typename H>
__global__ void KERNEL_CUHIP_encode_phase1_fill_with_filter(
    /* input */ E* in, size_t const in_len, H* in_bk, int const in_bklen,
    H const replacement,  //
    /* output */ H* encoded, E* outlier_val, uint32_t* outlier_idx, uint32_t* outlier_num)
{
  auto s_bk = reinterpret_cast<uint32_t*>(__codec_raw);

  // load from global memory
  for (auto idx = helper::local_tid_1();  //
       idx < in_bklen;                    //
       idx += helper::block_stride_1())
    s_bk[idx] = in_bk[idx];

  __syncthreads();

  for (auto idx = helper::global_tid_1(); idx < in_len; idx += helper::grid_stride_1()) {
    auto candidate = s_bk[(int)in[idx]];
    auto pw4 = reinterpret_cast<HuffmanWord<4>*>(&candidate);

    if (pw4->bitcount == pw4->OUTLIER_CUTOFF) {
      encoded[idx] = replacement;
      auto atomic_old_loc = atomicAdd(outlier_num, 1);
      outlier_val[atomic_old_loc] = in[idx];
      outlier_idx[atomic_old_loc] = idx;
      printf("inside kernel; hf outlier; atomic_old_loc: %d\n", atomic_old_loc);
    }
    else {
      encoded[idx] = candidate;
    }
  }
}

template <typename E, typename H, typename M = uint32_t>
__global__ void KERNEL_CUHIP_encode_phase1_fill_collect_metadata(
    E* in, size_t const in_len, H* in_bk, int const in_bklen, int const sublen, int const pardeg,
    int const repeat, H* encoded, M* par_nbit, M* par_ncell)
{
  using PW = HuffmanWord<sizeof(H)>;
  auto s_bk = reinterpret_cast<H*>(__codec_raw);
  // for one sublen of pts
  __shared__ uint32_t nbit;

  // load codebook
  for (auto i = threadIdx.x; i < in_bklen; i += blockDim.x) { s_bk[i] = in_bk[i]; }
  __syncthreads();

  for (auto n = 0; n < repeat; n++) {
    uint32_t p_nbit{0};
    if (threadIdx.x == 0) nbit = 0;
    __syncthreads();

    auto block_base = blockIdx.x * (repeat * sublen);
    auto part_id = block_base / (n * sublen);

    for (auto i = threadIdx.x; i < sublen; i += blockDim.x) {
      auto gid = i + (n * sublen) + block_base;
      auto word = s_bk[(int)in[gid]];
      encoded[gid] = word;
      p_nbit += ((PW*)&word)->bitcount;
    }
    atomicAdd(&nbit, p_nbit);
    __syncthreads();

    if (threadIdx.x == 0) {  //
      auto ncell = (nbit - 1) / (sizeof(H) * 8) + 1;
      if (part_id < pardeg) {
        par_nbit[part_id] = nbit;
        par_ncell[part_id] = ncell;
      }
    }

    __syncthreads();
  }
}

}  // namespace phf::experimental

namespace phf {

template <typename E, typename H>
__global__ void KERNEL_CUHIP_encode_phase1_fill(
    E* in, size_t const in_len, H* in_bk, int const in_bklen, H* out_encoded)
{
  auto s_bk = reinterpret_cast<H*>(__codec_raw);

  // load from global memory
  for (auto idx = helper::local_tid_1();  //
       idx < in_bklen;                    //
       idx += helper::block_stride_1())
    s_bk[idx] = in_bk[idx];

  __syncthreads();

  for (auto idx = helper::global_tid_1();  //
       idx < in_len;                       //
       idx += helper::grid_stride_1()      //
  )
    out_encoded[idx] = s_bk[(int)in[idx]];
}

template <typename H, typename M>
__global__ void KERNEL_CUHIP_encode_phase2_deflate(
    H* inout_inplace, size_t const len, M* par_nbit, M* par_ncell, int const sublen,
    int const pardeg)
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
      auto word_ptr = reinterpret_cast<struct HuffmanWord<sizeof(H)>*>(&packed_word);
      word_width = word_ptr->bitcount;
      word_ptr->bitcount = (uint8_t)0x0;

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
__global__ void KERNEL_CUHIP_encode_phase4_concatenate(
    H* gapped, M* par_entry, M* par_ncell, int const cfg_sublen, H* non_gapped)
{
  auto n = par_ncell[blockIdx.x];
  auto src = gapped + cfg_sublen * blockIdx.x;
  auto dst = non_gapped + par_entry[blockIdx.x];

  for (auto i = threadIdx.x; i < n; i += blockDim.x) {  // block-stride
    dst[i] = src[i];
  }
}

}  // namespace phf

namespace phf {

template <typename E, typename H, typename M>
__global__ void KERNEL_CUHIP_HF_decode(
    H* in, uint8_t* revbook, M* par_nbit, M* par_entry, int const revbook_nbyte, int const sublen,
    int const pardeg, E* out)
{
  constexpr auto CELL_BITWIDTH = sizeof(H) * 8;
  extern __shared__ uint8_t s_revbook[];
  constexpr auto block_dim = phf::HuffmanHelper::BLOCK_DIM_DEFLATE;

  auto single_thread_inflate = [&](H* input, E* out, int const total_bw) {
    int next_bit;
    auto idx_bit = 0, idx_byte = 0, idx_out = 0;
    H bufr = input[idx_byte];
    auto first = (H*)(s_revbook);
    auto entry = first + CELL_BITWIDTH;
    auto keys = (E*)(s_revbook + sizeof(H) * (2 * CELL_BITWIDTH));
    H v = (bufr >> (CELL_BITWIDTH - 1)) & 0x1;  // get the first bit
    auto l = 1, i = 0;

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
  };

  auto R = (revbook_nbyte - 1 + block_dim) / block_dim;

  for (auto i = 0; i < R; i++) {
    if (TIX + i * block_dim < revbook_nbyte)
      s_revbook[TIX + i * block_dim] = revbook[TIX + i * block_dim];
  }
  __syncthreads();

  auto gid = BIX * BDX + TIX;

  if (gid < pardeg) {
    single_thread_inflate(in + par_entry[gid], out + sublen * gid, par_nbit[gid]);
    __syncthreads();
  }
}

}  // namespace phf

#endif
