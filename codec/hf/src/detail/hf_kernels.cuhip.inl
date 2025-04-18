/**
 * @file hf_kernels.cuhip.inl
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

#include "hf_ood.hh"  // contains HuffmanHelper; TODO put in another file
#include "hf_word.hh"

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

using CompactIdx = uint32_t;
using CompactNum = uint32_t;
#define CompactVal T
#define CV CompactVal
#define CI CompactIdx
#define CN CompactNum

using Hf = uint32_t;

template <typename E, int ChunkSize = 1024, int ShardSize = 4, int MaxBkLen = 1024>
__global__ void KERNEL_CUHIP_Huffman_ReVISIT_lite(
    E* in_data, size_t const len, Hf* hf_book, const u4 runtime_bklen, u4* hf_bitstream,
    u4* hf_bits, u4* hf_cells, const u4 nblock, /* breaking handling */
    E* hf_brval, CI* hf_bridx, CN* hf_brnum)
{
  constexpr auto NumThreads = ChunkSize / ShardSize;
  // constexpr auto NumWarps = NumThreads / 32;

  __shared__ E s_to_encode[ChunkSize];
  auto const id_base = blockIdx.x * ChunkSize;

// dram.in_data to shmem.in_data
#pragma unroll
  for (auto ix = 0; ix < ShardSize; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < len) s_to_encode[threadIdx.x + ix * NumThreads] = in_data[id];
  }
  __syncthreads();

  // lite: hardcoded parameters
  constexpr auto ReduceTimes = 2u;
  constexpr auto ShuffleTimes = 8u;
  constexpr auto BITWIDTH = 32;

  static_assert(ShardSize == 1 << ReduceTimes, "Wrong reduce times.");
  static_assert(ChunkSize == 1 << (ReduceTimes + ShuffleTimes), "Wrong shuffle times.");

  __shared__ Hf s_book[MaxBkLen];
  __shared__ Hf s_reduced[NumThreads * 2];   // !!!! check types E and Hf
  __shared__ u4 s_bitcount[NumThreads * 2];  // !!!! check types E and Hf

  auto bitcount_of = [](Hf* _w) { return reinterpret_cast<HuffmanWord<4>*>(_w)->bitcount; };
  auto entry = [&]() -> size_t { return ChunkSize * blockIdx.x; };
  auto allowed_len = [&]() { return min((size_t)ChunkSize, len - entry()); };

  ////////// load codebook
  for (auto i = threadIdx.x; i < runtime_bklen; i += NumThreads) { s_book[i] = hf_book[i]; }
  __syncthreads();

  ////////// start of reduce-merge
  {
    auto p_bits{0u};
    Hf p_reduced{0x0};

    // per-thread loop, merge
    for (auto i = 0; i < ShardSize; i++) {
      auto idx = (threadIdx.x * ShardSize) + i;
      auto p_key = s_to_encode[idx];
      auto p_val = s_book[p_key];
      auto sym_bits = bitcount_of(&p_val);

      p_val <<= (BITWIDTH - sym_bits);
      p_reduced |= (p_val >> p_bits);
      p_bits += sym_bits * (idx < allowed_len());
    }

    // breaking handling
    if (p_bits > BITWIDTH) {
      p_bits = 0u;      // reset on breaking
      p_reduced = 0x0;  // reset on breaking, too
      auto p_val_ref = s_book[MaxBkLen];
      auto const sym_bits = bitcount_of(&p_val_ref);
      auto br_gidx_start = atomicAdd(hf_brnum, ShardSize);
#pragma unroll
      for (auto ix = 0u, br_lidx = (threadIdx.x * ShardSize); ix < ShardSize; ix++, br_lidx++) {
        hf_bridx[br_gidx_start + ix] = id_base + br_lidx;
        hf_brval[br_gidx_start + ix] = s_to_encode[br_lidx];

        auto p_val = p_val_ref;
        p_val <<= (BITWIDTH - sym_bits);
        p_reduced |= (p_val >> p_bits);
        p_bits += sym_bits * (br_lidx < allowed_len());
      }
    }

    // still for this thread only
    s_reduced[threadIdx.x] = p_reduced;
    s_bitcount[threadIdx.x] = p_bits;
  }
  __syncthreads();

  ////////// end of reduce-merge; start of shuffle-merge

  for (auto sf = ShuffleTimes, stride = 1u; sf > 0; sf--, stride *= 2) {
    auto l = threadIdx.x / (stride * 2) * (stride * 2);
    auto r = l + stride;

    auto lbc = s_bitcount[l];
    u4 used__units = lbc / BITWIDTH;
    u4 used___bits = lbc % BITWIDTH;
    u4 unused_bits = BITWIDTH - used___bits;

    auto lend = (Hf*)(s_reduced + l + used__units);
    auto this_point = s_reduced[threadIdx.x];
    auto lsym = this_point >> used___bits;
    auto rsym = this_point << unused_bits;

    if (threadIdx.x >= r and threadIdx.x < r + stride)
      atomicAnd((Hf*)(s_reduced + threadIdx.x), 0x0);
    __syncthreads();

    if (threadIdx.x >= r and threadIdx.x < r + stride) {
      atomicOr(lend + (threadIdx.x - r) + 0, lsym);
      atomicOr(lend + (threadIdx.x - r) + 1, rsym);
    }

    if (threadIdx.x == l) s_bitcount[l] += s_bitcount[r];
    __syncthreads();
  }
  ////////// end of shuffle-merge, start of outputting

  __shared__ u4 s_wunits;
  ull p_wunits;

  static_assert(BITWIDTH == 32, "Wrong bitwidth (!=32).");
  if (threadIdx.x == 0) {
    u4 p_bc = s_bitcount[0];
    p_wunits = (p_bc + 31) / 32;

    hf_bits[blockIdx.x] = p_bc;
    hf_cells[blockIdx.x] = p_wunits;

    s_wunits = p_wunits;
  }
  __syncthreads();

  if (threadIdx.x % 32 == 0 and threadIdx.x / 32 > 0) { p_wunits = s_wunits; }
  __syncthreads();

  p_wunits = __shfl_sync(0xffffffff, p_wunits, 0);

  for (auto i = threadIdx.x; i < p_wunits; i += blockDim.x) {
    Hf w = s_reduced[i];
    hf_bitstream[id_base + i] = w;
  }

  ////////// end of outputting the encoded

  // end of kernel
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
