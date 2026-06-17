// Author: Jiannan Tian
// Huffman kernel definitions

#ifndef HF_KERNEL_INL
#define HF_KERNEL_INL

#include <cstddef>
#include <cstdio>
#include <numeric>
#include <stdexcept>

#include "hf_impl.hh"
#include "single_inflate.inl"

constexpr int PHF_BLOCK_DIM_ENCODE = 256;
constexpr int PHF_BLOCK_DIM_DEFLATE = 256;

using BYTE = uint8_t;

extern __shared__ char __codec_raw[];

namespace phf::experimental {
// a duplicate from psz
template <typename T, typename M = u4>
__global__ void KCU_scatter(T* val, M* idx, int const n, T* out)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    int dst_idx = idx[tid];
    out[dst_idx] = val[tid];
  }
}

template <typename T, typename BreakCell, typename M = u4>
__global__ void KCU_scatter_breaks(
    BreakCell* sp_breaks, M* par_brnum, M* par_broffset, int const sublen, T* out)
{
  auto blk = blockIdx.x;
  auto count = par_brnum[blk];
  auto base = par_broffset[blk];

  for (auto i = threadIdx.x; i < count; i += blockDim.x) {
    auto cell = sp_breaks[base + i];
    out[(size_t)blk * sublen + cell.idx] = (T)cell.val;
  }
}

}  // namespace phf::experimental

namespace phf {

template <typename E, typename H>
__global__ void KCU_enc_ph1_fill(
    E* in, size_t const in_len, H* in_bk, int const in_bklen, H* out_encoded)
{
  auto s_bk = reinterpret_cast<H*>(__codec_raw);

  // load from global memory
  for (auto idx = threadIdx.x;  //
       idx < in_bklen;          //
       idx += blockDim.x)
    s_bk[idx] = in_bk[idx];

  __syncthreads();

  for (auto idx = blockIdx.x * blockDim.x + threadIdx.x;  //
       idx < in_len;                                      //
       idx += blockDim.x * gridDim.x                      //
  )
    out_encoded[idx] = s_bk[(int)in[idx]];
}

template <typename H, typename M>
__global__ void KCU_enc_ph2_deflate(
    H* inout_inplace, size_t const len, M* par_nbit, M* par_ncell, int const sublen,
    int const pardeg)
{
  constexpr int CELL_BITWIDTH = sizeof(H) * 8;

  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

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
__global__ void KCU_enc_ph4_concat(
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
__global__ void KCU_Huffman_ReVISIT_lite(
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

    if (p_bits > BITWIDTH) {
      p_bits = 0u;
      p_reduced = 0x0;
      auto p_val_ref = s_book[runtime_bklen / 2];
      auto const sym_bits_ref = bitcount_of(&p_val_ref);

#pragma unroll
      for (auto ix = 0u, br_lidx = (threadIdx.x * ShardSize); ix < ShardSize; ix++, br_lidx++) {
        auto p_key = s_to_encode[br_lidx];
        auto p_val = s_book[p_key];
        auto sym_bits = bitcount_of(&p_val);

        if (sym_bits > (BITWIDTH / ShardSize)) {
          auto br_gidx = atomicAdd(hf_brnum, 1u);
          hf_bridx[br_gidx] = id_base + br_lidx;
          hf_brval[br_gidx] = p_key;
          p_val = p_val_ref;
          sym_bits = sym_bits_ref;
        }

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

template <typename Ein, typename H, typename M, typename Eout = Ein>
__global__ void KCU_HF_decode(
    H* in, uint8_t* revbook, M* par_nbit, M* par_entry, int const revbook_nbyte, int const sublen,
    int const pardeg, Eout* out, uint8_t* par_encid /* nullable: HF coarse path passes nullptr */)
{
  extern __shared__ uint8_t s_revbook[];

  constexpr auto block_dim = PHF_BLOCK_DIM_DEFLATE;
  auto R = (revbook_nbyte - 1 + block_dim) / block_dim;

  for (auto i = 0; i < R; i++) {
    if (threadIdx.x + i * block_dim < revbook_nbyte)
      s_revbook[threadIdx.x + i * block_dim] = revbook[threadIdx.x + i * block_dim];
  }
  __syncthreads();

  auto gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < pardeg) {
    if (par_encid != nullptr and par_encid[gid] == 1) {
      // Incomp block: bitstream slot at par_entry[gid] holds raw Ein values.
      auto raw = (Ein*)(in + par_entry[gid]);
      auto dst = out + (size_t)sublen * gid;
      for (int i = 0; i < sublen; i++) dst[i] = (Eout)raw[i];
    }
    else {
      phf::single_thread_inflate<Ein, H, Ein, Eout>(
          in + par_entry[gid], out + (size_t)sublen * gid, s_revbook, par_nbit[gid]);
    }
    __syncthreads();
  }
}

}  // namespace phf

#define PHF_MODULE_TPL template <typename E, typename H>
#define PHF_MODULE_CLASS phf::cuhip::modules<E, H>
#define SETUP_DIV                                                  \
  auto div = [](auto whole, auto part) -> uint32_t {               \
    if (whole == 0) throw std::runtime_error("Dividend is zero."); \
    if (part == 0) throw std::runtime_error("Divisor is zero.");   \
    return (whole - 1) / part + 1;                                 \
  };

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_enc_ph1(
    E* in_data, const size_t data_len, H* in_book, const u4 book_len, const int num_SMs,
    H* out_bitstream, void* stream)
{
  SETUP_DIV;

  constexpr auto block_dim = PHF_BLOCK_DIM_ENCODE;
  auto grid_dim = div(data_len, block_dim);
  phf::KCU_enc_ph1_fill<E, H>                                             //
      <<<8 * num_SMs, 256, sizeof(H) * book_len, (cudaStream_t)stream>>>  //
      (in_data, data_len, in_book, book_len, out_bitstream);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_enc_ph2(
    H* in_data, const size_t data_len, phf::par_config hfpar, H* deflated, M* par_nbit,
    M* par_ncell, void* stream)
{
  SETUP_DIV;

  auto block_dim = PHF_BLOCK_DIM_DEFLATE;
  auto grid_dim = div(hfpar.pardeg, block_dim);
  phf::KCU_enc_ph2_deflate<H>                             //
      <<<grid_dim, block_dim, 0, (cudaStream_t)stream>>>  //
      (deflated, data_len, par_nbit, par_ncell, hfpar.sublen, hfpar.pardeg);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_fine_enc_ph1_2(
    E* in, const size_t len, H* book, const u4 bklen, H* bitstream, M* par_nbit, M* par_ncell,
    const u4 nblock, E* brval, u4* bridx, u4* brnum, void* stream)
{
  SETUP_DIV;
  constexpr int ChunkSize = 1024;
  constexpr int BlockDim = 256;
  auto grid_dim = div(len, ChunkSize);

  phf::KCU_Huffman_ReVISIT_lite<E>                       //
      <<<grid_dim, BlockDim, 0, (cudaStream_t)stream>>>  //
      (in, len, book, bklen, bitstream, par_nbit, par_ncell, nblock, brval, bridx, brnum);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_enc_ph3_sync(
    phf::par_config hfpar, M* d_par_nbit, M* h_par_nbit, M* d_par_ncell, M* h_par_ncell,
    M* d_par_entry, M* h_par_entry, size_t* outlen_nbit, size_t* outlen_ncell,
    float* time_cpu_time, void* stream)
{
  cudaMemcpyAsync(
      h_par_nbit, d_par_nbit, hfpar.pardeg * sizeof(M), cudaMemcpyDeviceToHost,
      (cudaStream_t)stream);
  cudaMemcpyAsync(
      h_par_ncell, d_par_ncell, hfpar.pardeg * sizeof(M), cudaMemcpyDeviceToHost,
      (cudaStream_t)stream);
  cudaStreamSynchronize((cudaStream_t)stream);

  memcpy(h_par_entry + 1, h_par_ncell, (hfpar.pardeg - 1) * sizeof(M));
  for (auto i = 1; i < hfpar.pardeg; i++) h_par_entry[i] += h_par_entry[i - 1];  // inclusive scan
  if (outlen_nbit)
    *outlen_nbit = std::accumulate(h_par_nbit, h_par_nbit + hfpar.pardeg, (size_t)0);
  if (outlen_ncell)
    *outlen_ncell = std::accumulate(h_par_ncell, h_par_ncell + hfpar.pardeg, (size_t)0);

  cudaMemcpyAsync(
      d_par_entry, h_par_entry, hfpar.pardeg * sizeof(M), cudaMemcpyHostToDevice,
      (cudaStream_t)stream);
  cudaStreamSynchronize((cudaStream_t)stream);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_enc_ph4(
    H* in_buf, const size_t len, M* par_entry, M* par_ncell, phf::par_config hfpar, H* bitstream,
    const size_t max_bitstream_len, void* stream)
{
  phf::KCU_enc_ph4_concat<H, M><<<hfpar.pardeg, 128, 0, (cudaStream_t)stream>>>  //
      (in_buf, par_entry, par_ncell, hfpar.sublen, bitstream);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode(
    E* in_data, size_t data_len, H* in_book, u4 book_len, int num_SMs, phf::par_config hfpar,
    // internal buffers
    H* d_scratch4, M* d_par_nbit, M* h_par_nbit, M* d_par_ncell, M* h_par_ncell, M* d_par_entry,
    M* h_par_entry, H* d_bitstream4, size_t bitstream_max_len,
    // output
    size_t* out_total_nbit, size_t* out_total_ncell, void* stream)
{
  GPU_coarse_enc_ph1(in_data, data_len, in_book, book_len, num_SMs, d_scratch4, stream);
  GPU_coarse_enc_ph2(d_scratch4, data_len, hfpar, d_scratch4, d_par_nbit, d_par_ncell, stream);
  GPU_coarse_enc_ph3_sync(
      hfpar, d_par_nbit, h_par_nbit, d_par_ncell, h_par_ncell, d_par_entry, h_par_entry,
      out_total_nbit, out_total_ncell, nullptr, stream);
  GPU_coarse_enc_ph4(
      d_scratch4, data_len, d_par_entry, d_par_ncell, hfpar, d_bitstream4, bitstream_max_len,
      stream);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_fine_encode(
    E* in_data, size_t data_len, H* in_book, u4 book_len, phf::par_config hfpar,
    // internal buffers
    H* d_scratch4, M* d_par_nbit, M* h_par_nbit, M* d_par_ncell, M* h_par_ncell, M* d_par_entry,
    M* h_par_entry, H* d_bitstream4, size_t bitstream_max_len, E* d_brval, u4* d_bridx,
    u4* d_brnum,
    // output
    size_t* out_total_nbit, size_t* out_total_ncell, void* stream)
{
  GPU_fine_enc_ph1_2(
      in_data, data_len, in_book, book_len, d_scratch4, d_par_nbit, d_par_ncell, hfpar.pardeg,
      d_brval, d_bridx, d_brnum, stream);
  GPU_coarse_enc_ph3_sync(
      hfpar, d_par_nbit, h_par_nbit, d_par_ncell, h_par_ncell, d_par_entry, h_par_entry,
      out_total_nbit, out_total_ncell, nullptr, stream);
  GPU_coarse_enc_ph4(
      d_scratch4, data_len, d_par_entry, d_par_ncell, hfpar, d_bitstream4, bitstream_max_len,
      stream);
}

PHF_MODULE_TPL template <typename Eout>
void PHF_MODULE_CLASS::GPU_coarse_decode(
    H* in_bitstream, uint8_t* in_revbook, size_t const revbook_len, M* in_par_nbit,
    M* in_par_entry, size_t const sublen, size_t const pardeg, Eout* out_decoded,
    uint8_t* in_par_encid /* nullptr for plain-HF coarse path */, void* stream)
{
  SETUP_DIV;
  auto const block_dim = PHF_BLOCK_DIM_DEFLATE;  // = deflating
  auto const grid_dim = div(pardeg, block_dim);

  phf::KCU_HF_decode<E, H, M, Eout>                                 //
      <<<grid_dim, block_dim, revbook_len, (cudaStream_t)stream>>>  //
      (in_bitstream, in_revbook, in_par_nbit, in_par_entry, revbook_len, sublen, pardeg,
       out_decoded, in_par_encid);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_scatter_breaks(
    psz::HFR_PBK_Breaks<128>* sp_breaks, u4* par_brnum, u4* par_broffset, int const sublen,
    int const pardeg, E* out, void* stream)
{
  constexpr int block_dim = 128;
  phf::experimental::KCU_scatter_breaks<E, psz::HFR_PBK_Breaks<128>, u4>
      <<<pardeg, block_dim, 0, (cudaStream_t)stream>>>(
          sp_breaks, par_brnum, par_broffset, sublen, out);
}

#undef PHF_MODULE_TPL
#undef PHF_MODULE_CLASS

#endif /* HF_KERNEL_INL */
