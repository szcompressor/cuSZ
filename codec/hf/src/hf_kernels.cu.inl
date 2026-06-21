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

#undef PHF_MODULE_TPL
#undef PHF_MODULE_CLASS

#endif /* HF_KERNEL_INL */
