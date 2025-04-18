#ifndef D3F59CBB_1CC2_441A_8ACD_E598BDD68687
#define D3F59CBB_1CC2_441A_8ACD_E598BDD68687

#include <cstddef>

#include "detail/hf_kernels.cuhip.inl"
#include "hf_kernels.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

#define PHF_MODULE_TPL template <typename E, typename H>
#define PHF_MODULE_CLASS phf::cuhip::modules<E, H>
#define SETUP_DIV                                                  \
  auto div = [](auto whole, auto part) -> uint32_t {               \
    if (whole == 0) throw std::runtime_error("Dividend is zero."); \
    if (part == 0) throw std::runtime_error("Divisor is zero.");   \
    return (whole - 1) / part + 1;                                 \
  };

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase1(
    E* in_data, const size_t data_len, H* in_book, const u4 book_len, const int numSMs,
    H* out_bitstream, void* stream)
{
  SETUP_DIV;
  constexpr auto BLOCK_DIM_ENCODE = 256;
  constexpr auto block_dim = BLOCK_DIM_ENCODE;
  auto grid_dim = div(data_len, block_dim);
  phf::KERNEL_CUHIP_encode_phase1_fill<E, H>                             //
      <<<8 * numSMs, 256, sizeof(H) * book_len, (cudaStream_t)stream>>>  //
      (in_data, data_len, in_book, book_len, out_bitstream);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase2(
    H* in_data, const size_t data_len, phf::par_config hfpar, H* deflated, M* par_nbit,
    M* par_ncell, void* stream)
{
  SETUP_DIV;
  constexpr auto BLOCK_DIM_DEFLATE = 256;
  auto block_dim = BLOCK_DIM_DEFLATE;
  auto grid_dim = div(hfpar.pardeg, block_dim);
  phf::KERNEL_CUHIP_encode_phase2_deflate<H>              //
      <<<grid_dim, block_dim, 0, (cudaStream_t)stream>>>  //
      (deflated, data_len, par_nbit, par_ncell, hfpar.sublen, hfpar.pardeg);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_fine_encode_phase1_2(
    E* in, const size_t len, H* book, const u4 bklen, H* bitstream, M* par_nbit, M* par_ncell,
    const u4 nblock, E* brval, u4* bridx, u4* brnum, void* stream)
{
  SETUP_DIV;
  constexpr int ChunkSize = 1024;
  constexpr int BlockDim = 256;
  auto grid_dim = div(len, ChunkSize);

  phf::KERNEL_CUHIP_Huffman_ReVISIT_lite<E>              //
      <<<grid_dim, BlockDim, 0, (cudaStream_t)stream>>>  //
      (in, len, book, bklen, bitstream, par_nbit, par_ncell, nblock, brval, bridx, brnum);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase3_sync(
    phf::par_config hfpar, M* d_par_nbit, M* h_par_nbit, M* d_par_ncell, M* h_par_ncell,
    M* d_par_entry, M* h_par_entry, size_t* outlen_nbit, size_t* outlen_ncell,
    float* time_cpu_time, void* stream)
{
  CHECK_GPU(cudaMemcpyAsync(
      h_par_nbit, d_par_nbit, hfpar.pardeg * sizeof(M), cudaMemcpyDeviceToHost,
      (cudaStream_t)stream));
  CHECK_GPU(cudaMemcpyAsync(
      h_par_ncell, d_par_ncell, hfpar.pardeg * sizeof(M), cudaMemcpyDeviceToHost,
      (cudaStream_t)stream));
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  memcpy(h_par_entry + 1, h_par_ncell, (hfpar.pardeg - 1) * sizeof(M));
  for (auto i = 1; i < hfpar.pardeg; i++) h_par_entry[i] += h_par_entry[i - 1];  // inclusive scan
  if (outlen_nbit)
    *outlen_nbit = std::accumulate(h_par_nbit, h_par_nbit + hfpar.pardeg, (size_t)0);
  if (outlen_ncell)
    *outlen_ncell = std::accumulate(h_par_ncell, h_par_ncell + hfpar.pardeg, (size_t)0);

  CHECK_GPU(cudaMemcpyAsync(
      d_par_entry, h_par_entry, hfpar.pardeg * sizeof(M), cudaMemcpyHostToDevice,
      (cudaStream_t)stream));
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase4(
    H* in_buf, const size_t len, M* par_entry, M* par_ncell, phf::par_config hfpar, H* bitstream,
    const size_t max_bitstream_len, void* stream)
{
  phf::KERNEL_CUHIP_encode_phase4_concatenate<H, M>
      <<<hfpar.pardeg, 128, 0, (cudaStream_t)stream>>>  //
      (in_buf, par_entry, par_ncell, hfpar.sublen, bitstream);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode(
    E* in_data, size_t data_len, H* in_book, u4 book_len, int numSMs, phf::par_config hfpar,
    // internal buffers
    H* d_scratch4, M* d_par_nbit, M* h_par_nbit, M* d_par_ncell, M* h_par_ncell, M* d_par_entry,
    M* h_par_entry, H* d_bitstream4, size_t bitstream_max_len,
    // output
    size_t* out_total_nbit, size_t* out_total_ncell, void* stream)
{
  GPU_coarse_encode_phase1(in_data, data_len, in_book, book_len, numSMs, d_scratch4, stream);
  GPU_coarse_encode_phase2(
      d_scratch4, data_len, hfpar, d_scratch4, d_par_nbit, d_par_ncell, stream);
  GPU_coarse_encode_phase3_sync(
      hfpar, d_par_nbit, h_par_nbit, d_par_ncell, h_par_ncell, d_par_entry, h_par_entry,
      out_total_nbit, out_total_ncell, nullptr, stream);
  GPU_coarse_encode_phase4(
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
  GPU_fine_encode_phase1_2(
      in_data, data_len, in_book, book_len, d_scratch4, d_par_nbit, d_par_ncell, hfpar.pardeg,
      d_brval, d_bridx, d_brnum, stream);
  GPU_coarse_encode_phase3_sync(
      hfpar, d_par_nbit, h_par_nbit, d_par_ncell, h_par_ncell, d_par_entry, h_par_entry,
      out_total_nbit, out_total_ncell, nullptr, stream);
  GPU_coarse_encode_phase4(
      d_scratch4, data_len, d_par_entry, d_par_ncell, hfpar, d_bitstream4, bitstream_max_len,
      stream);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_decode(
    H* in_bitstream, uint8_t* in_revbook, size_t const revbook_len, M* in_par_nbit,
    M* in_par_entry, size_t const sublen, size_t const pardeg, E* out_decoded, void* stream)
{
  SETUP_DIV;
  auto const block_dim = phf::HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
  auto const grid_dim = div(pardeg, block_dim);

  phf::KERNEL_CUHIP_HF_decode<E, H, M>                              //
      <<<grid_dim, block_dim, revbook_len, (cudaStream_t)stream>>>  //
      (in_bitstream, in_revbook, in_par_nbit, in_par_entry, revbook_len, sublen, pardeg,
       out_decoded);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_scatter(
    E* val, u4* idx, const u4 h_num, E* out, void* stream)
{
  SETUP_DIV;
  auto grid_dim = div(h_num, 128);
  phf::experimental::KERNEL_CUHIP_scatter<E, u4>
      <<<grid_dim, 128, 0, (cudaStream_t)stream>>>(val, idx, h_num, out);
}

#undef PHF_MODULE_TPL
#undef PHF_MODULE_CLASS

#endif /* D3F59CBB_1CC2_441A_8ACD_E598BDD68687 */
