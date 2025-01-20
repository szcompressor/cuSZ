#ifndef D3F59CBB_1CC2_441A_8ACD_E598BDD68687
#define D3F59CBB_1CC2_441A_8ACD_E598BDD68687

#include <cstddef>

#include "detail/hfcodec.cuhip.inl"
#include "hfcxx_module.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

#define PHF_MODULE_TPL template <typename E, typename H>
#define PHF_MODULE_CLASS phf::cuhip::modules<E, H>

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase1(
    phf::array<E> in, phf::array<H> book, const int numSMs, phf::array<H> out, void* stream)
{
  auto div = [](auto whole, auto part) -> uint32_t {
    if (whole == 0) throw std::runtime_error("Dividend is zero.");
    if (part == 0) throw std::runtime_error("Divisor is zero.");
    return (whole - 1) / part + 1;
  };

  constexpr auto BLOCK_DIM_ENCODE = 256;
  constexpr auto block_dim = BLOCK_DIM_ENCODE;
  auto grid_dim = div(in.len, block_dim);

  phf::KERNEL_CUHIP_encode_phase1_fill<E, H>                             //
      <<<8 * numSMs, 256, sizeof(H) * book.len, (cudaStream_t)stream>>>  //
      (in.buf, in.len, book.buf, book.len, out.buf);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase1_collect_metadata(
    phf::array<E> in, phf::array<H> book, const int numSMs, phf::array<H> out,
    phf::array<M> par_nbit, phf::array<M> par_ncell, phf::par_config hfpar, void* stream)
{
  auto div = [](auto whole, auto part) -> uint32_t {
    if (whole == 0) throw std::runtime_error("Dividend is zero.");
    if (part == 0) throw std::runtime_error("Divisor is zero.");
    return (whole - 1) / part + 1;
  };

  constexpr auto BLOCK_DIM_ENCODE = 256;
  constexpr auto block_dim = BLOCK_DIM_ENCODE;
  auto repeat = 10;
  auto grid_dim = div(in.len, repeat * hfpar.sublen);

  phf::experimental::KERNEL_CUHIP_encode_phase1_fill_collect_metadata<E, H>
      <<<grid_dim, block_dim, sizeof(H) * book.len, (cudaStream_t)stream>>>(
          in.buf, in.len, book.buf, book.len, hfpar.sublen, hfpar.pardeg, repeat, out.buf,
          par_nbit.buf, par_ncell.buf);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase2(
    phf::array<H> in, phf::par_config hfpar, phf::array<H> deflated, phf::array<M> par_nbit,
    phf::array<M> par_ncell, void* stream)
{
  auto div = [](auto whole, auto part) -> uint32_t {
    if (whole == 0) throw std::runtime_error("Dividend is zero.");
    if (part == 0) throw std::runtime_error("Divisor is zero.");
    return (whole - 1) / part + 1;
  };

  constexpr auto BLOCK_DIM_DEFLATE = 256;
  auto block_dim = BLOCK_DIM_DEFLATE;
  auto grid_dim = div(hfpar.pardeg, block_dim);

  phf::KERNEL_CUHIP_encode_phase2_deflate<H>              //
      <<<grid_dim, block_dim, 0, (cudaStream_t)stream>>>  //
      (deflated.buf, in.len, par_nbit.buf, par_ncell.buf, hfpar.sublen, hfpar.pardeg);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase3_sync(
    phf::array<M> d_par_nbit, phf::array<M> d_par_ncell,
    phf::array<M> d_par_entry,  //
    phf::par_config hfpar,      //
    phf::array<M> h_par_nbit, phf::array<M> h_par_ncell,
    phf::array<M> h_par_entry,                  //
    size_t* outlen_nbit, size_t* outlen_ncell,  //
    float* time_cpu_time, void* stream)
{
  CHECK_GPU(cudaMemcpyAsync(
      h_par_nbit.buf, d_par_nbit.buf, hfpar.pardeg * sizeof(M), cudaMemcpyDeviceToHost,
      (cudaStream_t)stream));
  CHECK_GPU(cudaMemcpyAsync(
      h_par_ncell.buf, d_par_ncell.buf, hfpar.pardeg * sizeof(M), cudaMemcpyDeviceToHost,
      (cudaStream_t)stream));
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  memcpy(h_par_entry.buf + 1, h_par_ncell.buf, (hfpar.pardeg - 1) * sizeof(M));
  for (auto i = 1; i < hfpar.pardeg; i++)
    h_par_entry.buf[i] += h_par_entry.buf[i - 1];  // inclusive scan
  if (outlen_nbit)
    *outlen_nbit = std::accumulate(h_par_nbit.buf, h_par_nbit.buf + hfpar.pardeg, (size_t)0);
  if (outlen_ncell)
    *outlen_ncell = std::accumulate(h_par_ncell.buf, h_par_ncell.buf + hfpar.pardeg, (size_t)0);

  CHECK_GPU(cudaMemcpyAsync(
      d_par_entry.buf, h_par_entry.buf, hfpar.pardeg * sizeof(M), cudaMemcpyHostToDevice,
      (cudaStream_t)stream));
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_encode_phase4(
    phf::array<H> buf, phf::array<M> par_entry, phf::array<M> par_ncell, phf::par_config hfpar,
    phf::array<H> bitstream, void* stream)
{
  phf::KERNEL_CUHIP_encode_phase4_concatenate<H, M>
      <<<hfpar.pardeg, 128, 0, (cudaStream_t)stream>>>  //
      (buf.buf, par_entry.buf, par_ncell.buf, hfpar.sublen, bitstream.buf);
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_coarse_decode(
    H* in_bitstream, uint8_t* in_revbook, size_t const revbook_len, M* in_par_nbit,
    M* in_par_entry, size_t const sublen, size_t const pardeg, E* out_decoded, void* stream)
{
  auto div = [](auto l, auto subl) { return (l - 1) / subl + 1; };
  auto const block_dim = phf::HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
  auto const grid_dim = div(pardeg, block_dim);

  phf::KERNEL_CUHIP_HF_decode<E, H, M>                              //
      <<<grid_dim, block_dim, revbook_len, (cudaStream_t)stream>>>  //
      (in_bitstream, in_revbook, in_par_nbit, in_par_entry, revbook_len, sublen, pardeg,
       out_decoded);
}

// TODO ret type (status) and exe_policy
// duplicate with psz's
PHF_MODULE_TPL void PHF_MODULE_CLASS::GPU_experimental_scatter(
    phf::sparse<E> compact, E* out, void* stream)
{
  auto grid_dim = (*(compact.host_num) - 1) / 128 + 1;

  phf::experimental::KERNEL_CUHIP_scatter<E, u4><<<grid_dim, 128, 0, (cudaStream_t)stream>>>(
      compact.val, compact.idx, *(compact.host_num), out);

  // return CUSZ_SUCCESS;
}

#undef PHF_MODULE_TPL
#undef PHF_MODULE_CLASS

#endif /* D3F59CBB_1CC2_441A_8ACD_E598BDD68687 */
