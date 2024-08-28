#ifndef D3F59CBB_1CC2_441A_8ACD_E598BDD68687
#define D3F59CBB_1CC2_441A_8ACD_E598BDD68687

#include <cstddef>

#include "detail/hfcodec.cuhip.inl"
#include "hfcxx_module.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

#define PHF_MODULE_TPL template <typename E, typename H, bool TIMING>
#define PHF_MODULE_CLASS phf::coarse::kernel_wrapper<E, H, TIMING>

PHF_MODULE_TPL void PHF_MODULE_CLASS::encode_phase1(
    hfcxx_array<E> in, hfcxx_array<H> book, const int numSMs,
    hfcxx_array<H> out, float* time_lossless, void* stream)
{
  auto div = [](auto whole, auto part) -> uint32_t {
    if (whole == 0) throw std::runtime_error("Dividend is zero.");
    if (part == 0) throw std::runtime_error("Divisor is zero.");
    return (whole - 1) / part + 1;
  };

  constexpr auto BLOCK_DIM_ENCODE = 256;
  constexpr auto block_dim = BLOCK_DIM_ENCODE;
  auto grid_dim = div(in.len, block_dim);

  if constexpr (TIMING) {
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING(stream);

    phf::KERNEL_CUHIP_encode_phase1_fill<E, H>                           //
        <<<8 * numSMs, 256, sizeof(H) * book.len, (cudaStream_t)stream>>>  //
        (in.buf, in.len, book.buf, book.len, out.buf);

    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

    float stage_time;
    TIME_ELAPSED_GPUEVENT(&stage_time);
    if (time_lossless) *time_lossless += stage_time;
  }
  else {
    phf::KERNEL_CUHIP_encode_phase1_fill<E, H>                           //
        <<<8 * numSMs, 256, sizeof(H) * book.len, (cudaStream_t)stream>>>  //
        (in.buf, in.len, book.buf, book.len, out.buf);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  }
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::encode_phase1_collect_metadata(
    hfcxx_array<E> in, hfcxx_array<H> book, const int numSMs,
    hfcxx_array<H> out, hfcxx_array<M> par_nbit, hfcxx_array<M> par_ncell,
    hfpar_description hfpar, float* time_lossless, void* stream)
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

  if constexpr (TIMING) {
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING(stream);

    phf::experimental::KERNEL_CUHIP_encode_phase1_fill_collect_metadata<E, H>
        <<<grid_dim, block_dim, sizeof(H) * book.len, (cudaStream_t)stream>>>(
            in.buf, in.len, book.buf, book.len, hfpar.sublen, hfpar.pardeg,
            repeat, out.buf, par_nbit.buf, par_ncell.buf);
    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

    float stage_time;
    TIME_ELAPSED_GPUEVENT(&stage_time);
    if (time_lossless) *time_lossless += stage_time;
  }
  else {
    phf::experimental::KERNEL_CUHIP_encode_phase1_fill_collect_metadata<E, H>
        <<<grid_dim, block_dim, sizeof(H) * book.len, (cudaStream_t)stream>>>(
            in.buf, in.len, book.buf, book.len, hfpar.sublen, hfpar.pardeg,
            repeat, out.buf, par_nbit.buf, par_ncell.buf);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  }
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::encode_phase2(
    hfcxx_array<H> in, hfpar_description hfpar, hfcxx_array<H> deflated,
    hfcxx_array<M> par_nbit, hfcxx_array<M> par_ncell, float* time_lossless,
    void* stream)
{
  auto div = [](auto whole, auto part) -> uint32_t {
    if (whole == 0) throw std::runtime_error("Dividend is zero.");
    if (part == 0) throw std::runtime_error("Divisor is zero.");
    return (whole - 1) / part + 1;
  };

  constexpr auto BLOCK_DIM_DEFLATE = 256;
  auto block_dim = BLOCK_DIM_DEFLATE;
  auto grid_dim = div(hfpar.pardeg, block_dim);

  if constexpr (TIMING) {
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING(stream);

    phf::KERNEL_CUHIP_encode_phase2_deflate<H>  //
        <<<grid_dim, block_dim, 0, (cudaStream_t)stream>>>  //
        (deflated.buf, in.len, par_nbit.buf, par_ncell.buf, hfpar.sublen,
         hfpar.pardeg);

    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

    float stage_time;
    TIME_ELAPSED_GPUEVENT(&stage_time);
    if (time_lossless) *time_lossless += stage_time;
  }
  else {
    phf::KERNEL_CUHIP_encode_phase2_deflate<H>  //
        <<<grid_dim, block_dim, 0, (cudaStream_t)stream>>>  //
        (deflated.buf, in.len, par_nbit.buf, par_ncell.buf, hfpar.sublen,
         hfpar.pardeg);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  }
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::encode_phase3(
    hfcxx_array<M> d_par_nbit, hfcxx_array<M> d_par_ncell,
    hfcxx_array<M> d_par_entry,  //
    hfpar_description hfpar,     //
    hfcxx_array<M> h_par_nbit, hfcxx_array<M> h_par_ncell,
    hfcxx_array<M> h_par_entry,                 //
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
    *outlen_nbit = std::accumulate(
        h_par_nbit.buf, h_par_nbit.buf + hfpar.pardeg, (size_t)0);
  if (outlen_ncell)
    *outlen_ncell = std::accumulate(
        h_par_ncell.buf, h_par_ncell.buf + hfpar.pardeg, (size_t)0);

  CHECK_GPU(cudaMemcpyAsync(
      d_par_entry.buf, h_par_entry.buf, hfpar.pardeg * sizeof(M), cudaMemcpyHostToDevice,
      (cudaStream_t)stream));
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::encode_phase4(
    hfcxx_array<H> buf, hfcxx_array<M> par_entry, hfcxx_array<M> par_ncell,
    hfpar_description hfpar, hfcxx_array<H> bitstream, float* time_lossless,
    void* stream)
{
  if constexpr (TIMING) {
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING(stream);

    phf::KERNEL_CUHIP_encode_phase4_concatenate<H, M>
        <<<hfpar.pardeg, 128, 0, (cudaStream_t)stream>>>  //
        (buf.buf, par_entry.buf, par_ncell.buf, hfpar.sublen, bitstream.buf);

    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

    float stage_time;
    TIME_ELAPSED_GPUEVENT(&stage_time);
    if (time_lossless) *time_lossless += stage_time;
  }
  else {
    phf::KERNEL_CUHIP_encode_phase4_concatenate<H, M>
        <<<hfpar.pardeg, 128, 0, (cudaStream_t)stream>>>  //
        (buf.buf, par_entry.buf, par_ncell.buf, hfpar.sublen, bitstream.buf);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  }
}

PHF_MODULE_TPL void PHF_MODULE_CLASS::phf_coarse_decode(
    hfcxx_array<H> bitstream, hfcxx_array<uint8_t> revbook,
    hfcxx_array<M> par_nbit, hfcxx_array<M> par_entry, hfpar_description hfpar,
    hfcxx_array<E> out, float* time_lossless, void* stream)
{
  auto div = [](auto l, auto subl) { return (l - 1) / subl + 1; };
  auto const block_dim = phf::HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
  auto const grid_dim = div(hfpar.pardeg, block_dim);

  if (TIMING) {
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING(stream);

    phf::KERNEL_CUHIP_decode_kernel<E, H, M>                         //
        <<<grid_dim, block_dim, revbook.len, (cudaStream_t)stream>>>  //
        (bitstream.buf, revbook.buf, par_nbit.buf, par_entry.buf, revbook.len,
         hfpar.sublen, hfpar.pardeg, out.buf);

    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

    TIME_ELAPSED_GPUEVENT(time_lossless);
    DESTROY_GPUEVENT_PAIR;
  }
  else {
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  }
}

// TODO ret type (status) and exe_policy
// duplicate with psz's
PHF_MODULE_TPL void PHF_MODULE_CLASS::phf_scatter_adhoc(
    hfcxx_compact<E> compact, E* out, f4* milliseconds, void* stream)
{
  auto grid_dim = (*(compact.host_num) - 1) / 128 + 1;

  if constexpr (TIMING) {
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING(stream);
    phf::experimental::KERNEL_CUHIP_scatter_adhoc<E, u4>
        <<<grid_dim, 128, 0, (cudaStream_t)stream>>>(
            compact.val, compact.idx, *(compact.host_num), out);
    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

    auto added_time = new float{0};
    TIME_ELAPSED_GPUEVENT(added_time);

    *milliseconds += *added_time;
    DESTROY_GPUEVENT_PAIR;
  }
  else {
    phf::experimental::KERNEL_CUHIP_scatter_adhoc<E, u4>
        <<<grid_dim, 128, 0, (cudaStream_t)stream>>>(
            compact.val, compact.idx, *(compact.host_num), out);
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  }

  // return CUSZ_SUCCESS;
}

#undef PHF_MODULE_TPL
#undef PHF_MODULE_CLASS

#endif /* D3F59CBB_1CC2_441A_8ACD_E598BDD68687 */
