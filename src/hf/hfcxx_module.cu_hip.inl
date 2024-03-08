#ifndef D3F59CBB_1CC2_441A_8ACD_E598BDD68687
#define D3F59CBB_1CC2_441A_8ACD_E598BDD68687

#include <cstddef>

#include "hf/detail/hfcodec.cu_hip.inl"
#include "hf/hfcodec.hh"
#include "hf/hfcxx_module.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

/**
 * @tparam E
 * @tparam H
 * @tparam M
 * @tparam TIMING default true to replicate the original
 */
template <typename E, typename H, typename M, bool TIMING>
void _2403::hf_encode_coarse_phase1(
    hfarray_cxx<E> in, hfarray_cxx<H> book, const int numSMs,
    hfarray_cxx<H> out, float* time_lossless, void* stream)
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

    psz::detail::hf_encode_phase1_fill<E, H>                             //
        <<<8 * numSMs, 256, sizeof(H) * book.len, (GpuStreamT)stream>>>  //
        (in.buf, in.len, book.buf, book.len, out.buf);

    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(GpuStreamSync(stream));

    float stage_time;
    TIME_ELAPSED_GPUEVENT(&stage_time);
    if (time_lossless) *time_lossless += stage_time;
  }
  else {
    psz::detail::hf_encode_phase1_fill<E, H>                             //
        <<<8 * numSMs, 256, sizeof(H) * book.len, (GpuStreamT)stream>>>  //
        (in.buf, in.len, book.buf, book.len, out.buf);
    CHECK_GPU(GpuStreamSync(stream));
  }
}

template <typename H, typename M, bool TIMING>
void _2403::hf_encode_coarse_phase2(
    hfarray_cxx<H> in, hfpar_description hfpar, hfarray_cxx<H> deflated,
    hfarray_cxx<M> par_nbit, hfarray_cxx<M> par_ncell, float* time_lossless,
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

    psz::detail::hf_encode_phase2_deflate<H>              //
        <<<grid_dim, block_dim, 0, (GpuStreamT)stream>>>  //
        (deflated.buf, in.len, par_nbit.buf, par_ncell.buf, hfpar.sublen,
         hfpar.pardeg);

    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(GpuStreamSync(stream));

    float stage_time;
    TIME_ELAPSED_GPUEVENT(&stage_time);
    if (time_lossless) *time_lossless += stage_time;
  }
  else {
    psz::detail::hf_encode_phase2_deflate<H>              //
        <<<grid_dim, block_dim, 0, (GpuStreamT)stream>>>  //
        (deflated.buf, in.len, par_nbit.buf, par_ncell.buf, hfpar.sublen,
         hfpar.pardeg);
    CHECK_GPU(GpuStreamSync(stream));
  }
}

template <typename M, bool TIMING>
void _2403::hf_encode_coarse_phase3(
    hfarray_cxx<M> d_par_nbit, hfarray_cxx<M> d_par_ncell,
    hfarray_cxx<M> d_par_entry,  //
    hfpar_description hfpar,     //
    hfarray_cxx<M> h_par_nbit, hfarray_cxx<M> h_par_ncell,
    hfarray_cxx<M> h_par_entry,                 //
    size_t* outlen_nbit, size_t* outlen_ncell,  //
    float* time_cpu_time, void* stream)
{
  CHECK_GPU(GpuMemcpyAsync(
      h_par_nbit.buf, d_par_nbit.buf, hfpar.pardeg * sizeof(M), GpuMemcpyD2H,
      (GpuStreamT)stream));
  CHECK_GPU(GpuMemcpyAsync(
      h_par_ncell.buf, d_par_ncell.buf, hfpar.pardeg * sizeof(M), GpuMemcpyD2H,
      (GpuStreamT)stream));
  CHECK_GPU(GpuStreamSync(stream));

  memcpy(h_par_entry.buf + 1, h_par_ncell.buf, (hfpar.pardeg - 1) * sizeof(M));
  for (auto i = 1; i < hfpar.pardeg; i++)
    h_par_entry.buf[i] += h_par_entry.buf[i - 1];  // inclusive scan
  if (outlen_nbit)
    *outlen_nbit = std::accumulate(
        h_par_nbit.buf, h_par_nbit.buf + hfpar.pardeg, (size_t)0);
  if (outlen_ncell)
    *outlen_ncell = std::accumulate(
        h_par_ncell.buf, h_par_ncell.buf + hfpar.pardeg, (size_t)0);

  CHECK_GPU(GpuMemcpyAsync(
      d_par_entry.buf, h_par_entry.buf, hfpar.pardeg * sizeof(M), GpuMemcpyH2D,
      (GpuStreamT)stream));
  CHECK_GPU(GpuStreamSync(stream));
}

template <typename H, typename M, bool TIMING>
void _2403::hf_encode_coarse_phase4(
    hfarray_cxx<H> buf, hfarray_cxx<M> par_entry, hfarray_cxx<M> par_ncell,
    hfpar_description hfpar, hfarray_cxx<H> bitstream, float* time_lossless,
    void* stream)
{
  if constexpr (TIMING) {
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING(stream);

    psz::detail::hf_encode_phase4_concatenate<H, M>
        <<<hfpar.pardeg, 128, 0, (GpuStreamT)stream>>>  //
        (buf.buf, par_entry.buf, par_ncell.buf, hfpar.sublen, bitstream.buf);

    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(GpuStreamSync(stream));

    float stage_time;
    TIME_ELAPSED_GPUEVENT(&stage_time);
    if (time_lossless) *time_lossless += stage_time;
  }
  else {
    psz::detail::hf_encode_phase4_concatenate<H, M>
        <<<hfpar.pardeg, 128, 0, (GpuStreamT)stream>>>  //
        (buf.buf, par_entry.buf, par_ncell.buf, hfpar.sublen, bitstream.buf);
    CHECK_GPU(GpuStreamSync(stream));
  }
}

template <typename E, typename H, typename M, bool TIMING>
void _2403::hf_decode_coarse(
    hfarray_cxx<H> bitstream, hfarray_cxx<uint8_t> revbook,
    hfarray_cxx<M> par_nbit, hfarray_cxx<M> par_entry, hfpar_description hfpar,
    hfarray_cxx<E> out, float* time_lossless, void* stream)
{
  auto const block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
  auto const grid_dim = psz_utils::get_npart(hfpar.pardeg, block_dim);

  if (TIMING) {
    CREATE_GPUEVENT_PAIR;
    START_GPUEVENT_RECORDING(stream);

    hf_decode_kernel<E, H, M>                                       //
        <<<grid_dim, block_dim, revbook.len, (GpuStreamT)stream>>>  //
        (bitstream.buf, revbook.buf, par_nbit.buf, par_entry.buf, revbook.len,
         hfpar.sublen, hfpar.pardeg, out.buf);

    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(GpuStreamSync(stream));

    TIME_ELAPSED_GPUEVENT(time_lossless);
    DESTROY_GPUEVENT_PAIR;
  }
  else {
    CHECK_GPU(GpuStreamSync(stream));
  }
}

#endif /* D3F59CBB_1CC2_441A_8ACD_E598BDD68687 */
