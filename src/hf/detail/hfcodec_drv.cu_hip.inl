/**
 * @file hfcodec_drv.cu_hip.inl
 * @author Jiannan Tian
 * @brief kernel wrappers; launching Huffman kernels
 * @version 0.3
 * @date 2022-11-02
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "cusz/type.h"
#include "hf/hfbk.cu.hh"
#include "hf/hfcodec.hh"
#include "hfcodec.cu_hip.inl"

template <typename E, typename H, typename M>
void psz::hf_encode_coarse_rev2(
    E* uncompressed, size_t const len, hf_book* book_desc,
    hf_bitstream* bitstream_desc, size_t* outlen_nbit, size_t* outlen_ncell,
    float* time_lossless, void* stream)
{
  auto div = [](auto whole, auto part) -> uint32_t {
    if (whole == 0) throw std::runtime_error("Dividend is zero.");
    if (part == 0) throw std::runtime_error("Divisor is zero.");
    return (whole - 1) / part + 1;
  };
  static const int BLOCK_DIM_ENCODE = 256;
  static const int BLOCK_DIM_DEFLATE = 256;

  CREATE_GPUEVENT_PAIR;

  H* d_buffer = (H*)bitstream_desc->buffer;
  H* d_bitstream = (H*)bitstream_desc->bitstream;
  H* d_book = (H*)book_desc->book;
  int const booklen = book_desc->bklen;
  int const sublen = bitstream_desc->sublen;
  int const pardeg = bitstream_desc->pardeg;
  int const numSMs = bitstream_desc->numSMs;
  // uint32_t* d_freq      = book_desc->freq;

  auto d_par_nbit = (M*)bitstream_desc->d_metadata->bits;
  auto d_par_ncell = (M*)bitstream_desc->d_metadata->cells;
  auto d_par_entry = (M*)bitstream_desc->d_metadata->entries;

  auto h_par_nbit = (M*)bitstream_desc->h_metadata->bits;
  auto h_par_ncell = (M*)bitstream_desc->h_metadata->cells;
  auto h_par_entry = (M*)bitstream_desc->h_metadata->entries;

// #ifdef PSZ_USE_HIP
//   cout << "[psz::dbg::hfcodec] entering phase1" << endl;
// #endif

  /* phase 1 */
  {
    auto block_dim = BLOCK_DIM_ENCODE;
    auto grid_dim = div(len, block_dim);

    START_GPUEVENT_RECORDING(stream);

    psz::detail::hf_encode_phase1_fill<E, H>                            //
        <<<8 * numSMs, 256, sizeof(H) * booklen, (GpuStreamT)stream>>>  //
        (uncompressed, len, d_book, booklen, d_buffer);

    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(GpuStreamSync(stream));

    float stage_time;
    TIME_ELAPSED_GPUEVENT(&stage_time);
    if (time_lossless) *time_lossless += stage_time;
  }

// #ifdef PSZ_USE_HIP
//   cout << "[psz::dbg::hfcodec] entering phase2" << endl;
// #endif

  /* phase 2 */
  {
    auto block_dim = BLOCK_DIM_DEFLATE;
    auto grid_dim = div(pardeg, block_dim);

    START_GPUEVENT_RECORDING(stream);

    psz::detail::hf_encode_phase2_deflate<H>              //
        <<<grid_dim, block_dim, 0, (GpuStreamT)stream>>>  //
        (d_buffer, len, d_par_nbit, d_par_ncell, sublen, pardeg);

    STOP_GPUEVENT_RECORDING(stream);
    CHECK_GPU(GpuStreamSync(stream));

    float stage_time;
    TIME_ELAPSED_GPUEVENT(&stage_time);
    if (time_lossless) *time_lossless += stage_time;
  }

// #ifdef PSZ_USE_HIP
//   cout << "[psz::dbg::hfcodec] entering phase3" << endl;
// #endif

  /* phase 3 */
  {
    CHECK_GPU(GpuMemcpyAsync(
        h_par_nbit, d_par_nbit, pardeg * sizeof(M), GpuMemcpyD2H,
        (GpuStreamT)stream));
    CHECK_GPU(GpuMemcpyAsync(
        h_par_ncell, d_par_ncell, pardeg * sizeof(M), GpuMemcpyD2H,
        (GpuStreamT)stream));
    CHECK_GPU(GpuStreamSync(stream));

    // for (auto i = 0; i < pardeg; i++)
    //   cout << i << " " << h_par_nbit[i] << endl;

    memcpy(h_par_entry + 1, h_par_ncell, (pardeg - 1) * sizeof(M));
    for (auto i = 1; i < pardeg; i++)
      h_par_entry[i] += h_par_entry[i - 1];  // inclusive scan

    CHECK_GPU(GpuMemcpyAsync(
        d_par_entry, h_par_entry, pardeg * sizeof(M), GpuMemcpyH2D,
        (GpuStreamT)stream));
    CHECK_GPU(GpuStreamSync(stream));
  }

#ifdef PSZ_USE_HIP
  // cout << "[psz::dbg::hfcodec] pardeg=" << pardeg << endl;
  // cout << "[psz::dbg::hfcodec] phase3 ends" << endl;

  // for (auto i = 1; i < 100; i++) {
  //   cout << "[psz::dbg::hfcodec] check h_par_ncell (first 100 entries): (idx) " << i << "\t" << h_par_ncell[i] << "\n";
  // }
  // cout << "[psz::dbg::hfcodec] entering phase4" << endl;
#endif

  /* phase 4 */
  {
    START_GPUEVENT_RECORDING(stream);

    psz::detail::hf_encode_phase4_concatenate<H, M>
        <<<pardeg, 128, 0, (GpuStreamT)stream>>>  //
        (d_buffer, d_par_entry, d_par_ncell, sublen, d_bitstream);

    STOP_GPUEVENT_RECORDING(stream);

    CHECK_GPU(GpuStreamSync(stream));

    float stage_time;
    TIME_ELAPSED_GPUEVENT(&stage_time);
    if (time_lossless) *time_lossless += stage_time;
  }

#ifdef PSZ_USE_HIP
    // cout << "[psz::dbg::hfcodec] entering phase5" << endl;
#endif

  /* phase 5: gather out sizes without memcpy */
  {
    if (outlen_nbit)
      *outlen_nbit =
          std::accumulate(h_par_nbit, h_par_nbit + pardeg, (size_t)0);
    if (outlen_ncell)
      *outlen_ncell =
          std::accumulate(h_par_ncell, h_par_ncell + pardeg, (size_t)0);
  }

#ifdef PSZ_USE_HIP
    // cout << "[psz::dbg::hfcodec] end of encoding" << endl;
#endif

}

template <typename E, typename H, typename M>
void psz::hf_decode_coarse(
    H* d_bitstream, uint8_t* d_revbook, int const revbook_nbyte, M* d_par_nbit,
    M* d_par_entry, int const sublen, int const pardeg, E* out_decompressed,
    float* time_lossless, void* stream)
{
  auto const block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
  auto const grid_dim = psz_utils::get_npart(pardeg, block_dim);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  hf_decode_kernel<E, H, M>                                         //
      <<<grid_dim, block_dim, revbook_nbyte, (GpuStreamT)stream>>>  //
      (d_bitstream, d_revbook, d_par_nbit, d_par_entry, revbook_nbyte, sublen,
       pardeg, out_decompressed);

  STOP_GPUEVENT_RECORDING(stream);
  GpuStreamSync(stream);

  TIME_ELAPSED_GPUEVENT(time_lossless);
  DESTROY_GPUEVENT_PAIR;
}
