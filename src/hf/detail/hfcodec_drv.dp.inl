/**
 * @file hf_codecg_drv.inl
 * @author Jiannan Tian
 * @brief kernel wrappers; launching Huffman kernels
 * @version 0.3
 * @date 2022-11-02
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "cusz/type.h"
// #include "hf/hfbk.cu.hh"
#include <chrono>

#include "hf/hfcodec.hh"
#include "hfcodec.dp.inl"
#include "utils/timer.hh"

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

  CREATE_GPUEVENT_PAIR;

  auto queue = (sycl::queue*)stream;

  /* phase 1 */
  {
    auto block_dim = BLOCK_DIM_ENCODE;
    auto grid_dim = div(len, block_dim);

    // START_GPUEVENT_RECORDING(queue);

    sycl::event e = queue->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<char, 1> __codec_huffman_uninitialized_acc_ct1(
          sycl::range<1>(sizeof(H) * booklen), cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, 8 * numSMs) * sycl::range<3>(1, 1, 256),
              sycl::range<3>(1, 1, 256)),
          [=](sycl::nd_item<3> item_ct1) {
            psz::detail::hf_encode_phase1_fill<E, H>(
                uncompressed, len, d_book, booklen, d_buffer, item_ct1,
                __codec_huffman_uninitialized_acc_ct1.get_pointer());
          });
    });

    e.wait();
    // STOP_GPUEVENT_RECORDING(queue);

    float stage_time;
    SYCL_TIME_DELTA(e, stage_time);
    // TIME_ELAPSED_GPUEVENT(&stage_time);
    if (time_lossless) *time_lossless += stage_time;
  }

  /* phase 2 */
  {
    auto block_dim = BLOCK_DIM_DEFLATE;
    auto grid_dim = div(pardeg, block_dim);

    // START_GPUEVENT_RECORDING(queue);

    sycl::event e = queue->parallel_for(
        sycl::nd_range<3>(
            grid_dim * sycl::range<3>(1, 1, block_dim),
            sycl::range<3>(1, 1, block_dim)),
        [=](sycl::nd_item<3> item_ct1) {
          psz::detail::hf_encode_phase2_deflate<H>(
              d_buffer, len, d_par_nbit, d_par_ncell, sublen, pardeg,
              item_ct1);
        });

    // STOP_GPUEVENT_RECORDING(queue);
    e.wait();

    float stage_time;
    SYCL_TIME_DELTA(e, stage_time);
    // TIME_ELAPSED_GPUEVENT(&stage_time);

    if (time_lossless) *time_lossless += stage_time;
  }

  /* phase 3 */
  {
    queue  //
        ->memcpy(h_par_nbit, d_par_nbit, pardeg * sizeof(M))
        .wait_and_throw();
    queue  //
        ->memcpy(h_par_ncell, d_par_ncell, pardeg * sizeof(M))
        .wait_and_throw();

    memcpy(h_par_entry + 1, h_par_ncell, (pardeg - 1) * sizeof(M));
    for (auto i = 1; i < pardeg; i++)
      h_par_entry[i] += h_par_entry[i - 1];  // inclusive scan

    queue  //
        ->memcpy(d_par_entry, h_par_entry, pardeg * sizeof(M))
        .wait_and_throw();
  }

  /* phase 4 */
  {
    // START_GPUEVENT_RECORDING(queue);

    sycl::event e = queue->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, pardeg) * sycl::range<3>(1, 1, 128),
            sycl::range<3>(1, 1, 128)),
        [=](sycl::nd_item<3> item_ct1) {
          psz::detail::hf_encode_phase4_concatenate<H, M>(
              d_buffer, d_par_entry, d_par_ncell, sublen, d_bitstream,
              item_ct1);
        });
    e.wait();

    // STOP_GPUEVENT_RECORDING(queue);

    float stage_time;
    // TIME_ELAPSED_GPUEVENT(&stage_time);

    if (time_lossless) *time_lossless += stage_time;
  }

  /* phase 5: gather out sizes without memcpy */
  {
    if (outlen_nbit)
      *outlen_nbit =
          std::accumulate(h_par_nbit, h_par_nbit + pardeg, (size_t)0);
    if (outlen_ncell)
      *outlen_ncell =
          std::accumulate(h_par_ncell, h_par_ncell + pardeg, (size_t)0);
  }

  // DESTROY_GPUEVENT_PAIR;
}

template <typename E, typename H, typename M>
void psz::hf_decode_coarse(
    H* d_bitstream, uint8_t* d_revbook, int const revbook_nbyte, M* d_par_nbit,
    M* d_par_entry, int const sublen, int const pardeg, E* out_decompressed,
    float* time_lossless, void* stream)
{
  auto const block_dim = HuffmanHelper::BLOCK_DIM_DEFLATE;  // = deflating
  auto const grid_dim = psz_utils::get_npart(pardeg, block_dim);

  // CREATE_GPUEVENT_PAIR;

  auto queue = (sycl::queue*)stream;

  // START_GPUEVENT_RECORDING(queue);

  sycl::event e = queue->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
        sycl::range<1>(revbook_nbyte), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, grid_dim) * sycl::range<3>(1, 1, block_dim),
            sycl::range<3>(1, 1, block_dim)),
        [=](sycl::nd_item<3> item_ct1) {
          hf_decode_kernel<E, H, M>(
              d_bitstream, d_revbook, d_par_nbit, d_par_entry, revbook_nbyte,
              sublen, pardeg, out_decompressed, item_ct1,
              dpct_local_acc_ct1.get_pointer());
        });
  });

  e.wait();

  // STOP_GPUEVENT_RECORDING(queue);
  // TIME_ELAPSED_GPUEVENT(time_lossless);

  // DESTROY_GPUEVENT_PAIR;
}
