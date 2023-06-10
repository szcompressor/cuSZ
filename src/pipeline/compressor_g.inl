/**
 * @file compressor_impl.cuh
 * @author Jiannan Tian
 * @brief cuSZ compressor of the default path
 * @version 0.3
 * @date 2023-06-06
 * (create) 2020-02-12; (release) 2020-09-20;
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#ifndef A2519F0E_602B_4798_A8EF_9641123095D9
#define A2519F0E_602B_4798_A8EF_9641123095D9

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>

#include "compressor.hh"
#include "header.h"
#include "hf/hf.hh"
#include "kernel/lorenzo_all.hh"
#include "kernel/spv_gpu.hh"
#include "stat/stat_g.hh"
#include "utils/config.hh"
#include "utils/cuda_err.cuh"

using std::cout;
using std::endl;

#define PRINT_ENTRY(VAR)                                    \
  printf(                                                   \
      "%d %-*s:  %'10u\n", (int)cusz_header::VAR, 14, #VAR, \
      header.entry[cusz_header::VAR]);

namespace cusz {

#define TEMPLATE_TYPE template <class Combination>

TEMPLATE_TYPE
void Compressor<Combination>::destroy()
{
  if (codec) delete codec;

  if (d_freq) cudaFree(d_freq);
  if (d_errctrl) cudaFree(d_errctrl);
  if (d_outlier) cudaFree(d_outlier);
  if (d_spval) cudaFree(d_spval);
  if (d_spidx) cudaFree(d_spidx);
}

TEMPLATE_TYPE
Compressor<Combination>::~Compressor() { destroy(); }

//------------------------------------------------------------------------------

// TODO
TEMPLATE_TYPE
void Compressor<Combination>::init(cusz_context* config, bool dbg_print)
{
  codec = new Codec;
  init_detail(config, dbg_print);
}

TEMPLATE_TYPE
void Compressor<Combination>::init(cusz_header* config, bool dbg_print)
{
  codec = new Codec;
  init_detail(config, dbg_print);
}

// template <class T>
// void peek_devdata(T* d_arr, size_t num = 20)
// {
//   thrust::for_each(
//       thrust::device, d_arr, d_arr + num,
//       [=] __device__ __host__(const T i) { printf("%u\t", i); });
//   printf("\n");
// }

TEMPLATE_TYPE
void Compressor<Combination>::compress(
    cusz_context* config, T* uncompressed, BYTE*& compressed,
    size_t& compressed_len, cudaStream_t stream, bool dbg_print)
{
  auto const eb = config->eb;
  auto const radius = config->radius;
  auto const pardeg = config->vle_pardeg;
  // auto const codecs_in_use = config->codecs_in_use;
  // auto const nz_density_factor = config->nz_density_factor;

  if (dbg_print) {
    cout << "eb\t" << eb << endl;
    cout << "radius\t" << radius << endl;
    cout << "pardeg\t" << pardeg << endl;
    // cout << "codecs_in_use\t" << codecs_in_use << endl;
    // cout << "nz_density_factor\t" << nz_density_factor << endl;
  }

  auto div = [](auto whole, auto part) { return (whole - 1) / part + 1; };

  data_len3 = dim3(config->x, config->y, config->z);

  // auto codec_force_fallback = config->codec_force_fallback();
  // header.codecs_in_use = codecs_in_use;
  // header.nz_density_factor = nz_density_factor;

  BYTE* d_codec_out{nullptr};
  size_t codec_outlen{0};

  size_t data_len = config->x * config->y * config->z;
  auto booklen = radius * 2;

  auto sublen = div(data_len, pardeg);

  auto update_header = [&]() {
    header.x = data_len3.x;
    header.y = data_len3.y;
    header.z = data_len3.z;
    header.w = 1;  // placeholder
    header.radius = radius;
    header.vle_pardeg = pardeg;
    header.eb = eb;
    header.byte_vle = use_fallback_codec ? 8 : 4;
    header.splen = splen;
  };

  /******************************************************************************/

  psz_comp_lorenzo_2output<T, E, FP>(
      uncompressed, data_len3, eb, radius, d_errctrl, d_outlier, nullptr,
      nullptr, &time_pred, stream);
  psz::stat::histogram<E>(
      d_errctrl, datalen_linearized, d_freq, booklen, &time_hist, stream);
  codec->build_codebook(d_freq, booklen, stream);
  codec->encode(
      d_errctrl, datalen_linearized, &d_codec_out, &codec_outlen, stream);
  psz::spv_gather<T, M>(
      d_outlier, datalen_linearized, d_spval, d_spidx, &splen, &time_sp,
      stream);

  /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

  /******************************************************************************/

  update_header();

  merge_subfiles(d_codec_out, codec_outlen, d_spval, d_spidx, splen, stream);

  // output
  compressed_len = psz_utils::filesize(&header);
  compressed = d_reserved_compressed;

  collect_compress_timerecord();

  // considering that codec can be consecutively in use, and can compress data
  // of different huff-byte
  use_fallback_codec = false;
}

TEMPLATE_TYPE
void Compressor<Combination>::clear_buffer()
{
  cudaMemset(d_errctrl, 0, sizeof(E) * datalen_linearized);
  cudaMemset(d_outlier, 0, sizeof(T) * datalen_linearized);
  cudaMemset(d_spval, 0, sizeof(T) * datalen_linearized * _23june_density);
  cudaMemset(d_spidx, 0, sizeof(M) * datalen_linearized * _23june_density);
  codec->clear_buffer();
}

TEMPLATE_TYPE
void Compressor<Combination>::decompress(
    cusz_header* header, BYTE* in_compressed, T* out_decompressed,
    cudaStream_t stream, bool dbg_print)
{
  // TODO host having copy of header when compressing
  if (not header) {
    header = new cusz_header;
    CHECK_CUDA(cudaMemcpyAsync(
        header, in_compressed, sizeof(cusz_header), cudaMemcpyDeviceToHost,
        stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  data_len3 = dim3(header->x, header->y, header->z);

  use_fallback_codec = header->byte_vle == 8;
  double const eb = header->eb;
  int const radius = header->radius;

  // The inputs of components are from `compressed`.
  auto d_vle = (BYTE*)(in_compressed + header->entry[cusz_header::VLE]);

  auto spval_nbyte = header->splen * sizeof(T);
  auto local_d_spval = (T*)(in_compressed + header->entry[cusz_header::SPFMT]);
  auto local_d_spidx =
      (uint32_t*)(in_compressed + header->entry[cusz_header::SPFMT] +
                  spval_nbyte);

  // wire and aliasing
  auto d_outlier = out_decompressed;
  auto d_outlier_xdata = out_decompressed;

  psz::spv_scatter<T, M>(
      local_d_spval, local_d_spidx, header->splen, d_outlier, &time_sp,
      stream);

  codec->decode(d_vle, d_errctrl);
  psz_decomp_lorenzo<T, E, FP>(
      d_errctrl, data_len3, d_outlier_xdata, nullptr, 0, eb, radius,
      d_outlier_xdata, &time_pred, stream);

  collect_decompress_timerecord();

  // clear state for the next decompression after reporting
  use_fallback_codec = false;
}

// public getter
TEMPLATE_TYPE
void Compressor<Combination>::export_header(cusz_header& ext_header)
{
  ext_header = header;
}

TEMPLATE_TYPE
void Compressor<Combination>::export_header(cusz_header* ext_header)
{
  *ext_header = header;
}

TEMPLATE_TYPE
void Compressor<Combination>::export_timerecord(TimeRecord* ext_timerecord)
{
  if (ext_timerecord) *ext_timerecord = timerecord;
}

TEMPLATE_TYPE
template <class CONFIG>
void Compressor<Combination>::init_detail(CONFIG* config, bool dbg_print)
{
  const auto radius = config->radius;
  const auto pardeg = config->vle_pardeg;
  // const auto density_factor = config->nz_density_factor;
  // const auto codec_config = config->codecs_in_use;
  const auto max_booklen = radius * 2;
  const auto x = config->x;
  const auto y = config->y;
  const auto z = config->z;

  datalen_linearized = x * y * z;

  {
    auto reserved_splen = datalen_linearized * _23june_density;
    cudaMalloc(&d_spval, reserved_splen * sizeof(T));
    cudaMemset(d_spval, 0x0, reserved_splen * sizeof(T));

    cudaMalloc(&d_spidx, reserved_splen * sizeof(uint32_t));
    cudaMemset(d_spval, 0x0, reserved_splen * sizeof(uint32_t));
  }

  codec->init(datalen_linearized, max_booklen, pardeg, dbg_print);

  // 23-june externalize buffers of spcodec obj
  {
    auto bytes = sizeof(uint32_t) * max_booklen;
    cudaMalloc(&d_freq, bytes);
    cudaMemset(d_freq, 0x0, bytes);
  }

  // 23-june externalize buffers of prediction obj
  {
    auto bytes1 = sizeof(E) * datalen_linearized;
    cudaMalloc(&d_errctrl, bytes1);
    cudaMemset(d_errctrl, 0x0, bytes1);

    auto bytes2 = sizeof(T) * datalen_linearized;
    cudaMalloc(&d_outlier, bytes2);
    cudaMemset(d_outlier, 0x0, bytes2);
  }

  CHECK_CUDA(
      cudaMalloc(&d_reserved_compressed, datalen_linearized * sizeof(T) / 2));
}

TEMPLATE_TYPE
void Compressor<Combination>::collect_compress_timerecord()
{
#define COLLECT_TIME(NAME, TIME) \
  timerecord.push_back({const_cast<const char*>(NAME), TIME});

  if (not timerecord.empty()) timerecord.clear();

  COLLECT_TIME("predict", time_pred);
  COLLECT_TIME("histogram", time_hist);
  COLLECT_TIME("book", codec->time_book());
  COLLECT_TIME("huff-enc", codec->time_lossless());
  COLLECT_TIME("outlier", time_sp);
}

TEMPLATE_TYPE
void Compressor<Combination>::collect_decompress_timerecord()
{
  if (not timerecord.empty()) timerecord.clear();

  COLLECT_TIME("outlier", time_sp);
  COLLECT_TIME("huff-dec", codec->time_lossless());
  COLLECT_TIME("predict", time_pred);
}

TEMPLATE_TYPE
void Compressor<Combination>::merge_subfiles(
    BYTE* d_codec_out, size_t codec_outlen, T* _d_spval, M* _d_spidx,
    size_t _splen, cudaStream_t stream)
{
  header.self_bytes = sizeof(cusz_header);
  uint32_t nbyte[cusz_header::END];
  nbyte[cusz_header::HEADER] = sizeof(cusz_header);
  nbyte[cusz_header::ANCHOR] = 0;
  nbyte[cusz_header::VLE] = sizeof(BYTE) * codec_outlen;
  nbyte[cusz_header::SPFMT] = (sizeof(T) + sizeof(M)) * _splen;

  header.entry[0] = 0;
  // *.END + 1; need to know the ending position
  for (auto i = 1; i < cusz_header::END + 1; i++)
    header.entry[i] = nbyte[i - 1];
  for (auto i = 1; i < cusz_header::END + 1; i++)
    header.entry[i] += header.entry[i - 1];

  CHECK_CUDA(cudaMemcpyAsync(
      d_reserved_compressed, &header, sizeof(header), cudaMemcpyHostToDevice,
      stream));

  {
    auto dst = d_reserved_compressed + header.entry[cusz_header::VLE];
    auto src = reinterpret_cast<BYTE*>(d_codec_out);
    CHECK_CUDA(cudaMemcpyAsync(
        dst, src, nbyte[cusz_header::VLE], cudaMemcpyDeviceToDevice, stream));
  }

  {
    // copy spval
    auto part1_nbyte = sizeof(T) * _splen;
    {
      auto dst = d_reserved_compressed + header.entry[cusz_header::SPFMT];
      auto src = reinterpret_cast<BYTE*>(_d_spval);
      CHECK_CUDA(cudaMemcpyAsync(
          dst, src, part1_nbyte, cudaMemcpyDeviceToDevice, stream));
    }
    // copy spidx
    {
      auto dst = d_reserved_compressed + header.entry[cusz_header::SPFMT] +
                 part1_nbyte;
      auto src = reinterpret_cast<BYTE*>(_d_spidx);
      CHECK_CUDA(cudaMemcpyAsync(
          dst, src, sizeof(uint32_t) * _splen, cudaMemcpyDeviceToDevice,
          stream));
    }
  }

  /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));
}

}  // namespace cusz

#undef PRINT_ENTRY
#undef COLLECT_TIME
#undef TEMPLATE_TYPE

#endif /* A2519F0E_602B_4798_A8EF_9641123095D9 */
