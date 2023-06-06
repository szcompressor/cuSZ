/**
 * @file compressor_impl.cuh
 * @author Jiannan Tian
 * @brief cuSZ compressor of the default path
 * @version 0.3
 * @date 2021-10-05
 * (create) 2020-02-12; (release) 2020-09-20;
 * (rev.1) 2021-01-16; (rev.2) 2021-07-12; (rev.3) 2021-09-06; (rev.4)
 * 2021-10-05
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#ifndef CUSZ_DEFAULT_PATH_CUH
#define CUSZ_DEFAULT_PATH_CUH

#include <cuda_runtime.h>
// #include <thrust/device_ptr.h>
// #include <thrust/execution_policy.h>
#include <iostream>

#include "compressor.hh"
#include "header.h"
#include "hf/hf.hh"
#include "kernel/lorenzo_all.hh"
#include "spcodec.inl"
#include "stat/stat_g.hh"
#include "utils/cuda_err.cuh"

#define DEFINE_DEV(VAR, TYPE) TYPE* d_##VAR{nullptr};
#define DEFINE_HOST(VAR, TYPE) TYPE* h_##VAR{nullptr};
#define FREEDEV(VAR) CHECK_CUDA(cudaFree(d_##VAR));
#define FREEHOST(VAR) CHECK_CUDA(cudaFreeHost(h_##VAR));

#define PRINT_ENTRY(VAR)                               \
  printf(                                              \
      "%d %-*s:  %'10u\n", (int)Header::VAR, 14, #VAR, \
      header.entry[Header::VAR]);

#define DEVICE2DEVICE_COPY(VAR, FIELD)                                      \
  if (nbyte[Header::FIELD] != 0 and VAR != nullptr) {                       \
    auto dst = d_reserved_compressed + header.entry[Header::FIELD];         \
    auto src = reinterpret_cast<BYTE*>(VAR);                                \
    CHECK_CUDA(cudaMemcpyAsync(                                             \
        dst, src, nbyte[Header::FIELD], cudaMemcpyDeviceToDevice, stream)); \
  }

#define ACCESSOR(SYM, TYPE) \
  reinterpret_cast<TYPE*>(in_compressed + header->entry[Header::SYM])

namespace cusz {

#define TEMPLATE_TYPE template <class Combination>

TEMPLATE_TYPE
uint32_t Compressor<Combination>::get_len_data()
{
  return data_len3.x * data_len3.y * data_len3.z;
}

TEMPLATE_TYPE
void Compressor<Combination>::destroy()
{
  if (spcodec) delete spcodec;
  if (codec) delete codec;

  if (d_freq) cudaFree(d_freq);
  if (_23june_d_errctrl) cudaFree(_23june_d_errctrl);
  if (_23june_d_outlier) cudaFree(_23june_d_outlier);
}

TEMPLATE_TYPE
Compressor<Combination>::~Compressor() { destroy(); }

//------------------------------------------------------------------------------

// TODO
TEMPLATE_TYPE
void Compressor<Combination>::init(Context* config, bool dbg_print)
{
  spcodec = new Spcodec;
  codec = new Codec;
  init_detail(config, dbg_print);
}

TEMPLATE_TYPE
void Compressor<Combination>::init(Header* config, bool dbg_print)
{
  spcodec = new Spcodec;
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
    Context* config, T* uncompressed, BYTE*& compressed,
    size_t& compressed_len, cudaStream_t stream, bool dbg_print)
{
  auto const eb = config->eb;
  auto const radius = config->radius;
  auto const pardeg = config->vle_pardeg;
  auto const codecs_in_use = config->codecs_in_use;
  auto const nz_density_factor = config->nz_density_factor;

  if (dbg_print) {
    std::cout << "eb\t" << eb << endl;
    std::cout << "radius\t" << radius << endl;
    std::cout << "pardeg\t" << pardeg << endl;
    std::cout << "codecs_in_use\t" << codecs_in_use << endl;
    std::cout << "nz_density_factor\t" << nz_density_factor << endl;
  }

  auto div = [](auto whole, auto part) { return (whole - 1) / part + 1; };

  data_len3 = dim3(config->x, config->y, config->z);
  auto codec_force_fallback = config->codec_force_fallback();

  header.codecs_in_use = codecs_in_use;
  header.nz_density_factor = nz_density_factor;

  T* d_anchor{nullptr};
  // E* d_errctrl{nullptr};
  // T* d_outlier{nullptr};
  // uint32_t* d_outlier_idx{nullptr};
  BYTE* d_spfmt{nullptr};
  size_t spfmt_outlen{0};

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
  };

  /******************************************************************************/

  psz_comp_lorenzo_2output<T, E, FP>(
      uncompressed, data_len3, eb, radius, _23june_d_errctrl,
      _23june_d_outlier, nullptr, nullptr, &time_pred, stream);
  psz::stat::histogram<E>(
      _23june_d_errctrl, _23june_datalen, d_freq, booklen, &time_hist, stream);
  codec->build_codebook(d_freq, booklen, stream);
  codec->encode(
      _23june_d_errctrl, _23june_datalen, &d_codec_out, &codec_outlen, stream);
  spcodec->encode(
      _23june_d_outlier, _23june_datalen, d_spfmt, spfmt_outlen, stream,
      dbg_print);

  /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

  /******************************************************************************/

  update_header();

  auto _23june_anchorlen = 0;
  subfile_collect(
      d_anchor, _23june_anchorlen,  //
      d_codec_out, codec_outlen,    //
      d_spfmt, spfmt_outlen,        //
      stream, dbg_print);

  // output
  compressed_len = ConfigHelper::get_filesize(&header);
  compressed = d_reserved_compressed;

  collect_compress_timerecord();

  // considering that codec can be consecutively in use, and can compress data
  // of different huff-byte
  use_fallback_codec = false;
}

TEMPLATE_TYPE
void Compressor<Combination>::clear_buffer()
{
  {
    cudaMemset(_23june_d_errctrl, 0, sizeof(E) * _23june_datalen);
    cudaMemset(_23june_d_outlier, 0, sizeof(T) * _23june_datalen);
  }
  codec->clear_buffer();
  spcodec->clear_buffer();
}

TEMPLATE_TYPE
void Compressor<Combination>::decompress(
    Header* header, BYTE* in_compressed, T* out_decompressed,
    cudaStream_t stream, bool dbg_print)
{
  // TODO host having copy of header when compressing
  if (not header) {
    header = new Header;
    CHECK_CUDA(cudaMemcpyAsync(
        header, in_compressed, sizeof(Header), cudaMemcpyDeviceToHost,
        stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  data_len3 = dim3(header->x, header->y, header->z);

  use_fallback_codec = header->byte_vle == 8;
  double const eb = header->eb;
  int const radius = header->radius;

  // The inputs of components are from `compressed`.
  auto d_anchor = ACCESSOR(ANCHOR, T);
  auto d_vle = ACCESSOR(VLE, BYTE);
  auto d_sp = ACCESSOR(SPFMT, BYTE);

  // wire the workspace
  auto d_errctrl = _23june_d_errctrl;  // reuse space

  // wire and aliasing
  auto d_outlier = out_decompressed;
  auto d_outlier_xdata = out_decompressed;

  spcodec->decode(d_sp, d_outlier, stream);
  codec->decode(d_vle, d_errctrl);
  psz_decomp_lorenzo<T, E, FP>(
      d_errctrl, data_len3, d_outlier_xdata, nullptr, 0, eb, radius,
      d_outlier_xdata, &time_pred, stream);

  // 23-06-05 simplifed

  collect_decompress_timerecord();

  // clear state for the next decompression after reporting
  use_fallback_codec = false;
}

// public getter
TEMPLATE_TYPE
void Compressor<Combination>::export_header(Header& ext_header)
{
  ext_header = header;
}

TEMPLATE_TYPE
void Compressor<Combination>::export_header(Header* ext_header)
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
  const auto cfg_radius = config->radius;
  const auto cfg_pardeg = config->vle_pardeg;
  const auto density_factor = config->nz_density_factor;
  const auto codec_config = config->codecs_in_use;
  const auto cfg_max_booklen = cfg_radius * 2;
  const auto x = config->x;
  const auto y = config->y;
  const auto z = config->z;

  _23june_datalen = x * y * z;

  spcodec->init(_23june_datalen, density_factor, dbg_print);
  codec->init(_23june_datalen, cfg_max_booklen, cfg_pardeg, dbg_print);

  {
    auto bytes = sizeof(uint32_t) * cfg_max_booklen;
    cudaMalloc(&d_freq, bytes);
    cudaMemset(d_freq, 0x0, bytes);
  }

  // 23-june externalize buffers of prediction obj
  {
    auto bytes1 = sizeof(E) * _23june_datalen;
    cudaMalloc(&_23june_d_errctrl, bytes1);
    cudaMemset(_23june_d_errctrl, 0x0, bytes1);

    auto bytes2 = sizeof(T) * _23june_datalen;
    cudaMalloc(&_23june_d_outlier, bytes2);
    cudaMemset(_23june_d_outlier, 0x0, bytes2);
  }

  CHECK_CUDA(
      cudaMalloc(&d_reserved_compressed, _23june_datalen * sizeof(T) / 2));
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
  COLLECT_TIME("outlier", spcodec->get_time_elapsed());
}

TEMPLATE_TYPE
void Compressor<Combination>::collect_decompress_timerecord()
{
  if (not timerecord.empty()) timerecord.clear();

  COLLECT_TIME("outlier", spcodec->get_time_elapsed());
  COLLECT_TIME("huff-dec", codec->time_lossless());
  COLLECT_TIME("predict", time_pred);
}

TEMPLATE_TYPE
void Compressor<Combination>::subfile_collect(
    T* d_anchor, size_t anchor_len, BYTE* d_codec_out, size_t codec_outlen,
    BYTE* d_spfmt_out, size_t spfmt_outlen, cudaStream_t stream,
    bool dbg_print)
{
  header.self_bytes = sizeof(Header);
  uint32_t nbyte[Header::END];
  nbyte[Header::HEADER] = sizeof(Header);
  nbyte[Header::ANCHOR] = sizeof(T) * anchor_len;
  nbyte[Header::VLE] = sizeof(BYTE) * codec_outlen;
  nbyte[Header::SPFMT] = sizeof(BYTE) * spfmt_outlen;

  header.entry[0] = 0;
  // *.END + 1; need to know the ending position
  for (auto i = 1; i < Header::END + 1; i++) {
    header.entry[i] = nbyte[i - 1];
  }
  for (auto i = 1; i < Header::END + 1; i++) {
    header.entry[i] += header.entry[i - 1];
  }

  auto debug_header_entry = [&]() {
    printf("\nsubfile collect in compressor:\n");
    printf("  ENTRIES\n");

    PRINT_ENTRY(HEADER);
    PRINT_ENTRY(ANCHOR);
    PRINT_ENTRY(VLE);
    PRINT_ENTRY(SPFMT);
    PRINT_ENTRY(END);
    printf("\n");
  };

  if (dbg_print) debug_header_entry();

  CHECK_CUDA(cudaMemcpyAsync(
      d_reserved_compressed, &header, sizeof(header), cudaMemcpyHostToDevice,
      stream));

  DEVICE2DEVICE_COPY(d_anchor, ANCHOR)
  DEVICE2DEVICE_COPY(d_codec_out, VLE)
  DEVICE2DEVICE_COPY(d_spfmt_out, SPFMT)

  /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));
}

}  // namespace cusz

#undef FREEDEV
#undef FREEHOST
#undef DEFINE_DEV
#undef DEFINE_HOST
#undef DEVICE2DEVICE_COPY
#undef PRINT_ENTRY
#undef ACCESSOR
#undef COLLECT_TIME
#undef TEMPLATE_TYPE

#endif
