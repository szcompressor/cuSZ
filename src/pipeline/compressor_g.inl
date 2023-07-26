/**
 * @file compressor_g.inl
 * @author Jiannan Tian
 * @brief compression pipeline
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
#include "kernel/l23.hh"
#include "kernel/spv_gpu.hh"
#include "layout.h"
#include "stat/stat.hh"
#include "utils/config.hh"
#include "utils/cuda_err.cuh"
#include "utils2/memseg_cxx.hh"

using std::cout;
using std::endl;

#define PRINT_ENTRY(VAR)                                    \
  printf(                                                   \
      "%d %-*s:  %'10u\n", (int)cusz_header::VAR, 14, #VAR, \
      header.entry[cusz_header::VAR]);

namespace cusz {

template <class C>
Compressor<C>* Compressor<C>::destroy()
{
  if (codec) delete codec;

  delete freq;
  delete errctrl;
  delete outlier;
  delete spval;
  delete spidx;
  delete compressed;

  return this;
}

template <class C>
Compressor<C>::~Compressor()
{
  destroy();
}

//------------------------------------------------------------------------------

// TODO
template <class C>
Compressor<C>* Compressor<C>::init(cusz_context* config, bool dbg_print)
{
  codec = new Codec;
  init_detail(config, dbg_print);
  return this;
}

template <class C>
Compressor<C>* Compressor<C>::init(cusz_header* config, bool dbg_print)
{
  codec = new Codec;
  init_detail(config, dbg_print);
  return this;
}

template <class C>
template <class CONFIG>
Compressor<C>* Compressor<C>::init_detail(CONFIG* config, bool debug)
{
  const auto radius = config->radius;
  const auto pardeg = config->vle_pardeg;
  // const auto density_factor = config->nz_density_factor;
  // const auto codec_config = config->codecs_in_use;
  const auto booklen = radius * 2;
  const auto x = config->x;
  const auto y = config->y;
  const auto z = config->z;

  len = x * y * z;

  codec->init(len, booklen, pardeg, debug);

  compressed = new pszmem_cxx<BYTE>(len * 2, 1, 1, "compressed");
  errctrl = new pszmem_cxx<E>(x, y, z, "errctrl");
  outlier = new pszmem_cxx<T>(x, y, z, "errctrl");
  freq = new pszmem_cxx<uint32_t>(booklen, 1, 1, "freq");
  spval = new pszmem_cxx<T>(x, y, z, "spval");
  spidx = new pszmem_cxx<uint32_t>(x, y, z, "spidx");

  compressed->control({Malloc, MallocHost});
  errctrl->control({Malloc, MallocHost});
  outlier->control({Malloc, MallocHost});
  freq->control({Malloc, MallocHost});
  spval->control({Malloc, MallocHost});
  spidx->control({Malloc, MallocHost});

  return this;
}

template <class C>
Compressor<C>* Compressor<C>::compress(
    cusz_context* config, T* in, BYTE*& out, size_t& outlen,
    cudaStream_t stream, bool dbg_print)
{
  auto const eb = config->eb;
  auto const radius = config->radius;
  auto const pardeg = config->vle_pardeg;

  if (dbg_print) {
    cout << "eb\t" << eb << endl;
    cout << "radius\t" << radius << endl;
    cout << "pardeg\t" << pardeg << endl;
  }

  auto div = [](auto whole, auto part) { return (whole - 1) / part + 1; };

  len3 = dim3(config->x, config->y, config->z);

  BYTE* d_codec_out{nullptr};
  size_t codec_outlen{0};

  size_t data_len = config->x * config->y * config->z;
  auto booklen = radius * 2;

  auto sublen = div(data_len, pardeg);

  auto update_header = [&]() {
    header.x = len3.x;
    header.y = len3.y;
    header.z = len3.z;
    header.w = 1;  // placeholder
    header.radius = radius;
    header.vle_pardeg = pardeg;
    header.eb = eb;
    header.splen = splen;
    // header.byte_vle = use_fallback_codec ? 8 : 4;
  };

  /******************************************************************************/

  psz_comp_l23<T, E, FP>(
      in, len3, eb, radius, errctrl->dptr(), outlier->dptr(), &time_pred,
      stream);
  psz::stat::histogram<psz_policy::CUDA, E>(
      errctrl->dptr(), len, freq->dptr(), booklen, &time_hist, stream);
  codec->build_codebook(freq->dptr(), booklen, stream);
  codec->encode(errctrl->dptr(), len, &d_codec_out, &codec_outlen, stream);
  psz::spv_gather<T, M>(
      outlier->dptr(), len, spval->dptr(), spidx->dptr(), &splen, &time_sp,
      stream);

  /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

  /******************************************************************************/

  update_header();

  merge_subfiles(
      d_codec_out, codec_outlen, spval->dptr(), spidx->dptr(), splen, stream);

  // output
  outlen = psz_utils::filesize(&header);
  compressed->m->len = outlen;
  compressed->m->bytes = outlen;
  out = compressed->dptr();

  collect_comp_time();

  // TODO fallback hendling
  // use_fallback_codec = false;

  return this;
}

template <class C>
Compressor<C>* Compressor<C>::merge_subfiles(
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
  for (auto i = 1; i < Header::END + 1; i++) header.entry[i] = nbyte[i - 1];
  for (auto i = 1; i < Header::END + 1; i++)
    header.entry[i] += header.entry[i - 1];

  auto D2D = cudaMemcpyDeviceToDevice;
  // TODO no need to copy header to device
  CHECK_CUDA(cudaMemcpyAsync(
      compressed->dptr(), &header, sizeof(header), cudaMemcpyHostToDevice,
      stream));

  // device-side copy
  auto dst1 = compressed->dptr() + header.entry[Header::VLE];
  auto src1 = d_codec_out;
  CHECK_CUDA(cudaMemcpyAsync(dst1, src1, nbyte[Header::VLE], D2D, stream));

  // copy spval
  auto part1_nbyte = sizeof(T) * _splen;
  auto dst2 = compressed->dptr() + header.entry[Header::SPFMT];
  auto src2 = _d_spval;
  CHECK_CUDA(cudaMemcpyAsync(dst2, src2, part1_nbyte, D2D, stream));
  // copy spidx
  auto dst3 = compressed->dptr() + header.entry[Header::SPFMT] + part1_nbyte;
  auto src3 = _d_spidx;
  CHECK_CUDA(
      cudaMemcpyAsync(dst3, src3, sizeof(uint32_t) * _splen, D2D, stream));

  /* debug */ CHECK_CUDA(cudaStreamSynchronize(stream));

  return this;
}

template <class C>
Compressor<C>* Compressor<C>::dump_intermediate(
    std::vector<pszmem_dump> list, char const* basename)
{
  for (auto& i : list) {
    char __[256];

    auto ofn = [&](char const* suffix) {
      strcpy(__, basename);
      strcat(__, suffix);
      return __;
    };

    // TODO check if compressed len updated
    if (i == PszArchive)
      compressed->control({H2D})->file(ofn(".psz_archive"), ToFile);
    else if (i == PszQuant)
      errctrl->control({H2D})->file(ofn(".psz_quant"), ToFile);
    else if (i == PszHist)
      freq->control({H2D})->file(ofn(".psz_hist"), ToFile);
    else if (i == PszSpVal)
      spval->control({H2D})->file(ofn(".psz_spval"), ToFile);
    else if (i == PszSpIdx)
      spidx->control({H2D})->file(ofn(".psz_spidx"), ToFile);
    else if (i > PszHf______ and i < END)
      codec->dump_intermediate({i}, basename);
    else
      printf("[psz::dump] not a valid segment to dump.");
  }

  return this;
}

template <class C>
Compressor<C>* Compressor<C>::clear_buffer()
{
  codec->clear_buffer();

  errctrl->control({ClearDevice});
  outlier->control({ClearDevice});
  spval->control({ClearDevice});
  spidx->control({ClearDevice});
  compressed->control({ClearDevice});

  return this;
}

template <class C>
Compressor<C>* Compressor<C>::decompress(
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

  len3 = dim3(header->x, header->y, header->z);

  // use_fallback_codec = header->byte_vle == 8;
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
  auto local_d_outlier = out_decompressed;
  auto local_d_xdata = out_decompressed;

  psz::spv_scatter<T, M>(
      local_d_spval, local_d_spidx, header->splen, local_d_outlier, &time_sp,
      stream);

  codec->decode(d_vle, errctrl->dptr());
  psz_decomp_l23<T, E, FP>(
      errctrl->dptr(), len3, local_d_outlier, eb, radius, local_d_xdata,
      &time_pred, stream);

  collect_decomp_time();

  // clear state for the next decompression after reporting
  // use_fallback_codec = false;

  return this;
}

// public getter
template <class C>
Compressor<C>* Compressor<C>::export_header(cusz_header& ext_header)
{
  ext_header = header;
  return this;
}

template <class C>
Compressor<C>* Compressor<C>::export_header(cusz_header* ext_header)
{
  *ext_header = header;
  return this;
}

template <class C>
Compressor<C>* Compressor<C>::export_timerecord(TimeRecord* ext_timerecord)
{
  if (ext_timerecord) *ext_timerecord = timerecord;
  return this;
}

template <class C>
Compressor<C>* Compressor<C>::collect_comp_time()
{
#define COLLECT_TIME(NAME, TIME) \
  timerecord.push_back({const_cast<const char*>(NAME), TIME});

  if (not timerecord.empty()) timerecord.clear();

  COLLECT_TIME("predict", time_pred);
  COLLECT_TIME("histogram", time_hist);
  COLLECT_TIME("book", codec->time_book());
  COLLECT_TIME("huff-enc", codec->time_lossless());
  COLLECT_TIME("outlier", time_sp);

  return this;
}

template <class C>
Compressor<C>* Compressor<C>::collect_decomp_time()
{
  if (not timerecord.empty()) timerecord.clear();

  COLLECT_TIME("outlier", time_sp);
  COLLECT_TIME("huff-dec", codec->time_lossless());
  COLLECT_TIME("predict", time_pred);

  return this;
}

}  // namespace cusz

#undef PRINT_ENTRY
#undef COLLECT_TIME

#endif /* A2519F0E_602B_4798_A8EF_9641123095D9 */
