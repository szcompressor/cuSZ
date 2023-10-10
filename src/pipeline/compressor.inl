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

#include <stdexcept>

#include "busyheader.hh"
#include "compressor.hh"
#include "cusz/type.h"
#include "header.h"
#include "hf/hf.hh"
#include "kernel.hh"
#include "log.hh"
#include "mem.hh"
#include "port.hh"
#include "utils/config.hh"
#include "utils/err.hh"

#define COR          \
  template <class C> \
  Compressor<C>* Compressor<C>

// [psz::note] psz::histogram is left for evaluating purpose
// compared to psz::histsp
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_1API)
#define PSZ_HIST(...) psz::histsp<PROPER_GPU_BACKEND, E>(__VA_ARGS__);
#elif defined(PSZ_USE_HIP)
#define PSZ_HIST(...) psz::histogram<PROPER_GPU_BACKEND, E>(__VA_ARGS__);
#endif

namespace cusz {

COR::destroy()
{
  delete mem;
  delete codec;

  return this;
}

template <class C>
Compressor<C>::~Compressor()
{
  destroy();
}

//------------------------------------------------------------------------------

template <class C>
template <class CONFIG>
Compressor<C>* Compressor<C>::init(CONFIG* ctx, bool debug)
{
  codec = new Codec;

  const auto radius = ctx->radius;
  const auto pardeg = ctx->vle_pardeg;
  // const auto density_factor = ctx->nz_density_factor;
  // const auto codec_ctx = ctx->codecs_in_use;
  const auto booklen = radius * 2;
  const auto x = ctx->x;
  const auto y = ctx->y;
  const auto z = ctx->z;

  len = x * y * z;

  mem = new pszmempool_cxx<T, E, H>(x, radius, y, z);

  codec->init(mem->len, booklen, pardeg, debug);

  return this;
}

COR::compress_predict(pszctx* ctx, T* in, void* stream)
{
  auto spline_in_use = [&]() { return ctx->pred_type == Spline; };

  auto const eb = ctx->eb;
  auto const radius = ctx->radius;
  auto const pardeg = ctx->vle_pardeg;

  // [psz::note::TODO] compat layer or explicit macro
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  auto len3 = dim3(ctx->x, ctx->y, ctx->z);
#elif defined(PSZ_USE_1API)
  auto len3 = sycl::range<3>(ctx->z, ctx->y, ctx->x);
#endif

  /* prediction-quantization with compaction */
  {
    if (spline_in_use()) {
#ifdef PSZ_USE_CUDA
      mem->od->dptr(in);
      spline_construct(
          mem->od, mem->ac, mem->e, (void*)mem->compact, eb, radius,
          &time_pred, stream);
#else
      throw runtime_error(
          "[psz::error] spline_construct not implemented other than CUDA.");
#endif
    }
    else {
      psz_comp_l23r<T, E>(
          in, len3, eb, radius, mem->ectrl(), (void*)mem->compact, &time_pred,
          stream);
    }

    if (spline_in_use()) {
      PSZDBG_LOG("interp: done");
      PSZDBG_PTR_WHERE(mem->ectrl());
      PSZDBG_VAR("pipeline", len);
    }
    if (spline_in_use()) {
      PSZSANITIZE_QUANTCODE(mem->e->control({D2H})->hptr(), len, booklen);
    }
  }

  /* make outlier count seen on host */
  {
    mem->compact->make_host_accessible((GpuStreamT)stream);
    ctx->splen = mem->compact->num_outliers();
  }

  return this;
}

COR::compress_histogram(pszctx* ctx, void* stream)
{
  auto spline_in_use = [&]() { return ctx->pred_type == Spline; };
  auto booklen = ctx->radius * 2;

  /* statistics: histogram */
  {
    PSZ_HIST(mem->ectrl(), len, mem->hist(), booklen, &time_hist, stream);
    // if (spline_in_use())
    // {PSZSANITIZE_HIST_OUTPUT(mem->ht->control({D2H})->hptr(), booklen);}
    if (spline_in_use()) { PSZDBG_LOG("histsp gpu: done"); }
  }

  return this;
}

COR::compress_encode(pszctx* ctx, void* stream)
{
  auto spline_in_use = [&]() { return ctx->pred_type == Spline; };
  auto booklen = ctx->radius * 2;

  /* Huffman encoding */
  {
    codec->build_codebook(mem->ht, booklen, stream);
    // [TODO] CR estimation must be after building codebook; need a flag.
    if (ctx->report_cr_est) {
      auto overhead = spline_in_use() ? sizeof(T) * mem->ac->len() : 0;
      codec->calculate_CR(mem->e, sizeof(T), overhead);
    }
    if (spline_in_use()) { PSZDBG_LOG("codebook: done"); }
    // if (spline_in_use()){PSZSANITIZE_HIST_BK(mem->ht->hptr(),
    // codec->bk4->hptr(), booklen);}
    codec->encode(mem->ectrl(), len, &comp_hf_out, &comp_hf_outlen, stream);
    if (spline_in_use()) { PSZDBG_LOG("encoding done"); }
  }

  return this;
}

COR::compress_update_header(pszctx* ctx, void* stream)
{
  auto spline_in_use = [&]() { return ctx->pred_type == Spline; };

  header.x = ctx->x, header.y = ctx->y, header.z = ctx->z,
  header.w = 1;  // placeholder
  header.radius = ctx->radius, header.eb = ctx->eb;
  header.vle_pardeg = ctx->vle_pardeg;
  header.splen = ctx->splen;
  header.pred_type = ctx->pred_type;
  header.dtype = PszType<T>::type;

  // TODO no need to copy header to device
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  CHECK_GPU(GpuMemcpyAsync(
      mem->compressed() + 0, &header, sizeof(header), GpuMemcpyH2D,
      (GpuStreamT)stream));
#elif defined(PSZ_USE_1API)
  auto queue = (sycl::queue*)stream;
  queue->memcpy(mem->compressed() + 0, &header, sizeof(header));
#endif

  if (spline_in_use()) { PSZDBG_LOG("update header: done"); }

  return this;
}

COR::compress_wrapup(BYTE** out, szt* outlen)
{
  /* output of this function */
  *out = mem->_compressed->dptr();
  *outlen = psz_utils::filesize(&header);
  mem->_compressed->m->len = *outlen;
  mem->_compressed->m->bytes = *outlen;

  return this;
}

COR::compress(pszctx* ctx, T* in, BYTE** out, size_t* outlen, void* stream)
{
  auto spline_in_use = [&]() { return ctx->pred_type == Spline; };

  PSZSANITIZE_PSZCTX(ctx);

  compress_predict(ctx, in, stream);
  compress_histogram(ctx, stream);
  compress_encode(ctx, stream);
  compress_merge(ctx, stream);
  compress_update_header(ctx, stream);
  compress_wrapup(out, outlen);
  compress_collect_kerneltime();

  return this;
}

COR::compress_merge(pszctx* ctx, void* stream)
#if defined(PSZ_USE_1API)
try
#endif
{

#if defined(PSZ_USE_1API)
  auto queue = (sycl::queue*)stream;
#endif

  auto spline_in_use = [&]() { return ctx->pred_type == Spline; };

  auto splen = mem->compact->num_outliers();
  auto pred_type = ctx->pred_type;

  auto dst = [&](int FIELD, szt offset = 0) {
    return (void*)(mem->compressed() + header.entry[FIELD] + offset);
  };

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  auto concat_d2d = [&](int FIELD, void* src, u4 dst_offset = 0) {
    CHECK_GPU(GpuMemcpyAsync(
        dst(FIELD, dst_offset), src, nbyte[FIELD], GpuMemcpyD2D,
        (GpuStreamT)stream));
  };
#elif defined(PSZ_USE_1API)
  auto concat_d2d = [&](int FIELD, void* src, u4 dst_offset = 0) {
    queue->memcpy(dst(FIELD, dst_offset), src, nbyte[FIELD]);
  };
#endif

  ////////////////////////////////////////////////////////////////
  nbyte[Header::HEADER] = sizeof(Header);
  nbyte[Header::VLE] = sizeof(BYTE) * comp_hf_outlen;
  nbyte[Header::ANCHOR] = pred_type == Spline ? sizeof(T) * mem->ac->len() : 0;
  nbyte[Header::SPFMT] = (sizeof(T) + sizeof(M)) * splen;

  header.entry[0] = 0;
  // *.END + 1; need to know the ending position
  for (auto i = 1; i < Header::END + 1; i++) header.entry[i] = nbyte[i - 1];
  for (auto i = 1; i < Header::END + 1; i++)
    header.entry[i] += header.entry[i - 1];

  // copy anchor
  if (pred_type == Spline) concat_d2d(Header::ANCHOR, mem->anchor(), 0);
  concat_d2d(Header::VLE, comp_hf_out, 0);

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  CHECK_GPU(GpuMemcpyAsync(
      dst(Header::SPFMT, 0), mem->compact_val(), sizeof(T) * splen,
      GpuMemcpyD2D, (GpuStreamT)stream));
  CHECK_GPU(GpuMemcpyAsync(
      dst(Header::SPFMT, sizeof(T) * splen), mem->compact_idx(),
      sizeof(M) * splen, GpuMemcpyD2D, (GpuStreamT)stream));
  /* debug */ CHECK_GPU(GpuStreamSync(stream));
#elif defined(PSZ_USE_1API)
  queue->memcpy(
      dst(Header::SPFMT, 0),  //
      mem->compact_val(), sizeof(T) * splen);
  queue->memcpy(
      dst(Header::SPFMT, sizeof(T) * splen), mem->compact_idx(),
      sizeof(M) * splen);
  /* debug */ queue->wait();
#endif

  if (spline_in_use()) { PSZDBG_LOG("merge buf: done"); }

  return this;
}
#if defined(PSZ_USE_1API)
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
#endif

COR::dump(std::vector<pszmem_dump> list, char const* basename)
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
      mem->_compressed->control({D2H})->file(ofn(".psz_archive"), ToFile);
    else if (i == PszQuant)
      mem->e->control({D2H})->file(ofn(".psz_quant"), ToFile);
    else if (i == PszHist)
      mem->ht->control({D2H})->file(ofn(".psz_hist"), ToFile);
    else if (i > PszHf______ and i < END)
      codec->dump({i}, basename);
    else
      printf("[psz::dump] not a valid segment to dump.");
  }

  return this;
}

COR::clear_buffer()
{
  codec->clear_buffer();
  mem->clear_buffer();
  return this;
}

COR::decompress_predict(
    pszheader* header, BYTE* in, T* ext_anchor, T* out, uninit_stream_t stream)
{
  auto access = [&](int FIELD, szt offset_nbyte = 0) {
    return (void*)(in + header->entry[FIELD] + offset_nbyte);
  };

  const auto eb = header->eb;
  const auto radius = header->radius;

  if (in and ext_anchor)
    throw std::runtime_error(
        "[psz::error] One of external in and ext_anchor must be null.");

  auto d_anchor = ext_anchor ? ext_anchor : (T*)access(Header::ANCHOR);
  // wire and aliasing
  auto d_space = out;
  auto d_xdata = out;

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  auto len3 = dim3(header->x, header->y, header->z);
#elif defined(PSZ_USE_1API)
  auto len3 = sycl::range<3>(header->z, header->y, header->x);
#endif

  if (header->pred_type == Spline) {
#ifdef PSZ_USE_CUDA
    mem->xd->dptr(out);

    // TODO release borrow
    auto aclen3 = mem->ac->template len3<dim3>();
    pszmem_cxx<T> anchor(aclen3.x, aclen3.y, aclen3.z);
    anchor.dptr(d_anchor);

    // [psz::TODO] throw exception

    spline_reconstruct(
        &anchor, mem->e, mem->xd, eb, radius, &time_pred, stream);
#else
    throw runtime_error(
        "[psz::error] spline_reconstruct not implemented other than CUDA.");
#endif
  }
  else {
    psz_decomp_l23<T, E, FP>(
        mem->ectrl(), len3, d_space, eb, radius, d_xdata, &time_pred, stream);
  }

  return this;
}

COR::decompress_decode(pszheader* header, BYTE* in, uninit_stream_t stream)
{
  auto access = [&](int FIELD, szt offset_nbyte = 0) {
    return (void*)(in + header->entry[FIELD] + offset_nbyte);
  };

  codec->decode((B*)access(Header::VLE), mem->ectrl(), stream);
  return this;
}

COR::decompress_scatter(
    pszheader* header, BYTE* in, T* d_space, uninit_stream_t stream)
{
  auto access = [&](int FIELD, szt offset_nbyte = 0) {
    return (void*)(in + header->entry[FIELD] + offset_nbyte);
  };

  // The inputs of components are from `compressed`.
  auto d_anchor = (T*)access(Header::ANCHOR);
  auto d_spval = (T*)access(Header::SPFMT);
  auto d_spidx = (M*)access(Header::SPFMT, header->splen * sizeof(T));

  psz::spv_scatter_naive<PROPER_GPU_BACKEND, T, M>(
      d_spval, d_spidx, header->splen, d_space, &time_sp, stream);

  return this;
}

COR::decompress(pszheader* header, BYTE* in, T* out, void* stream)
{
  // TODO host having copy of header when compressing
  if (not header) {
    header = new Header;
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    CHECK_GPU(GpuMemcpyAsync(
        header, in, sizeof(Header), GpuMemcpyD2H, (GpuStreamT)stream));
    CHECK_GPU(GpuStreamSync(stream));
#elif defined(PSZ_USE_1API)
    ((sycl::queue*)stream)->memcpy(header, in, sizeof(Header));
    ((sycl::queue*)stream)->wait();
#endif
  }

  // wire and alias
  auto d_space = out, d_xdata = out;

  decompress_scatter(header, in, d_space, stream);
  decompress_decode(header, in, stream);
  decompress_predict(header, in, nullptr, d_xdata, stream);
  decompress_collect_kerneltime();

  return this;
}

// public getter
COR::export_header(pszheader& ext_header)
{
  ext_header = header;
  return this;
}

COR::export_header(pszheader* ext_header)
{
  *ext_header = header;
  return this;
}

COR::export_timerecord(TimeRecord* ext_timerecord)
{
  if (ext_timerecord) *ext_timerecord = timerecord;
  return this;
}

COR::compress_collect_kerneltime()
{
#define COLLECT_TIME(NAME, TIME) \
  timerecord.push_back({const_cast<const char*>(NAME), TIME});

  if (not timerecord.empty()) timerecord.clear();

  COLLECT_TIME("predict", time_pred);
  COLLECT_TIME("histogram", time_hist);
  COLLECT_TIME("book", codec->time_book());
  COLLECT_TIME("huff-enc", codec->time_lossless());
  // COLLECT_TIME("outlier", time_sp);

  return this;
}

COR::decompress_collect_kerneltime()
{
  if (not timerecord.empty()) timerecord.clear();

  COLLECT_TIME("outlier", time_sp);
  COLLECT_TIME("huff-dec", codec->time_lossless());
  COLLECT_TIME("predict", time_pred);

  return this;
}

}  // namespace cusz

#undef COLLECT_TIME
#undef COR

#endif /* A2519F0E_602B_4798_A8EF_9641123095D9 */
