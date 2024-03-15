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

#include "compressor.hh"
#include "exception/exception.hh"
#include "hf/hfclass.hh"
#include "kernel.hh"
#include "log.hh"
#include "mem.hh"
#include "port.hh"
#include "utils/config.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

using std::cerr;
using std::endl;

#define COR          \
  template <class C> \
  Compressor<C>* Compressor<C>

// [psz::note] pszcxx_histogram_generic is left for evaluating purpose
// compared to pszcxx_histogram_cauchy
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_1API)
#define PSZ_HIST(...) \
  pszcxx_histogram_cauchy<PROPER_GPU_BACKEND, E>(__VA_ARGS__);
#elif defined(PSZ_USE_HIP)
#define PSZ_HIST(...) \
  pszcxx_histogram_generic<PROPER_GPU_BACKEND, E>(__VA_ARGS__);
#endif

void concate_memcpy_d2d(
    void* dst, void* src, size_t nbyte, void* stream, const char* _file_,
    const int _line_)
{
  if (nbyte != 0) {
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    AD_HOC_CHECK_GPU_WITH_LINE(
        GpuMemcpyAsync(dst, src, nbyte, GpuMemcpyD2D, (GpuStreamT)stream),
        _file_, _line_);
#elif defined(PSZ_USE_1API)
    ((sycl::queue*)stream)->memcpy(dst(FIELD), src, nbyte);
#endif
  }
}

#define DST(FIELD, OFFSET) \
  ((void*)(mem->compressed() + header.entry[FIELD] + OFFSET))

#define SKIP_ON_FAILURE \
  if (not error_list.empty()) return this;

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
  const auto booklen = radius * 2;
  const auto x = ctx->x, y = ctx->y, z = ctx->z;
  len = x * y * z;

  // TODO [2403] need to merge the following
  mem = new pszmempool_cxx<T, E, H>(x, radius, y, z);

  // TODO [2403] ad hoc
  _2403_compact = new pszcompact_cxx<T>{
      mem->compact->d_val, mem->compact->d_idx, mem->compact->d_num,
      mem->compact->h_num, mem->compact->reserved_len};
  codec->init(mem->len, booklen, pardeg, debug);

  return this;
}

COR::compress_predict(pszctx* ctx, T* in, void* stream)
try {
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  auto len3 = dim3(ctx->x, ctx->y, ctx->z);
#elif defined(PSZ_USE_1API)
  auto len3 = sycl::range<3>(ctx->z, ctx->y, ctx->x);
#endif

  /* prediction-quantization with compaction */
  if (ctx->pred_type == Spline) {
#ifdef PSZ_USE_CUDA
    mem->od->dptr(in);
    pszcxx_predict_spline(
        mem->od, mem->ac, mem->e, (void*)mem->compact, ctx->eb, ctx->radius,
        &time_pred, stream);
#else
    throw runtime_error(
        "[psz::error] pszcxx_predict_spline not implemented other than "
        "CUDA.");
#endif
  }
  else {
    _2401::pszpred_lrz<T>::pszcxx_predict_lorenzo(
        {in, ctx->_2403_pszlen}, {ctx->eb, ctx->radius},
        {mem->e->dptr(), ctx->data_len}, *_2403_compact, &time_pred, stream);

    PSZDBG_LOG("interp: done");
    PSZDBG_PTR_WHERE(mem->ectrl());
    PSZDBG_VAR("pipeline", len);
    PSZSANITIZE_QUANTCODE(mem->e->control({D2H})->hptr(), len, booklen);
  }

  /* make outlier count seen on host */
  mem->compact->make_host_accessible((GpuStreamT)stream);
  ctx->splen = mem->compact->num_outliers();

  return this;
}
catch (const psz::exception_outlier_overflow& e) {
  cerr << e.what() << endl;
  cerr << "outlier numbers: " << mem->compact->num_outliers() << "\t";
  cerr << (mem->compact->num_outliers() * 100.0 / len) << "%" << endl;
  error_list.push_back(PSZ_ERROR_OUTLIER_OVERFLOW);
  return this;
}

COR::compress_encode(pszctx* ctx, void* stream)
{
  auto booklen = ctx->dict_size;

  /* Huffman encoding */
  codec->build_codebook(mem->ht, booklen, stream);
  // [TODO] CR estimation must be after building codebook; need a flag.
  if (ctx->report_cr_est) {
    codec->calculate_CR(
        mem->e, sizeof(T),
        (sizeof(T) * mem->ac->len()) * (ctx->pred_type == Spline));
  }
  PSZDBG_LOG("codebook: done");
  // PSZSANITIZE_HIST_BK(mem->ht->hptr(), codec->bk4->hptr(), booklen);

  codec->encode(mem->ectrl(), len, &comp_hf_out, &comp_hf_outlen, stream);

  PSZDBG_LOG("encoding done");

  return this;
}

COR::compress_encode_use_prebuilt(pszctx* ctx, void* stream)
{
  throw psz::exception_placeholder();

  return this;
}

COR::compress(pszctx* ctx, T* in, BYTE** out, size_t* outlen, void* stream)
{
  PSZSANITIZE_PSZCTX(ctx);

  compress_predict(ctx, in, stream);

  /* statistics: histogram */
  PSZ_HIST(mem->ectrl(), len, mem->hist(), ctx->dict_size, &time_hist, stream);
  PSZDBG_LOG("histsp gpu: done");

  // TODO constexpr if
  if (ctx->use_prebuilt_hfbk)
    compress_encode_use_prebuilt(ctx, stream);
  else
    compress_encode(ctx, stream);

  compress_merge_update_header(ctx, out, outlen, stream);
  compress_collect_kerneltime();

  // TODO constexpr if and runtime if
  if (ctx->dump_quantcode) optional_dump(ctx, pszmem_dump::PszQuant);
  if (ctx->dump_hist) optional_dump(ctx, pszmem_dump::PszHist);
  // TODO export or view
  if (not error_list.empty()) ctx->there_is_memerr = true;

  return this;
}

COR::compress_merge_update_header(
    pszctx* ctx, BYTE** out, szt* outlen, void* stream)
try {
  // merge
  // SKIP_ON_FAILURE;

  auto splen = mem->compact->num_outliers();
  auto pred_type = ctx->pred_type;

  nbyte[Header::HEADER] = sizeof(Header);
  nbyte[Header::VLE] = sizeof(BYTE) * comp_hf_outlen;
  nbyte[Header::ANCHOR] = pred_type == Spline ? sizeof(T) * mem->ac->len() : 0;
  nbyte[Header::SPFMT] = (sizeof(T) + sizeof(M)) * splen;

  // clang-format off
  header.entry[0] = 0;
  // *.END + 1; need to know the ending position
  for (auto i = 1; i < Header::END + 1; i++) header.entry[i] = nbyte[i - 1];
  for (auto i = 1; i < Header::END + 1; i++) header.entry[i] += header.entry[i - 1];

  concate_memcpy_d2d(DST(Header::ANCHOR, 0), mem->anchor(), nbyte[Header::ANCHOR], stream, __FILE__, __LINE__);
  concate_memcpy_d2d(DST(Header::VLE, 0), comp_hf_out, nbyte[Header::VLE], stream, __FILE__, __LINE__);
  concate_memcpy_d2d(DST(Header::SPFMT, 0), mem->compact_val(), sizeof(T) * splen, stream, __FILE__, __LINE__);
  concate_memcpy_d2d(DST(Header::SPFMT, sizeof(T) * splen), mem->compact_idx(), sizeof(M) * splen, stream, __FILE__, __LINE__);
  // clang-format on

  PSZDBG_LOG("merge buf: done");

  // update header

  header.x = ctx->x, header.y = ctx->y, header.z = ctx->z,
  header.w = 1;  // placeholder
  header.radius = ctx->radius, header.eb = ctx->eb;
  header.vle_pardeg = ctx->vle_pardeg;
  header.splen = ctx->splen;
  header.pred_type = ctx->pred_type;
  header.dtype = PszType<T>::type;

  PSZDBG_LOG("update header: done");

  // wrap up
  // SKIP_ON_FAILURE;

  /* output of this function */
  *out = mem->_compressed->dptr();
  *outlen = psz_utils::filesize(&header);
  mem->_compressed->m->len = *outlen;
  mem->_compressed->m->bytes = *outlen;

  return this;
}
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
catch (const psz ::exception_gpu_general& e) {
  std ::cerr << e.what() << std ::endl;
  error_list.push_back(PSZ_ERROR_GPU_GENERAL);
  return this;
}
#endif
#if defined(PSZ_USE_1API)
catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  return this;
}
#endif

COR::optional_dump(pszctx* ctx, pszmem_dump const i)
{
  char __[256];

  auto ofn = [&](char const* suffix) {
    return string(ctx->file_input) + string(suffix);
  };

  if (i == PszQuant) {
    auto ofname = ofn(".psz_quant");
    mem->e->control({D2H})->file(ofname.c_str(), ToFile);
    cout << "dumping quantization code: " << ofname << endl;
  }
  else if (i == PszHist) {
    auto ofname = ofn(".psz_hist");
    mem->ht->control({D2H})->file(ofname.c_str(), ToFile);
    cout << "dumping histogram: " << ofname << endl;
  }
  // else if (i > PszHf______ and i < END)
  //   codec->dump({i}, basename);
  else
    printf("[psz::dump] not a valid segment to dump.");

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

  // const auto eb = header->eb;
  // const auto radius = header->radius;

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

  auto _adhoc_pszlen = pszlen{header->x, header->y, header->z, 1};
  auto _adhoc_linear = header->z * header->y * header->x;

  if (header->pred_type == Spline) {
#ifdef PSZ_USE_CUDA
    mem->xd->dptr(out);

    // TODO release borrow
    auto aclen3 = mem->ac->template len3<dim3>();
    pszmem_cxx<T> anchor(aclen3.x, aclen3.y, aclen3.z);
    anchor.dptr(d_anchor);

    // [psz::TODO] throw exception

    pszcxx_reverse_predict_spline(
        &anchor, mem->e, mem->xd, header->eb, header->radius, &time_pred,
        stream);
#else
    throw runtime_error(
        "[psz::error] pszcxx_reverse_predict_spline not implemented other "
        "than CUDA.");
#endif
  }
  else {
    _2401::pszpred_lrz<T>::pszcxx_reverse_predict_lorenzo(
        {mem->ectrl(), _adhoc_linear}, {d_space, _adhoc_linear},
        {header->eb, (int)header->radius}, {d_xdata, _adhoc_pszlen},
        &time_pred, stream);

    // pszcxx_reverse_predict_lorenzo<T, E, FP>(
    //     mem->ectrl(), len3, d_space, header->eb, header->radius, d_xdata,
    //     &time_pred, stream);
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

  if (header->splen != 0)
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
