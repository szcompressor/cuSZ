/**
 * @file compressor.inl
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
#include <type_traits>

#include "compressor.hh"
#include "exception/exception.hh"
#include "hfclass.hh"
#include "kernel.hh"
#include "log.hh"
#include "module/cxx_module.hh"
#include "port.hh"
#include "utils/config.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

#define COLLECT_TIME(NAME, TIME) \
  timerecord.push_back({const_cast<const char*>(NAME), TIME});

// [psz::note] pszcxx_histogram_generic is left for evaluating purpose
// compared to pszcxx_histogram_cauchy
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_1API)
#define PSZ_HIST(...) \
  pszcxx_compat_histogram_cauchy<PROPER_GPU_BACKEND, E>(__VA_ARGS__);
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
        cudaMemcpyAsync(
            dst, src, nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream),
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

namespace psz {

template <class C>
struct Compressor<C>::impl {
  using Codec = typename C::Codec;
  using BYTE = uint8_t;
  using B = u1;

  using T = typename C::T;
  using E = typename C::E;
  using FP = typename C::FP;
  using M = typename C::M;
  using Header = psz_header;

  using H = u4;
  using H4 = u4;
  using H8 = u8;
  using TimeRecord = std::vector<std::tuple<const char*, double>>;
  using timerecord_t = TimeRecord*;

  static const int FORCED_ALIGN = 128;
  static const int HEADER = 0;
  static const int ANCHOR = 1;
  static const int VLE = 2;
  static const int SPFMT = 3;
  static const int END = 4;

  // encapsulations
  Codec* codec;  // has standalone internals
  pszmempool_cxx<T, E, H>* mem;
  std::vector<pszerror> error_list;
  TimeRecord timerecord;

  float timerecord_v2[STAGE_END];

  // variables
  Header header;
  size_t len;
  BYTE* comp_hf_out{nullptr};
  size_t comp_hf_outlen{0};

  // arrays
  uint32_t nbyte[END];

  float time_pred, time_hist, time_sp;

  ~impl() { delete mem, delete codec; }

  template <class CONFIG>
  void init(CONFIG* ctx, bool iscompression, bool debug)
  {
    const auto radius = ctx->radius;
    const auto pardeg = ctx->vle_pardeg;
    const auto bklen = radius * 2;
    const auto x = ctx->x, y = ctx->y, z = ctx->z;
    len = x * y * z;

    mem = new pszmempool_cxx<T, E, H>(x, radius, y, z, iscompression);
    codec = new Codec(mem->len, bklen, pardeg, debug);
  }

  void compress_predict(pszctx* ctx, T* in, void* stream)
  try {
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    auto len3 = dim3(ctx->x, ctx->y, ctx->z);
#elif defined(PSZ_USE_1API)
    auto len3 = sycl::range<3>(ctx->z, ctx->y, ctx->x);
#endif

    /* prediction-quantization with compaction */
    if (ctx->pred_type == Spline) {
#ifdef PSZ_USE_CUDA

      memobj<T> local_spline_in(
          ctx->x, ctx->y, ctx->z, "local pszmem for spline input");
      local_spline_in.dptr(in);

      pszcxx_predict_spline(
          &local_spline_in, mem->_anchor, mem->_ectrl, (void*)mem->compact,
          ctx->eb, ctx->radius, &time_pred, stream);
#else
      throw runtime_error(
          "[psz::error] pszcxx_predict_spline not implemented other than "
          "CUDA.");
#endif
    }
    else {
      if (ctx->use_proto_lorenzo)
        pszcxx_predict_lorenzo<T, E, SYNC_BY_STREAM, true>(
            {in, get_len3(ctx)}, {ctx->eb, ctx->radius},
            {mem->_ectrl->dptr(), ctx->data_len}, mem->outlier(), &time_pred,
            stream);
      else
        pszcxx_predict_lorenzo<T, E, SYNC_BY_STREAM, false>(
            {in, get_len3(ctx)}, {ctx->eb, ctx->radius},
            {mem->_ectrl->dptr(), ctx->data_len}, mem->outlier(), &time_pred,
            stream);

      PSZDBG_LOG("interp: done");
      PSZDBG_PTR_WHERE(mem->ectrl());
      PSZDBG_VAR("pipeline", len);
      PSZSANITIZE_QUANTCODE(mem->_ectrl->control({D2H})->hptr(), len, booklen);
    }

    /* make outlier count seen on host */
    mem->compact->make_host_accessible((cudaStream_t)stream);
    ctx->splen = mem->compact->num_outliers();
    // return this;
  }
  catch (const psz::exception_outlier_overflow& e) {
    cerr << e.what() << endl;
    cerr << "outlier numbers: " << mem->compact->num_outliers() << "\t";
    cerr << (mem->compact->num_outliers() * 100.0 / len) << "%" << endl;
    error_list.push_back(PSZ_ERROR_OUTLIER_OVERFLOW);
    // return this;
  }

  void compress_encode(pszctx* ctx, void* stream)
  {
    codec->buildbook(mem->_hist->dptr(), stream);
    // [TODO] CR estimation must be after building codebook; need a flag.
    if (ctx->report_cr_est) {
      codec->calculate_CR(
          mem->_ectrl, mem->_hist, sizeof(T),
          (sizeof(T) * mem->_anchor->len()) * (ctx->pred_type == Spline));
    }
    PSZDBG_LOG("codebook: done");
    // PSZSANITIZE_HIST_BK(mem->_hist->hptr(), codec->bk4->hptr(), booklen);

    codec->encode(mem->ectrl(), len, &comp_hf_out, &comp_hf_outlen, stream);

    PSZDBG_LOG("encoding done");
  }

  void compress_encode_use_prebuilt(pszctx* ctx, void* stream)
  {
    throw psz::exception_placeholder();
  }

  void compress_merge_update_header(
      pszctx* ctx, BYTE** out, szt* outlen, void* stream)
  try {
    // SKIP_ON_FAILURE;

    auto splen = mem->compact->num_outliers();
    auto pred_type = ctx->pred_type;

    nbyte[HEADER] = sizeof(header);
    nbyte[VLE] = sizeof(BYTE) * comp_hf_outlen;
    nbyte[ANCHOR] = pred_type == Spline ? sizeof(T) * mem->_anchor->len() : 0;
    nbyte[SPFMT] = (sizeof(T) + sizeof(M)) * splen;

    // clang-format off
    header.entry[0] = 0;
    // *.END + 1; need to know the ending position
    for (auto i = 1; i < END + 1; i++) header.entry[i] = nbyte[i - 1];
    for (auto i = 1; i < END + 1; i++) header.entry[i] += header.entry[i - 1];

    concate_memcpy_d2d(DST(ANCHOR, 0), mem->anchor(), nbyte[ANCHOR], stream, __FILE__, __LINE__);
    concate_memcpy_d2d(DST(VLE, 0), comp_hf_out, nbyte[VLE], stream, __FILE__, __LINE__);
    concate_memcpy_d2d(DST(SPFMT, 0), mem->compact_val(), sizeof(T) * splen, stream, __FILE__, __LINE__);
    concate_memcpy_d2d(DST(SPFMT, sizeof(T) * splen), mem->compact_idx(), sizeof(M) * splen, stream, __FILE__, __LINE__);
    // clang-format on

    PSZDBG_LOG("merge buf: done");

    // update header
    header.x = ctx->x, header.y = ctx->y, header.z = ctx->z,
    header.w = 1;  // placeholder
    header.use_proto_lorenzo = ctx->use_proto_lorenzo;
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
    *outlen = pszheader_filesize(&header);
    mem->_compressed->set_len(*outlen);

    // return this;
  }
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  catch (const psz ::exception_gpu_general& e) {
    std ::cerr << e.what() << std ::endl;
    error_list.push_back(PSZ_ERROR_GPU_GENERAL);
    // return this;
  }
#endif
#if defined(PSZ_USE_1API)
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    // return this;
  }
#endif

  void decompress_predict(
      psz_header* header, BYTE* in, T* ext_anchor, T* out, psz_stream_t stream)
  {
    auto access = [&](int FIELD, szt offset_nbyte = 0) {
      return (void*)(in + header->entry[FIELD] + offset_nbyte);
    };

    if (in and ext_anchor)
      throw std::runtime_error(
          "[psz::error] One of external in and ext_anchor must be null.");

    auto d_anchor = ext_anchor ? ext_anchor : (T*)access(ANCHOR);
    // wire and aliasing
    auto d_space = out;
    auto d_xdata = out;

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    auto len3 = dim3(header->x, header->y, header->z);
#elif defined(PSZ_USE_1API)
    auto len3 = sycl::range<3>(header->z, header->y, header->x);
#endif

    auto _adhoc_pszlen = psz_len3{header->x, header->y, header->z};
    auto _adhoc_linear = header->z * header->y * header->x;

    if (header->pred_type == Spline) {
#ifdef PSZ_USE_CUDA
      auto __xdata =
          new memobj<T>(header->x, header->y, header->z, "__xdata", {});
      __xdata->dptr(out);

      // TODO release borrow
      auto aclen3 = mem->_anchor->len3();
      memobj<T> anchor(aclen3.x, aclen3.y, aclen3.z);
      anchor.dptr(d_anchor);

      // [psz::TODO] throw exception

      pszcxx_reverse_predict_spline(
          &anchor, mem->_ectrl, __xdata, header->eb, header->radius,
          &time_pred, stream);
#else
      throw runtime_error(
          "[psz::error] pszcxx_reverse_predict_spline not implemented other "
          "than CUDA.");
#endif
    }
    else {
      if (header->use_proto_lorenzo)
        pszcxx_reverse_predict_lorenzo<T, E, SYNC_BY_STREAM, true>(
            {mem->ectrl(), _adhoc_linear}, {d_space, _adhoc_linear},
            {header->eb, (int)header->radius}, {d_xdata, _adhoc_pszlen},
            &time_pred, stream);
      else
        pszcxx_reverse_predict_lorenzo<T, E, SYNC_BY_STREAM, false>(
            {mem->ectrl(), _adhoc_linear}, {d_space, _adhoc_linear},
            {header->eb, (int)header->radius}, {d_xdata, _adhoc_pszlen},
            &time_pred, stream);
    }
  }

  void decompress_decode(psz_header* header, BYTE* in, psz_stream_t stream)
  {
    auto access = [&](int FIELD, szt offset_nbyte = 0) {
      return (void*)(in + header->entry[FIELD] + offset_nbyte);
    };
    codec->decode((B*)access(VLE), mem->ectrl(), stream);
  }

  void decompress_scatter(
      psz_header* header, BYTE* in, T* d_space, psz_stream_t stream)
  {
    auto access = [&](int FIELD, szt offset_nbyte = 0) {
      return (void*)(in + header->entry[FIELD] + offset_nbyte);
    };

    // The inputs of components are from `compressed`.
    auto d_anchor = (T*)access(ANCHOR);
    auto d_spval = (T*)access(SPFMT);
    auto d_spidx = (M*)access(SPFMT, header->splen * sizeof(T));

    if (header->splen != 0)
      psz::spv_scatter_naive<PROPER_GPU_BACKEND, T, M>(
          d_spval, d_spidx, header->splen, d_space, &time_sp, stream);
    // return this;
  }

  void clear_buffer() { codec->clear_buffer(), mem->clear_buffer(); }

  void compress_collect_kerneltime()
  {
    if (not timerecord.empty()) timerecord.clear();

    COLLECT_TIME("predict", time_pred);
    COLLECT_TIME("histogram", time_hist);
    COLLECT_TIME("book", codec->time_book());
    COLLECT_TIME("huff-enc", codec->time_lossless());

    timerecord_v2[STAGE_PREDICT] = time_pred;
    timerecord_v2[STAGE_HISTOGRM] = time_hist;
    timerecord_v2[STAGE_BOOK] = codec->time_book();
    timerecord_v2[STAGE_HUFFMAN] = codec->time_lossless();
  }

  void decompress_collect_kerneltime()
  {
    if (not timerecord.empty()) timerecord.clear();

    COLLECT_TIME("outlier", time_sp);
    COLLECT_TIME("huff-dec", codec->time_lossless());
    COLLECT_TIME("predict", time_pred);

    timerecord_v2[STAGE_OUTLIER] = time_sp;
    timerecord_v2[STAGE_HUFFMAN] = codec->time_lossless();
    timerecord_v2[STAGE_PREDICT] = time_pred;
  }
};

//------------------------------------------------------------------------------

template <class C>
template <class CONFIG>
Compressor<C>* Compressor<C>::init(CONFIG* ctx, bool iscompression, bool debug)
{
  pimpl->template init<CONFIG>(ctx, iscompression, debug);
  return this;
}

template <class C>
Compressor<C>::Compressor() : pimpl{std::make_unique<impl>()} {};

template <class C>
Compressor<C>::Compressor(psz_context* ctx, bool debug) :
    pimpl{std::make_unique<impl>()}
{
  pimpl->template init(ctx, true /* comp */, debug);
}

template <class C>
Compressor<C>::Compressor(psz_header* header, bool debug) :
    pimpl{std::make_unique<impl>()}
{
  pimpl->template init(header, false /* decomp */, debug);
}

template <class C>
Compressor<C>::~Compressor(){};

template <class C>
Compressor<C>* Compressor<C>::compress(
    pszctx* ctx, T* in, BYTE** out, size_t* outlen, void* stream)
{
  PSZSANITIZE_PSZCTX(ctx);

  pimpl->compress_predict(ctx, in, stream);

  /* statistics: histogram */
  PSZ_HIST(
      pimpl->mem->ectrl(), pimpl->len, pimpl->mem->hist(), ctx->dict_size,
      &pimpl->time_hist, stream);
  PSZDBG_LOG("histsp gpu: done");

  // TODO constexpr if
  if (ctx->use_prebuilt_hfbk)
    pimpl->compress_encode_use_prebuilt(ctx, stream);
  else
    pimpl->compress_encode(ctx, stream);

  pimpl->compress_merge_update_header(ctx, out, outlen, stream);
  pimpl->compress_collect_kerneltime();

  // TODO export or view
  if (not pimpl->error_list.empty()) ctx->there_is_memerr = true;

  return this;
}

template <class C>
Compressor<C>* Compressor<C>::clear_buffer()
{
  pimpl->clear_buffer();
  return this;
}

template <class C>
Compressor<C>* Compressor<C>::decompress(
    psz_header* header, BYTE* in, T* out, void* stream)
{
  // TODO host having copy of header when compressing
  if (not header) {
    header = new psz_header;
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    CHECK_GPU(cudaMemcpyAsync(
        header, in, sizeof(psz_header), cudaMemcpyDeviceToHost,
        (cudaStream_t)stream));
    CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
#elif defined(PSZ_USE_1API)
    ((sycl::queue*)stream)->memcpy(header, in, sizeof(Header));
    ((sycl::queue*)stream)->wait();
#endif
  }

  // wire and alias
  auto d_space = out, d_xdata = out;

  pimpl->decompress_scatter(header, in, d_space, stream);
  pimpl->decompress_decode(header, in, stream);
  pimpl->decompress_predict(header, in, nullptr, d_xdata, stream);
  pimpl->decompress_collect_kerneltime();

  return this;
}

// public getter
template <class C>
Compressor<C>* Compressor<C>::dump_compress_intermediate(
    pszctx* ctx, psz_stream_t stream)
{
  auto dump_name = [&](string t, string suffix = ".quant") -> string {
    return string(ctx->file_input) + "." + string(ctx->char_meta_eb) + suffix +
           "_" + t;
  };

  if (ctx->dump_hist) {
    // TODO to be portable
    cudaStreamSynchronize((cudaStream_t)stream);
    auto& d = pimpl->mem->_hist;
    // TODO caution! lift hardcoded dtype (hist)
    d->control({D2H})->file(dump_name("u4", ".hist").c_str(), ToFile);
  }
  if (ctx->dump_quantcode) {
    cudaStreamSynchronize((cudaStream_t)stream);
    auto& d = pimpl->mem->_ectrl;
    // TODO caution! list hardcoded dtype (quant)
    d->control({D2H})->file(dump_name("u2", ".quant").c_str(), ToFile);
  }

  return this;
}

// public getter
template <class C>
Compressor<C>* Compressor<C>::export_header(psz_header& ext_header)
{
  ext_header = pimpl->header;
  return this;
}

template <class C>
Compressor<C>* Compressor<C>::export_header(psz_header* ext_header)
{
  if (ext_header) *ext_header = pimpl->header;
  return this;
}

template <class C>
Compressor<C>* Compressor<C>::export_timerecord(TimeRecord* ext_timerecord)
{
  if (ext_timerecord) *ext_timerecord = pimpl->timerecord;
  return this;
}

template <class C>
Compressor<C>* Compressor<C>::export_timerecord(float* ext_timerecord)
{
  if (ext_timerecord)
    for (auto i = 0; i < STAGE_END - 1; i++)
      ext_timerecord[i] = pimpl->timerecord_v2[i];
  return this;
}

}  // namespace psz

#undef COLLECT_TIME

#endif /* A2519F0E_602B_4798_A8EF_9641123095D9 */
