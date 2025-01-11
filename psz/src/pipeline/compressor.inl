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
#include "module/cxx_module.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

#define COLLECT_TIME(NAME, TIME) timerecord.push_back({const_cast<const char*>(NAME), TIME});

void concate_memcpy_d2d(
    void* dst, void* src, size_t nbyte, void* stream, const char* _file_, const int _line_)
{
  if (nbyte != 0) {
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    AD_HOC_CHECK_GPU_WITH_LINE(
        cudaMemcpyAsync(dst, src, nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream), _file_,
        _line_);
#elif defined(PSZ_USE_1API)
    ((sycl::queue*)stream)->memcpy(dst(FIELD), src, nbyte);
#endif
  }
}

#define DST(FIELD, OFFSET) ((void*)(mem->compressed() + header.entry[FIELD] + OFFSET))

#define SKIP_ON_FAILURE \
  if (not error_list.empty()) return this;

namespace psz {

template <class C>
struct Compressor<C>::impl {
  using CodecHF = typename C::CodecHF;
  using CodecFZG = typename C::CodecFZG;

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
  static const int ENCODED = 2;
  static const int SPFMT = 3;
  static const int END = 4;

  // encapsulations
  CodecHF* codec_hf;    // Codecs have standalone internals.
  CodecFZG* codec_fzg;  //
  pszmempool_cxx<T, E, H>* mem;
  std::vector<pszerror> error_list;
  TimeRecord timerecord;

  float timerecord_v2[STAGE_END];

  // variables
  Header header;
  size_t len;
  BYTE* comp_codec_out{nullptr};
  size_t comp_codec_outlen{0};

  // arrays
  uint32_t nbyte[END];

  float time_pred, time_hist, time_sp;

  ~impl()
  {
    delete mem;
    if (codec_hf) delete codec_hf;
    if (codec_fzg) delete codec_fzg;
  }

  template <class CONFIG>
  void init(CONFIG* ctx, bool iscompression, bool debug)
  {
    const auto radius = ctx->radius;
    const auto pardeg = ctx->vle_pardeg;
    const auto bklen = radius * 2;
    const auto x = ctx->x, y = ctx->y, z = ctx->z;
    len = x * y * z;

    mem = new pszmempool_cxx<T, E, H>(x, radius, y, z, iscompression);

    if (ctx->codec1_type == Huffman)
      codec_hf = new CodecHF(mem->len, bklen, pardeg, debug);
    else if (ctx->codec1_type == FZGPUCodec)
      codec_fzg = new CodecFZG(mem->len);
    else {
      throw std::runtime_error("[psz] codec other than Huffman or FZGPUCodec is not supported.");
    }
  }

  void compress_predict(pszctx* ctx, T* in, void* stream)
  try {
    auto len3 = MAKE_GPU_LEN3(ctx->x, ctx->y, ctx->z);
    auto stride3 = MAKE_GPU_LEN3(1, ctx->x, ctx->x * ctx->y);
    auto len3_std = MAKE_STD_LEN3(ctx->x, ctx->y, ctx->z);

    if (ctx->pred_type == Lorenzo)
      psz::module::GPU_c_lorenzo_nd_with_outlier<T, false, E>(
          in, len3_std, mem->ectrl(), (void*)mem->outlier(), ctx->eb, ctx->radius, &time_pred,
          stream);
    else if (ctx->pred_type == LorenzoZigZag)
      psz::module::GPU_c_lorenzo_nd_with_outlier<T, true, E>(
          in, len3_std, mem->ectrl(), (void*)mem->outlier(), ctx->eb, ctx->radius, &time_pred,
          stream);
    else if (ctx->pred_type == LorenzoProto)
      psz::module::GPU_PROTO_c_lorenzo_nd_with_outlier<T, E>(
          in, len3_std, mem->ectrl(), (void*)mem->outlier(), ctx->eb, ctx->radius, &time_pred,
          stream);
    else if (ctx->pred_type == Spline)
      psz::cuhip::GPU_predict_spline(
          in, len3, stride3, mem->ectrl(), mem->_ectrl->len3(), mem->_ectrl->stride3(),
          mem->anchor(), mem->_anchor->len3(), mem->_anchor->stride3(), (void*)mem->compact,
          ctx->eb, ctx->radius, &time_pred, stream);

    /* make outlier count seen on host */
    mem->compact->make_host_accessible((cudaStream_t)stream);
    ctx->splen = mem->compact->num_outliers();
  }
  catch (const psz::exception_outlier_overflow& e) {
    cerr << e.what() << endl;
    cerr << "outlier numbers: " << mem->compact->num_outliers() << "\t";
    cerr << (mem->compact->num_outliers() * 100.0 / len) << "%" << endl;
    error_list.push_back(PSZ_ERROR_OUTLIER_OVERFLOW);
  }

  void compress_histogram(pszctx* ctx, void* stream)
  {
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_1API)

    if (ctx->hist_type == psz_histogramtype::HistogramSparse)
      psz::module::GPU_histogram_Cauchy<E>(
          mem->ectrl(), len, mem->hist(), ctx->dict_size, &time_hist, stream);
    else if (ctx->hist_type == psz_histogramtype::HistogramGeneric)
      psz::module::GPU_histogram_generic<E>(
          mem->ectrl(), len, mem->hist(), ctx->dict_size, &time_hist, stream);

#elif defined(PSZ_USE_HIP)
    // not implemented
#endif
  }

  void compress_encode(pszctx* ctx, void* stream)
  {
    if (ctx->codec1_type == Huffman) {
      codec_hf->buildbook(mem->_hist->dptr(), stream);
      codec_hf->encode(mem->ectrl(), len, &comp_codec_out, &comp_codec_outlen, stream);
    }
    else if (ctx->codec1_type == FZGPUCodec) {
      codec_fzg->encode(mem->ectrl(), len, &comp_codec_out, &comp_codec_outlen, stream);
    }
  }

  void compress_encode_use_prebuilt(pszctx* ctx, void* stream)
  {
    throw psz::exception_placeholder();
  }

  void compress_merge_update_header(pszctx* ctx, BYTE** out, szt* outlen, void* stream)
  try {
    auto splen = mem->compact->num_outliers();
    auto pred_type = ctx->pred_type;

    nbyte[HEADER] = sizeof(header);
    nbyte[ENCODED] = sizeof(BYTE) * comp_codec_outlen;
    nbyte[ANCHOR] = pred_type == Spline ? sizeof(T) * mem->_anchor->len() : 0;
    nbyte[SPFMT] = (sizeof(T) + sizeof(M)) * splen;

    // clang-format off
    header.entry[0] = 0;
    // *.END + 1; need to know the ending position
    for (auto i = 1; i < END + 1; i++) header.entry[i] = nbyte[i - 1];
    for (auto i = 1; i < END + 1; i++) header.entry[i] += header.entry[i - 1];

    concate_memcpy_d2d(DST(ANCHOR, 0), mem->anchor(), nbyte[ANCHOR], stream, __FILE__, __LINE__);
    concate_memcpy_d2d(DST(ENCODED, 0), comp_codec_out, nbyte[ENCODED], stream, __FILE__, __LINE__);
    concate_memcpy_d2d(DST(SPFMT, 0), mem->compact_val(), sizeof(T) * splen, stream, __FILE__, __LINE__);
    concate_memcpy_d2d(DST(SPFMT, sizeof(T) * splen), mem->compact_idx(), sizeof(M) * splen, stream, __FILE__, __LINE__);
    // clang-format on

    // update header::metadata (.w is placeheld.)
    header.x = ctx->x, header.y = ctx->y, header.z = ctx->z, header.w = 1;
    header.splen = ctx->splen;
    // update header::comp_config
    header.radius = ctx->radius, header.eb = ctx->eb;
    header.vle_pardeg = ctx->vle_pardeg;
    // update header::atrributes
    header.dtype = PszType<T>::type;
    header.pred_type = ctx->pred_type;
    header.hist_type = ctx->hist_type;
    header.codec1_type = ctx->codec1_type;

    /* output of this function */
    *out = mem->compressed();
    *outlen = pszheader_filesize(&header);
  }
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  catch (const psz ::exception_gpu_general& e) {
    std ::cerr << e.what() << std ::endl;
    error_list.push_back(PSZ_ERROR_GPU_GENERAL);
  }
#endif
#if defined(PSZ_USE_1API)
  catch (sycl::exception const& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__
              << std::endl;
  }
#endif

  void decompress_predict(psz_header* header, BYTE* in, T* ext_anchor, T* out, psz_stream_t stream)
  {
    auto access = [&](int FIELD, szt offset_nbyte = 0) {
      return (void*)(in + header->entry[FIELD] + offset_nbyte);
    };

    if (in and ext_anchor)
      throw std::runtime_error("[psz::error] One of external in and ext_anchor must be null.");

    auto d_anchor = ext_anchor ? ext_anchor : (T*)access(ANCHOR);
    auto d_space = out, d_xdata = out;  // aliases

    auto len3 = MAKE_GPU_LEN3(header->x, header->y, header->z);
    auto stride3 = MAKE_GPU_LEN3(1, header->x, header->x * header->y);
    auto len3_std = MAKE_STD_LEN3(header->x, header->y, header->z);

    if (header->pred_type == Lorenzo)
      psz::module::GPU_x_lorenzo_nd<T, false, E>(
          mem->ectrl(), d_space, d_xdata, len3_std, header->eb, header->radius, &time_pred,
          stream);
    else if (header->pred_type == LorenzoZigZag)
      psz::module::GPU_x_lorenzo_nd<T, true, E>(
          mem->ectrl(), d_space, d_xdata, len3_std, header->eb, header->radius, &time_pred,
          stream);
    else if (header->pred_type == LorenzoProto)
      psz::module::GPU_PROTO_x_lorenzo_nd<T, E>(
          mem->ectrl(), d_space, d_xdata, len3_std, header->eb, header->radius, &time_pred,
          stream);
    else if (header->pred_type == Spline)
      psz::cuhip::GPU_reverse_predict_spline(
          mem->ectrl(), mem->_ectrl->len3(), mem->_ectrl->stride3(),  //
          d_anchor, mem->_anchor->len3(), mem->_anchor->stride3(),    //
          d_xdata, len3, stride3,                                     //
          header->eb, header->radius, &time_pred, stream);
  }

  void decompress_decode(psz_header* header, BYTE* in, psz_stream_t stream)
  {
    auto access = [&](int FIELD, szt offset_nbyte = 0) {
      return (void*)(in + header->entry[FIELD] + offset_nbyte);
    };
    if (header->codec1_type == Huffman) {
      codec_hf->decode((B*)access(ENCODED), mem->ectrl(), stream);
    }
    else if (header->codec1_type == FZGPUCodec) {
      codec_fzg->decode(
          (B*)access(ENCODED), pszheader_filesize(header), mem->ectrl(), mem->len, stream);
    }
    else {
      throw std::runtime_error("[psz] codec other than Huffman or FZGPUCodec is not supported.");
    }
  }

  void decompress_scatter(psz_header* header, BYTE* in, T* d_space, psz_stream_t stream)
  {
    auto access = [&](int FIELD, szt offset_nbyte = 0) {
      return (void*)(in + header->entry[FIELD] + offset_nbyte);
    };

    // The inputs of components are from `compressed`.
    auto d_anchor = (T*)access(ANCHOR);
    auto d_spval = (T*)access(SPFMT);
    auto d_spidx = (M*)access(SPFMT, header->splen * sizeof(T));

    if (header->splen != 0)
      psz::spv_scatter_naive<PROPER_RUNTIME, T, M>(
          d_spval, d_spidx, header->splen, d_space, &time_sp, stream);
    // return this;
  }

  void clear_buffer()
  {
    if (codec_hf) codec_hf->clear_buffer(), mem->clear_buffer();
  }

  void compress_collect_kerneltime()
  {
    if (not timerecord.empty()) timerecord.clear();

    COLLECT_TIME("predict", time_pred);

    if (codec_hf) {
      COLLECT_TIME("histogram", time_hist);
      COLLECT_TIME("book", codec_hf->time_book());
      COLLECT_TIME("huff-enc", codec_hf->time_lossless());
    }

    timerecord_v2[STAGE_PREDICT] = time_pred;

    if (codec_hf) {
      timerecord_v2[STAGE_HISTOGRM] = time_hist;
      timerecord_v2[STAGE_BOOK] = codec_hf->time_book();
      timerecord_v2[STAGE_HUFFMAN] = codec_hf->time_lossless();
    }
  }

  void decompress_collect_kerneltime()
  {
    if (not timerecord.empty()) timerecord.clear();

    COLLECT_TIME("outlier", time_sp);

    if (codec_hf) { COLLECT_TIME("huff-dec", codec_hf->time_lossless()); }
    COLLECT_TIME("predict", time_pred);

    timerecord_v2[STAGE_OUTLIER] = time_sp;
    if (codec_hf) { timerecord_v2[STAGE_HUFFMAN] = codec_hf->time_lossless(); }
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
Compressor<C>::Compressor(psz_context* ctx, bool debug) : pimpl{std::make_unique<impl>()}
{
  pimpl->template init<psz_context>(ctx, true /* comp */, debug);
}

template <class C>
Compressor<C>::Compressor(psz_header* header, bool debug) : pimpl{std::make_unique<impl>()}
{
  pimpl->template init<psz_header>(header, false /* decomp */, debug);
}

template <class C>
Compressor<C>::~Compressor(){};

template <class C>
Compressor<C>* Compressor<C>::compress(
    pszctx* ctx, T* in, BYTE** out, size_t* outlen, void* stream)
{
  pimpl->compress_predict(ctx, in, stream);

  if (ctx->codec1_type == Huffman) pimpl->compress_histogram(ctx, stream);

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
Compressor<C>* Compressor<C>::decompress(psz_header* header, BYTE* in, T* out, void* stream)
{
  // TODO host having copy of header when compressing
  if (not header) {
    header = new psz_header;
#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
    CHECK_GPU(cudaMemcpyAsync(
        header, in, sizeof(psz_header), cudaMemcpyDeviceToHost, (cudaStream_t)stream));
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
Compressor<C>* Compressor<C>::dump_compress_intermediate(pszctx* ctx, psz_stream_t stream)
{
  auto dump_name = [&](string t, string suffix = ".quant") -> string {
    return string(ctx->file_input) + "." + string(ctx->char_meta_eb) + suffix + "_" + t;
  };

  if (ctx->dump_hist) {
    cudaStreamSynchronize((cudaStream_t)stream);
    auto& d = pimpl->mem->_hist;
    // TODO caution! lift hardcoded dtype (hist)
    d->control({D2H})->file(dump_name("u4", ".hist").c_str(), ToFile);
  }
  if (ctx->dump_quantcode) {
    cudaStreamSynchronize((cudaStream_t)stream);
    auto& d = pimpl->mem->_ectrl;
    d->control({MallocHost, D2H})
        ->file(dump_name("u" + to_string(sizeof(E)), ".quant").c_str(), ToFile);
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
    for (auto i = 0; i < STAGE_END - 1; i++) ext_timerecord[i] = pimpl->timerecord_v2[i];
  return this;
}

}  // namespace psz

#undef COLLECT_TIME

#endif /* A2519F0E_602B_4798_A8EF_9641123095D9 */
