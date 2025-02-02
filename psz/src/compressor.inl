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

#include <stdexcept>
#include <type_traits>

#include "compbuf.hh"
#include "compressor.hh"
#include "cusz/type.h"
#include "hfclass.hh"
#include "kernel.hh"
#include "mem/cxx_backends.h"
#include "module/cxx_module.hh"
#include "port.hh"
#include "utils/err.hh"
#include "utils/io.hh"
#include "utils/timer.hh"

#define COLLECT_TIME(NAME, TIME) timerecord.push_back({const_cast<const char*>(NAME), TIME});

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)

#define CONCAT_ON_DEVICE(dst, src, nbyte, stream) \
  if (nbyte != 0) cudaMemcpyAsync(dst, src, nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);

#elif defined(PSZ_USE_1API)

#define CONCAT_ON_DEVICE(dst, src, nbyte, stream) \
  if (nbyte != 0) ((sycl::queue*)stream)->memcpy(dst, src, nbyte);

#endif

#define DST(FIELD, OFFSET) ((void*)(mem->compressed() + ctx->header->entry[FIELD] + OFFSET))

namespace psz {

template <typename DType>
struct Compressor<DType>::impl {
  using T = DType;
  using E = u2;
  using FP = T;
  using M = u4;
  using BYTE = u1;
  using B = u1;
  using H = u4;
  using H4 = u4;
  using H8 = u8;

  using CodecHF = phf::HuffmanCodec<E>;
  using CodecFZG = psz::FzgCodec;
  using Header = psz_header;
  using TimeRecord = std::vector<std::tuple<const char*, double>>;
  using timerecord_t = TimeRecord*;
  using Buf = CompressorBuffer<DType>;

  static const int HEADER = 0;
  static const int ANCHOR = 1;
  static const int ENCODED = 2;
  static const int SPFMT = 3;
  static const int END = 4;

  // encapsulations
  GPU_EVENT event_start, event_end;
  CodecHF* codec_hf;    // Codecs have standalone internals.
  CodecFZG* codec_fzg;  //
  Buf* mem;
  std::vector<pszerror> error_list;
  TimeRecord timerecord;
  int hist_generic_grid_dim, hist_generic_block_dim, hist_generic_shmem_use, hist_generic_repeat;

  float timerecord_v2[STAGE_END];

  // variables
  size_t len;
  BYTE* comp_codec_out{nullptr};
  size_t comp_codec_outlen{0};

  // arrays
  uint32_t nbyte[END];

  float time_pred, time_hist, time_sp;

  double eb, eb_r, ebx2, ebx2_r;

  ~impl()
  {
    delete mem;
    if (codec_hf) delete codec_hf;
    if (codec_fzg) delete codec_fzg;
    event_destroy_pair(event_start, event_end);
  }

  void compress_init(pszctx* ctx, bool debug)
  {
    constexpr auto iscompression = true;

    // extract context
    const auto radius = ctx->header->radius;
    const auto pardeg = ctx->header->vle_pardeg;
    const auto bklen = radius * 2;
    const auto x = ctx->header->x, y = ctx->header->y, z = ctx->header->z;
    len = x * y * z;

    // initialize internal buffers
    mem = new CompressorBuffer<DType>(x, y, z, radius, iscompression);

    // initialize profiling
    std::tie(event_start, event_end) = event_create_pair();

    // optimize component(s)
    psz::module::GPU_histogram_generic_optimizer_on_initialization<E>(
        len, bklen, hist_generic_grid_dim, hist_generic_block_dim, hist_generic_shmem_use,
        hist_generic_repeat);

    // initialize component(s)
    // TODO decrease memory use
    codec_hf = new CodecHF(mem->len, bklen, pardeg, debug);
    codec_fzg = new CodecFZG(mem->len);
  }

  void decompress_init(psz_header* ctx, bool debug)
  {
    constexpr auto iscompression = false;

    // extract context
    const auto radius = ctx->radius;
    const auto pardeg = ctx->vle_pardeg;
    const auto bklen = radius * 2;
    const auto x = ctx->x, y = ctx->y, z = ctx->z;
    len = x * y * z;

    // initialize internal buffers
    mem = new CompressorBuffer<DType>(x, y, z, radius, iscompression);

    // initialize profiling
    std::tie(event_start, event_end) = event_create_pair();

    // initialize component(s)
    if (ctx->codec1_type == Huffman)
      codec_hf = new CodecHF(mem->len, bklen, pardeg, debug);
    else if (ctx->codec1_type == FZGPUCodec)
      codec_fzg = new CodecFZG(mem->len);
    else
      codec_hf = new CodecHF(mem->len, bklen, pardeg, debug);
  }

  void compress_data_processing(pszctx* ctx, T* in, void* stream)
  {
    auto len3_std = MAKE_STDLEN3(ctx->header->x, ctx->header->y, ctx->header->z);

    event_recording_start(event_start, stream);

    eb = ctx->header->eb, eb_r = 1 / eb;
    ebx2 = eb * 2, ebx2_r = 1 / ebx2;

    if (ctx->header->pred_type == Lorenzo)
      psz::module::GPU_c_lorenzo_nd_with_outlier<T, false, E>(
          in, len3_std, mem->ectrl(), (void*)mem->outlier(), ebx2, ebx2_r, ctx->header->radius,
          stream);
    else if (ctx->header->pred_type == LorenzoZigZag)
      psz::module::GPU_c_lorenzo_nd_with_outlier<T, true, E>(
          in, len3_std, mem->ectrl(), (void*)mem->outlier(), ebx2, ebx2_r, ctx->header->radius,
          stream);
    else if (ctx->header->pred_type == LorenzoProto)
      psz::module::GPU_PROTO_c_lorenzo_nd_with_outlier<T, E>(
          in, len3_std, mem->ectrl(), (void*)mem->outlier(), ebx2, ebx2_r, ctx->header->radius,
          stream);
    else if (ctx->header->pred_type == Spline)
      psz::module::GPU_predict_spline(
          in, len3_std, mem->ectrl(), mem->ectrl_len3(), mem->anchor(), mem->anchor_len3(),
          (void*)mem->compact, ebx2, eb_r, ctx->header->radius, stream);

    event_recording_stop(event_end, stream);
    event_time_elapsed(event_start, event_end, &time_pred);

    /* make outlier count seen on host */
    sync_by_stream(stream);
    ctx->header->splen = mem->compact->num_outliers();

    if (ctx->header->codec1_type != Huffman) goto ENCODING_STEP;

    event_recording_start(event_start, stream);

    if (ctx->header->hist_type == psz_histotype::HistogramSparse)
      psz::module::GPU_histogram_Cauchy<E>(mem->ectrl(), len, mem->hist(), ctx->dict_size, stream);
    else if (ctx->header->hist_type == psz_histotype::HistogramGeneric)
      psz::module::GPU_histogram_generic<E>(
          mem->ectrl(), len, mem->hist(), ctx->dict_size, hist_generic_grid_dim,
          hist_generic_block_dim, hist_generic_shmem_use, hist_generic_repeat, stream);

    event_recording_stop(event_end, stream);
    event_time_elapsed(event_start, event_end, &time_hist);

  ENCODING_STEP:

    if (ctx->header->codec1_type == Huffman)
      codec_hf->buildbook(mem->hist(), stream)
          ->encode(mem->ectrl(), len, &comp_codec_out, &comp_codec_outlen, stream);
    else if (ctx->header->codec1_type == FZGPUCodec)
      codec_fzg->encode(mem->ectrl(), len, &comp_codec_out, &comp_codec_outlen, stream);
  }

  void compress_merge_update_header(pszctx* ctx, BYTE** out, szt* outlen, void* stream)
  {
    auto pred_type = ctx->header->pred_type;

    nbyte[HEADER] = sizeof(psz_header);
    nbyte[ENCODED] = sizeof(BYTE) * comp_codec_outlen;
    nbyte[ANCHOR] = pred_type == Spline ? sizeof(T) * mem->anchor_len() : 0;
    nbyte[SPFMT] = (sizeof(T) + sizeof(M)) * ctx->header->splen;

    // clang-format off
    ctx->header->entry[0] = 0;
    // *.END + 1; need to know the ending position
    for (auto i = 1; i < END + 1; i++) ctx->header->entry[i] = nbyte[i - 1];
    for (auto i = 1; i < END + 1; i++) ctx->header->entry[i] += ctx->header->entry[i - 1];

    CONCAT_ON_DEVICE(DST(ANCHOR, 0), mem->anchor(), nbyte[ANCHOR], stream);
    CONCAT_ON_DEVICE(DST(ENCODED, 0), comp_codec_out, nbyte[ENCODED], stream);
    CONCAT_ON_DEVICE(DST(SPFMT, 0), mem->compact_val(), sizeof(T) * ctx->header->splen, stream);
    CONCAT_ON_DEVICE(DST(SPFMT, sizeof(T) * ctx->header->splen), mem->compact_idx(), sizeof(M) * ctx->header->splen, stream);
    // clang-format on

    /* output of this function */
    *out = mem->compressed();
    *outlen = pszheader_filesize(ctx->header);
  }

  void decompress_data_processing(psz_header* header, BYTE* in, T* out, psz_stream_t stream)
  {
    auto access = [&](int FIELD, szt offset_nbyte = 0) {
      return (void*)(in + header->entry[FIELD] + offset_nbyte);
    };

    auto d_anchor = (T*)access(ANCHOR);
    auto d_spval = (T*)access(SPFMT);
    auto d_spidx = (M*)access(SPFMT, header->splen * sizeof(T));
    auto d_space = out, d_xdata = out;  // aliases
    auto len3_std = MAKE_STDLEN3(header->x, header->y, header->z);

    eb = header->eb;
    eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  STEP_SCATTER:

    if (header->splen != 0)
      psz::spv_scatter_naive<PROPER_RUNTIME, T, M>(
          d_spval, d_spidx, header->splen, d_space, &time_sp, stream);

  STEP_DECODING:

    if (header->codec1_type == Huffman)
      codec_hf->decode((B*)access(ENCODED), mem->ectrl(), stream);
    else if (header->codec1_type == FZGPUCodec)
      codec_fzg->decode(
          (B*)access(ENCODED), pszheader_filesize(header), mem->ectrl(), mem->len, stream);

  STEP_PREDICT:

    event_recording_start(event_start, stream);
    if (header->pred_type == Lorenzo)
      psz::module::GPU_x_lorenzo_nd<T, false, E>(
          mem->ectrl(), d_space, d_xdata, len3_std, ebx2, ebx2_r, header->radius, stream);
    else if (header->pred_type == LorenzoZigZag)
      psz::module::GPU_x_lorenzo_nd<T, true, E>(
          mem->ectrl(), d_space, d_xdata, len3_std, ebx2, ebx2_r, header->radius, stream);
    else if (header->pred_type == LorenzoProto)
      psz::module::GPU_PROTO_x_lorenzo_nd<T, E>(
          mem->ectrl(), d_space, d_xdata, len3_std, ebx2, ebx2_r, header->radius, stream);
    else if (header->pred_type == Spline)
      psz::module::GPU_reverse_predict_spline(
          mem->ectrl(), mem->ectrl_len3(), d_anchor, mem->anchor_len3(), d_xdata, len3_std, ebx2,
          eb_r, header->radius, stream);

    event_recording_stop(event_end, stream);
    event_time_elapsed(event_start, event_end, &time_pred);
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

template <typename DType>
Compressor<DType>::Compressor(psz_context* ctx, bool debug) :
    pimpl{std::make_unique<impl>()}, header_ref(ctx->header)
{
  pimpl->compress_init(ctx, debug);
}

template <typename DType>
Compressor<DType>::Compressor(psz_header* header, bool debug) :
    pimpl{std::make_unique<impl>()}, header_ref(header)
{
  pimpl->decompress_init(header, debug);
}

template <typename DType>
Compressor<DType>::~Compressor(){};

template <typename DType>
void Compressor<DType>::compress(pszctx* ctx, T* in, BYTE** out, size_t* outlen, void* stream)
{
  pimpl->compress_data_processing(ctx, in, stream);
  pimpl->compress_merge_update_header(ctx, out, outlen, stream);
  pimpl->compress_collect_kerneltime();
}

template <typename DType>
void Compressor<DType>::clear_buffer()
{
  if (pimpl->codec_hf) pimpl->codec_hf->clear_buffer();
  if (pimpl->codec_fzg) pimpl->codec_fzg->clear_buffer();
  pimpl->mem->clear_buffer();
}

template <typename DType>
void Compressor<DType>::decompress(psz_header* header, BYTE* in, T* out, void* stream)
{
  pimpl->decompress_data_processing(header, in, out, stream);
  pimpl->decompress_collect_kerneltime();
}

// public getter
template <typename DType>
void Compressor<DType>::dump_compress_intermediate(pszctx* ctx, psz_stream_t stream)
{
  auto dump_name = [&](string t, string suffix = ".quant") -> string {
    return string(ctx->cli->file_input)                                                //
           + "." + string(ctx->cli->char_mode) + "_" + string(ctx->cli->char_meta_eb)  //
           + "." + "bk_" + to_string(ctx->header->radius * 2)                          //
           + "." + suffix + "_" + t;
  };

  cudaStreamSynchronize((cudaStream_t)stream);

  if (ctx->cli->dump_hist) {
    auto h_hist = MAKE_UNIQUE_HOST(typename CompressorBuffer<DType>::Freq, pimpl->mem->bklen);
    memcpy_allkinds<D2H>(h_hist.get(), pimpl->mem->hist(), pimpl->mem->bklen, stream);
    _portable::utils::tofile(dump_name("u4", "ht"), h_hist.get(), pimpl->mem->bklen);
  }
  if (ctx->cli->dump_quantcode) {
    auto h_ectrl = MAKE_UNIQUE_HOST(E, pimpl->len);
    memcpy_allkinds<D2H>(h_ectrl.get(), pimpl->mem->ectrl(), pimpl->len, stream);
    _portable::utils::tofile(
        dump_name("u" + to_string(sizeof(E)), "qt"), h_ectrl.get(), pimpl->len);
  }
}

// public getter
template <typename DType>
void Compressor<DType>::export_header(psz_header& ext_header)
{
  // ext_header = pimpl->header;
  ext_header = *header_ref;
}

template <typename DType>
void Compressor<DType>::export_timerecord(TimeRecord* ext_timerecord)
{
  if (ext_timerecord) *ext_timerecord = pimpl->timerecord;
}

template <typename DType>
void Compressor<DType>::export_timerecord(float* ext_timerecord)
{
  if (ext_timerecord)
    for (auto i = 0; i < STAGE_END - 1; i++) ext_timerecord[i] = pimpl->timerecord_v2[i];
}

}  // namespace psz

#undef COLLECT_TIME
