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

#include "cusz/context.h"
#include "cusz/type.h"
#include "detail/compbuf.hh"
#include "detail/composite.hh"
#include "detail/compressor.hh"
#include "detail/port.hh"
#include "hf_hl.hh"
#include "kernel/hist.hh"
#include "kernel/predictor.hh"
#include "kernel/spv.hh"
#include "lc_gen/lc_gen.h"
#include "mem/cxx_backends.h"
#include "utils/err.hh"
#include "utils/io.hh"
#include "utils/timer.hh"

using Toggle = psz::Toggle;

template <typename T, Toggle ZigZag>
using GPU_c_lorenzo_nd =
    psz::module::GPU_c_lorenzo_nd<T, psz::PredConfig<T, psz::PredFunc<ZigZag>>>;

template <typename T, Toggle ZigZag>
using GPU_x_lorenzo_nd =
    psz::module::GPU_x_lorenzo_nd<T, psz::PredConfig<T, psz::PredFunc<ZigZag>>>;

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
Compressor<DType>::Compressor(psz_context* ctx) : header_ref(ctx->header)
{
  constexpr auto iscompression = true;

  // extract context
  const auto pardeg = ctx->header->vle_pardeg;
  const auto x = ctx->header->x, y = ctx->header->y, z = ctx->header->z;
  len = x * y * z;

  // optimize component(s)
  psz::module::GPU_histogram_generic<E>::init(
      len, mem->max_bklen, hist_generic_grid_dim, hist_generic_block_dim, hist_generic_shmem_use,
      hist_generic_repeat);

  // initialize internal buffers
  mem = new CompressorBuffer<DType>(x, y, z, iscompression);
  buf_hf = new phf::Buf<E>(mem->len, mem->max_bklen);
}

template <typename DType>
Compressor<DType>::Compressor(psz_header* header) : header_ref(header)
{
  constexpr auto iscompression = false;

  // extract context
  const auto pardeg = header->vle_pardeg;
  const auto x = header->x, y = header->y, z = header->z;
  len = x * y * z;

  // initialize internal buffers
  mem = new CompressorBuffer<DType>(x, y, z, iscompression);
  buf_hf = new phf::Buf<E>(mem->len, mem->max_bklen);
}

template <typename DType>
Compressor<DType>::~Compressor()
{
  if (mem) delete mem;
  if (buf_hf) delete buf_hf;
};

template <typename DType>
void Compressor<DType>::compress(pszctx* ctx, T* in, BYTE** out, size_t* outlen, void* stream)
{
  compress_data_processing(ctx, in, stream);
  compress_merge_update_header(ctx, out, outlen, stream);
}

template <typename DType>
void Compressor<DType>::compress_data_processing(pszctx* ctx, T* in, void* stream)
{
  auto len3_std = MAKE_STDLEN3(ctx->header->x, ctx->header->y, ctx->header->z);

  eb = ctx->header->eb, eb_r = 1 / eb;
  ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  if (ctx->header->pred_type == Lorenzo)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagDisabled>::kernel(
        in, len3_std, mem->ectrl(), (void*)mem->outlier(), mem->top1(), eb, ctx->header->radius,
        stream);
  else if (ctx->header->pred_type == LorenzoZigZag)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagEnabled>::kernel(
        in, len3_std, mem->ectrl(), (void*)mem->outlier(), mem->top1(), eb, ctx->header->radius,
        stream);
  else if (ctx->header->pred_type == LorenzoProto)
    psz::module::GPU_PROTO_c_lorenzo_nd_with_outlier<T, E>::kernel(
        in, len3_std, mem->ectrl(), (void*)mem->outlier(), ebx2, ebx2_r, ctx->header->radius,
        stream);
  else if (ctx->header->pred_type == Spline) {
    float time;
    if constexpr (std::is_same_v<T, float>)
      psz::module::GPU_spline_construct<T, E>::kernel_v1(
          in, len3_std, mem->anchor(), mem->anchor_len3(), mem->ectrl(), (void*)mem->compact, eb,
          eb, ctx->header->radius, ctx->header->intp_param, &time, mem->profiled_errors(),
          mem->profiled_errors_h(), mem->profiled_errors_len(), stream);
  }

  /* make outlier count seen on host */
  sync_by_stream(stream);
  ctx->header->splen = mem->compact->num_outliers();

  if (ctx->header->codec1_type != Huffman) goto ENCODING_STEP;

  if (ctx->header->hist_type == psz_histotype::HistogramSparse)
    psz::module::GPU_histogram_Cauchy<E>::kernel(
        mem->ectrl(), len, mem->hist(), ctx->dict_size, stream);
  else if (ctx->header->hist_type == psz_histotype::HistogramGeneric)
    psz::module::GPU_histogram_generic<E>::kernel(
        mem->ectrl(), len, mem->hist(), ctx->dict_size, hist_generic_grid_dim,
        hist_generic_block_dim, hist_generic_shmem_use, hist_generic_repeat, stream);

ENCODING_STEP:

  memcpy_allkinds<D2H>(mem->h_hist.get(), mem->d_hist.get(), ctx->dict_size);
  phf::high_level<E>::build_book(buf_hf, mem->h_hist.get(), ctx->dict_size, stream);

  phf_header dummy_header;
  phf::high_level<E>::encode(
      buf_hf, mem->ectrl(), len, &comp_codec_out, &comp_codec_outlen, dummy_header, stream);
}

template <typename DType>
void Compressor<DType>::compress_merge_update_header(
    pszctx* ctx, BYTE** out, size_t* outlen, void* stream)
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

template <typename DType>
void Compressor<DType>::clear_buffer()
{
  mem->clear_buffer();
}

template <typename DType>
void Compressor<DType>::decompress(psz_header* header, BYTE* in, T* out, psz_stream_t stream)
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

  phf_header h;
  memcpy_allkinds<D2H>((B*)&h, (B*)access(ENCODED), sizeof(phf_header));
  phf::high_level<E>::decode(buf_hf, h, (B*)access(ENCODED), mem->ectrl(), stream);

STEP_PREDICT:

  if (header->pred_type == Lorenzo)
    GPU_x_lorenzo_nd<T, Toggle::ZigZagDisabled>::kernel(
        mem->ectrl(), d_space, d_xdata, len3_std, eb, header->radius, stream);
  else if (header->pred_type == LorenzoZigZag)
    GPU_x_lorenzo_nd<T, Toggle::ZigZagEnabled>::kernel(
        mem->ectrl(), d_space, d_xdata, len3_std, eb, header->radius, stream);
  else if (header->pred_type == LorenzoProto)
    psz::module::GPU_PROTO_x_lorenzo_nd<T, E>::kernel(
        mem->ectrl(), d_space, d_xdata, len3_std, ebx2, ebx2_r, header->radius, stream);
  else if (header->pred_type == Spline) {
    float time;
    if constexpr (std::is_same_v<T, float>)
      psz::module::GPU_spline_reconstruct<T, E>::kernel_v1(
          d_anchor, mem->anchor_len3(), mem->ectrl(), d_xdata, len3_std, d_space, eb,
          header->radius, header->intp_param, &time, stream);
  }
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
    auto h_hist = MAKE_UNIQUE_HOST(typename CompressorBuffer<DType>::Freq, mem->max_bklen);
    memcpy_allkinds<D2H>(h_hist.get(), mem->hist(), ctx->header->radius * 2, stream);
    _portable::utils::tofile(dump_name("u4", "ht"), h_hist.get(), ctx->header->radius * 2);
  }
  if (ctx->cli->dump_quantcode) {
    cout << "[psz::dump] dumping quantization codebook to file: " << dump_name("quant") << endl;
    auto h_ectrl = MAKE_UNIQUE_HOST(E, len);
    memcpy_allkinds<D2H>(h_ectrl.get(), mem->ectrl(), len, stream);
    _portable::utils::tofile(dump_name("u" + to_string(sizeof(E)), "qt"), h_ectrl.get(), len);
  }
}

// public getter
template <typename DType>
void Compressor<DType>::export_header(psz_header& ext_header)
{
  ext_header = *header_ref;
}

}  // namespace psz

#undef COLLECT_TIME

template <typename T, typename E>
using CP = psz::compression_pipeline<T, E>;

#define PIPELINE(RET_TYPE)          \
  template <typename T, typename E> \
  RET_TYPE psz::compression_pipeline<T, E>

PIPELINE(void*)::compress_init(psz_context* ctx)
{
  constexpr auto iscompression = true;

  // extract context
  const auto pardeg = ctx->header->vle_pardeg;
  const auto x = ctx->header->x, y = ctx->header->y, z = ctx->header->z;

  // initialize internal buffers
  auto mem = new CompressorBuffer<T, E>(x, y, z, iscompression);
  mem->register_header(ctx->header);
  // buf_hf = new phf::Buf<E>(mem->len, mem->max_bklen);

  // optimize component(s)
  psz::module::GPU_histogram_generic<E>::init(
      mem->len, mem->max_bklen, mem->hist_generic_grid_dim, mem->hist_generic_block_dim,
      mem->hist_generic_shmem_use, mem->hist_generic_repeat);

  cout << mem->len << endl;

  return mem;
}

PIPELINE(void*)::decompress_init(psz_header* header)
{
  constexpr auto iscompression = false;

  // extract context
  const auto pardeg = header->vle_pardeg;
  const auto x = header->x, y = header->y, z = header->z;

  // initialize internal buffers
  auto mem = new CompressorBuffer<T, E>(x, y, z, iscompression);
  mem->register_header(header);
  return mem;
}

namespace {

template <typename T, typename E>
void compress_impl(pszctx* ctx, PSZ_BUF* mem, T* in, u1** out, size_t* outlen, void* stream)
{
  using M = typename PSZ_BUF::M;
  const auto pred_type = ctx->header->pred_type;
  const auto hist_type = ctx->header->hist_type;
  const auto codec1_type = ctx->header->codec1_type;
  const auto codec2_type = ctx->header->_future_codec2_type;

  auto len3_std = MAKE_STDLEN3(ctx->header->x, ctx->header->y, ctx->header->z);

  auto eb = ctx->header->eb, eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;
  const auto len = mem->len;

  //// prediction and quantization ///////////////////////////////////////////////
  if (pred_type == Lorenzo)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagDisabled>::kernel(
        in, len3_std, mem->ectrl(), (void*)mem->outlier(), mem->top1(), eb, ctx->header->radius,
        stream);
  else if (pred_type == LorenzoZigZag)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagEnabled>::kernel(
        in, len3_std, mem->ectrl(), (void*)mem->outlier(), mem->top1(), eb, ctx->header->radius,
        stream);
  else if (pred_type == LorenzoProto)
    psz::module::GPU_PROTO_c_lorenzo_nd_with_outlier<T, E>::kernel(
        in, len3_std, mem->ectrl(), (void*)mem->outlier(), ebx2, ebx2_r, ctx->header->radius,
        stream);
  else if (pred_type == Spline) {
    float time;
    if constexpr (std::is_same_v<T, float>)
      psz::module::GPU_spline_construct<T, E>::kernel_v1(
          in, len3_std, mem->anchor(), mem->anchor_len3(), mem->ectrl(), (void*)mem->compact, eb,
          eb, ctx->header->radius, ctx->header->intp_param, &time, mem->profiled_errors(),
          mem->profiled_errors_h(), mem->profiled_errors_len(), stream);
  }

  //// retrieve outlier count on host ////////////////////////////////////////////
  sync_by_stream(stream);
  ctx->header->splen = mem->compact->num_outliers();

  //// encoding: pass 1 (not updated in OOD ver.) ////////////////////////////////
  auto run_Huffman_precursor = [&]() {
    if (hist_type == psz_histotype::HistogramSparse)
      psz::module::GPU_histogram_Cauchy<E>::kernel(
          mem->ectrl(), len, mem->hist(), ctx->dict_size, stream);
    else if (hist_type == psz_histotype::HistogramGeneric)
      psz::module::GPU_histogram_generic<E>::kernel(
          mem->ectrl(), len, mem->hist(), ctx->dict_size, mem->hist_generic_grid_dim,
          mem->hist_generic_block_dim, mem->hist_generic_shmem_use, mem->hist_generic_repeat,
          stream);
    memcpy_allkinds<D2H>(mem->h_hist.get(), mem->d_hist.get(), ctx->dict_size);
  };

  if (codec1_type == Huffman) {
    [[deprecated("TODO: dummy header for HF?")]] phf_header dummy_header;

    run_Huffman_precursor();
    phf::high_level<E>::build_book(mem->buf_hf, mem->h_hist.get(), ctx->dict_size, stream);
    phf::high_level<E>::encode(
        mem->buf_hf, mem->ectrl(), len, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream);
  }
  else if (codec1_type == LC) {
    [[deprecated(
        "TODO: should not be a modified pointer (comp_codec_out) inside TCMS_COMPRESS where an "
        "allocation happens; also, remove CPU-side time_tcms")]] float time_tcms;
    if constexpr (std::is_same_v<E, u1>)
      TCMS_COMPRESS(
          mem->ectrl(), len * sizeof(E), &mem->comp_codec_out, &mem->comp_codec_outlen, &time_tcms,
          stream);
    else
      throw std::runtime_error("The reference pipeline using TCMS requires E == u1.");
  }

  //// merge segments after encoding pass 1 /////////////////////////////
  mem->nbyte[HEADER] = sizeof(psz_header);
  mem->nbyte[ENCODED] = sizeof(u1) * mem->comp_codec_outlen;
  mem->nbyte[ANCHOR] = pred_type == Spline ? sizeof(T) * mem->anchor_len() : 0;
  mem->nbyte[SPFMT] = (sizeof(T) + sizeof(M)) * ctx->header->splen;

  // clang-format off
  ctx->header->entry[0] = 0;
  // *.END + 1; need to know the ending position
  for (auto i = 1; i < END + 1; i++) ctx->header->entry[i] = mem->nbyte[i - 1];
  for (auto i = 1; i < END + 1; i++) ctx->header->entry[i] += ctx->header->entry[i - 1];

  CONCAT_ON_DEVICE(DST(ANCHOR, 0), mem->anchor(), mem->nbyte[ANCHOR], stream);
  CONCAT_ON_DEVICE(DST(ENCODED, 0), mem->comp_codec_out, mem->nbyte[ENCODED], stream);
  CONCAT_ON_DEVICE(DST(SPFMT, 0), mem->compact_val(), sizeof(T) * ctx->header->splen, stream);
  CONCAT_ON_DEVICE(DST(SPFMT, sizeof(T) * ctx->header->splen), mem->compact_idx(), sizeof(M) * ctx->header->splen, stream);
  // clang-format on

  //// encoding: pass 2 is transparent to compressor. ////////////////////////////

  //-- pipeline 1, Huffman + LC-RTR --------------------------------------------//
  if (codec1_type == Huffman and codec2_type == LC) {
    // 0 HEADER
    // ---------  ENC2-RTR: start
    // 1 ENC1-HF
    // 2 ANCHOR
    // 3 SPFMT
    // ---------  ENC2-RTR: end
    // 4 END
    [[deprecate("TODO: remove CPU-side time_rtr")]] float time_rtr;
    [[deprecated(
        "TODO: should not be a modified pointer inside RTR_COMPRESS where an allocation "
        "happens")]] byte_t* comp_rtr_out;
    size_t comp_rtr_outlen;

    RTR_COMPRESS(
        (uint8_t*)DST(ENCODED, 0), mem->nbyte[ENCODED] + mem->nbyte[ANCHOR] + mem->nbyte[SPFMT],
        &comp_rtr_out, &comp_rtr_outlen, &time_rtr, stream);
    cudaMemcpyAsync(
        DST(ENCODED, 0), (void*)comp_rtr_out, comp_rtr_outlen, cudaMemcpyDeviceToDevice,
        (cudaStream_t)stream);
    sync_by_stream(stream);
    ctx->header->entry[ENC_PASS2_END] = ctx->header->entry[ENCODED] + comp_rtr_outlen;
  }
  else if (codec1_type == LC and codec2_type == LC) {
    [[deprecate("TODO: remove CPU-side time_bitr")]] float time_bitr;
    [[deprecated(
        "TODO: should not be a modified pointer inside BITR_COMPRESS where an allocation "
        "happens")]] byte_t* comp_bitr_out;
    size_t comp_bitr_outlen;

    BITR_COMPRESS(
        (uint8_t*)DST(ANCHOR, 0), mem->nbyte[ANCHOR] + mem->nbyte[SPFMT], &comp_bitr_out,
        &comp_bitr_outlen, &time_bitr, stream);
    cudaMemcpyAsync(
        DST(ANCHOR, 0), (void*)comp_bitr_out, comp_bitr_outlen, cudaMemcpyDeviceToDevice,
        (cudaStream_t)stream);
    sync_by_stream(stream);
    ctx->header->entry[ENC_PASS2_END] = ctx->header->entry[ANCHOR] + comp_bitr_outlen;
  }

  //// outputs of this function //////////////////////////////////////////////////
  *out = mem->compressed();
  *outlen = pszheader_filesize(ctx->header);
}

}  // namespace

PIPELINE(void)::compress(pszctx* ctx, PSZ_BUF* mem, T* in, u1** out, size_t* outlen, void* stream)
{
  compress_impl(ctx, mem, in, out, outlen, stream);
}

PIPELINE(void)::decompress(psz_header* header, PSZ_BUF* mem, u1* in, T* out, psz_stream_t stream)
{
  using M = typename PSZ_BUF::M;
  using B = typename PSZ_BUF::B;
  phf_header h;

  auto access = [&](int FIELD, szt offset_nbyte = 0) {
    return (void*)(in + header->entry[FIELD] + offset_nbyte);
  };

  T* d_space = out;  // alias 1
  T* d_xdata = out;  // alias 2

  T* d_anchor = nullptr;
  T* d_spval = nullptr;
  M* d_spidx = nullptr;

  auto len3_std = MAKE_STDLEN3(header->x, header->y, header->z);
  const auto len = header->x * header->y * header->z;
  const auto eb = header->eb, eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  //// decoding passes 1 and 2 (-> encoding passes 2 and 1) //////////////////////

  //-- pipeline 1, Huffman + LC-RTR --------------------------------------------//
  if (header->codec1_type == Huffman and header->_future_codec2_type == LC) {
    // 0 HEADER
    // ---------  ENC2-RTR: start
    // 1 ENC1-HF
    // 2 ANCHOR
    // 3 SPFMT
    // ---------  ENC2-RTR: end
    // 4 END
    [[deprecated("TODO: remove CPU-side time_rtr")]] float time_rtr;
    [[deprecated(
        "TODO: remove extra ad hoc buffer, rtr_dec_output, which is allocated inside "
        "RTR_DECOMPRESS")]] void* rtr_dec_output;
    // decoding pass 1. input: (overwritten) encoded segment, output: *_dec_out.
    RTR_DECOMPRESS((uint8_t*)access(ENCODED), &rtr_dec_output, &time_rtr);
    // decoding pass 2. input: *_dec_out, output: mem->ectrl().
    memcpy_allkinds<D2H>((B*)&h, (B*)rtr_dec_output, sizeof(phf_header));  // get HF header
    phf::high_level<E>::decode(mem->buf_hf, h, (B*)rtr_dec_output, mem->ectrl(), stream);
    // assign the rest (pipeline 1)
    [[deprecated("TODO: make it a func (1)")]] auto bytes_hf_coded =
        header->entry[ENCODED - 1] - header->entry[ENCODED];
    [[deprecated("TODO: make it a func (2)")]]
    auto bytes_encoded_anchor = header->entry[ENCODED + 2] - header->entry[ENCODED];
    auto bytes_spval = header->splen * sizeof(T);
    d_anchor = (T*)((uint8_t*)rtr_dec_output + bytes_hf_coded);
    d_spval = (T*)((uint8_t*)rtr_dec_output + bytes_encoded_anchor);
    d_spidx = (M*)((uint8_t*)rtr_dec_output + bytes_encoded_anchor + bytes_spval);
  }
  //-- pipeline 2, LC-RCMS + LC-BITR -------------------------------------------//
  else if (header->codec1_type == LC and header->_future_codec2_type == LC) {
    // 0 HEADER
    // 1 ENC1-TCMS
    // -----------  ENC2-BITR: start
    // 2 ANCHOR
    // 3 SPFMT
    // -----------  ENC2-BITR: end
    // 4 END
    [[deprecated("TODO: remove CPU-side time_bitr")]] float time_bitr;
    [[deprecated(
        "TODO: remove ad hoc bitr_dec_output, which is allocated inside BITR_DECOMPRESS")]] void*
        bitr_dec_output;
    [[deprecate("TODO: remove CPU-side time_tcms")]] float time_tcms;
    [[deprecated(
        "TODO: remove ad hoc tcms_dec_output, which is allocated inside TCMS_DECOMPRESS")]] void*
        tcms_dec_output;

    // decoding pass 1
    BITR_DECOMPRESS((uint8_t*)access(ANCHOR), &bitr_dec_output, &time_bitr);
    // decoding pass 2
    TCMS_DECOMPRESS((uint8_t*)access(ENCODED), &tcms_dec_output, &time_tcms);
    [[deprecated("TODO: should not be a memcpy")]] memcpy_allkinds<D2D>(
        (B*)mem->ectrl(), (B*)tcms_dec_output, len * sizeof(E));
    [[deprecated("TODO: make it a func (3)")]] auto bytes_anchor =
        header->entry[ANCHOR + 1] - header->entry[ANCHOR];
    auto bytes_spval = header->splen * sizeof(T);
    // assign the rest (pipeline 2)
    d_anchor = (T*)d_anchor;
    d_spval = (T*)((uint8_t*)bitr_dec_output + bytes_anchor);
    d_spidx = (M*)((uint8_t*)bitr_dec_output + bytes_anchor + bytes_spval);
  }
  //-- pipeline 3 (default), Huffman + Null ------------------------------------//
  else if (header->codec1_type == LC and header->_future_codec2_type == NullCodec) {
    // 0 HEADER
    // 1 ENC1-HF
    // 2 ANCHOR
    // 3 SPFMT
    // 4 END
    d_anchor = (T*)access(ANCHOR);
    d_spval = (T*)access(SPFMT);
    d_spidx = (M*)access(SPFMT, header->splen * sizeof(T));

    memcpy_allkinds<D2H>((B*)&h, (B*)access(ENCODED), sizeof(phf_header));  // get HF header
    phf::high_level<E>::decode(mem->buf_hf, h, (B*)access(ENCODED), mem->ectrl(), stream);
  }

  //// parallel scatter after decoding pass(es) //////////////////////////////////
  [[deprecated("TODO: remove CPU-side time_sp")]] float time_sp;
  if (header->splen != 0)
    psz::spv_scatter_naive<PROPER_RUNTIME, T, M>(
        d_spval, d_spidx, header->splen, d_space, &time_sp, stream);

  //// reverse prediction ////////////////////////////////////////////////////////
  if (header->pred_type == Lorenzo)
    GPU_x_lorenzo_nd<T, Toggle::ZigZagDisabled>::kernel(
        mem->ectrl(), d_space, d_xdata, len3_std, eb, header->radius, stream);
  else if (header->pred_type == LorenzoZigZag)
    GPU_x_lorenzo_nd<T, Toggle::ZigZagEnabled>::kernel(
        mem->ectrl(), d_space, d_xdata, len3_std, eb, header->radius, stream);
  else if (header->pred_type == LorenzoProto)
    psz::module::GPU_PROTO_x_lorenzo_nd<T, E>::kernel(
        mem->ectrl(), d_space, d_xdata, len3_std, ebx2, ebx2_r, header->radius, stream);
  else if (header->pred_type == Spline) {
    float time;
    if constexpr (std::is_same_v<T, float>)
      psz::module::GPU_spline_reconstruct<T, E>::kernel_v1(
          d_anchor, mem->anchor_len3(), mem->ectrl(), d_xdata, len3_std, d_space, eb,
          header->radius, header->intp_param, &time, stream);
  }
}

PIPELINE(void)::release(PSZ_BUF* mem)
{
  if (mem) delete mem;
}

PIPELINE(void)::compress_dump_internal_buf(pszctx* ctx, PSZ_BUF* mem, psz_stream_t stream)
{
  auto dump_name = [&](string t, string suffix = ".quant") -> string {
    return string(ctx->cli->file_input)                                                //
           + "." + string(ctx->cli->char_mode) + "_" + string(ctx->cli->char_meta_eb)  //
           + "." + "bk_" + to_string(ctx->header->radius * 2)                          //
           + "." + suffix + "_" + t;
  };

  sync_by_stream(stream);

  if (ctx->cli->dump_hist) {
    using Freq = typename PSZ_BUF::Freq;
    auto h_hist = MAKE_UNIQUE_HOST(Freq, mem->max_bklen);
    memcpy_allkinds<D2H>(h_hist.get(), mem->hist(), ctx->header->radius * 2, stream);
    _portable::utils::tofile(dump_name("u4", "ht"), h_hist.get(), ctx->header->radius * 2);
  }
  if (ctx->cli->dump_quantcode) {
    cout << "[psz::dump] dumping quantization codebook to file: " << dump_name("quant") << endl;
    auto h_ectrl = MAKE_UNIQUE_HOST(E, mem->len);
    memcpy_allkinds<D2H>(h_ectrl.get(), mem->ectrl(), mem->len, stream);
    _portable::utils::tofile(dump_name("u" + to_string(sizeof(E)), "qt"), h_ectrl.get(), mem->len);
  }
}

#undef PIPELINE