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

#include "compressor.hh"
#include "cusz/context.h"
#include "cusz/type.h"
#include "detail/composite.hh"
#include "detail/port.hh"
#include "hf_hl.hh"
#include "kernel/hist.hh"
#include "kernel/predictor.hh"
#include "kernel/spvn.hh"
#include "mem/buf_comp.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"
#include "mem/sp_interface.h"
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

#define DST(FIELD, OFFSET) ((void*)(mem->compressed_d() + ctx->header->entry[FIELD] + OFFSET))

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
  mem = new Buf_Comp<DType>(x, y, z, iscompression);
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
  mem = new Buf_Comp<DType>(x, y, z, iscompression);
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
        in, len3_std, mem->ectrl_d(), (void*)mem->buf_outlier2(), mem->top1_d(), eb,
        ctx->header->radius, stream);
  else if (ctx->header->pred_type == LorenzoZigZag)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagEnabled>::kernel(
        in, len3_std, mem->ectrl_d(), (void*)mem->buf_outlier2(), mem->top1_d(), eb,
        ctx->header->radius, stream);
  else if (ctx->header->pred_type == LorenzoProto)
    psz::module::GPU_PROTO_c_lorenzo_nd_with_outlier<T, E>::kernel(
        in, len3_std, mem->ectrl_d(), (void*)mem->buf_outlier2(), ebx2, ebx2_r,
        ctx->header->radius, stream);
  else if (ctx->header->pred_type == Spline)
    psz::module::GPU_predict_spline(
        in, len3_std, mem->ectrl_d(), mem->ectrl_len3(), mem->anchor_d(), mem->anchor_len3(),
        (void*)mem->buf_outlier(), ebx2, eb_r, ctx->header->radius, stream);

  /* make outlier count seen on host */
  sync_by_stream(stream);
  [[deprecated("outlier handling not updated in OOD; use non-OOD instead")]] auto _splen =
      mem->outlier2_host_get_num();
  ctx->header->splen = _splen;

  if (ctx->header->codec1_type != Huffman) goto ENCODING_STEP;

  if (ctx->header->hist_type == psz_histotype::HistogramSparse)
    psz::module::GPU_histogram_Cauchy<E>::kernel(
        mem->ectrl_d(), len, mem->hist_d(), ctx->dict_size, stream);
  else if (ctx->header->hist_type == psz_histotype::HistogramGeneric)
    psz::module::GPU_histogram_generic<E>::kernel(
        mem->ectrl_d(), len, mem->hist_d(), ctx->dict_size, hist_generic_grid_dim,
        hist_generic_block_dim, hist_generic_shmem_use, hist_generic_repeat, stream);

ENCODING_STEP:

  memcpy_allkinds<D2H>(mem->hist_h(), mem->hist_d(), ctx->dict_size);
  phf::high_level<E>::build_book(buf_hf, mem->hist_h(), ctx->dict_size, stream);

  phf_header dummy_header;
  phf::high_level<E>::encode(
      buf_hf, mem->ectrl_d(), len, &comp_codec_out, &comp_codec_outlen, dummy_header, stream);
}

template <typename DType>
void Compressor<DType>::compress_merge_update_header(
    pszctx* ctx, BYTE** out, size_t* outlen, void* stream)
{
  auto pred_type = ctx->header->pred_type;

  nbyte[PSZ_HEADER] = sizeof(psz_header);
  nbyte[PSZ_ENCODED] = sizeof(BYTE) * comp_codec_outlen;
  nbyte[PSZ_ANCHOR] = pred_type == Spline ? sizeof(T) * mem->anchor_len() : 0;
  // nbyte[PSZ_SPFMT] = (sizeof(T) + sizeof(M)) * ctx->header->splen;
  nbyte[PSZ_SPFMT] = sizeof(_portable::compact_cell<T, M>) * ctx->header->splen;

  // clang-format off
  ctx->header->entry[0] = 0;
  // *.END + 1; need to know the ending position
  for (auto i = 1; i < PSZ_END + 1; i++) ctx->header->entry[i] = nbyte[i - 1];
  for (auto i = 1; i < PSZ_END + 1; i++) ctx->header->entry[i] += ctx->header->entry[i - 1];

  CONCAT_ON_DEVICE(DST(PSZ_ANCHOR, 0), mem->anchor_d(), nbyte[PSZ_ANCHOR], stream);
  CONCAT_ON_DEVICE(DST(PSZ_ENCODED, 0), comp_codec_out, nbyte[PSZ_ENCODED], stream);
  CONCAT_ON_DEVICE(DST(PSZ_SPFMT, 0), mem->outlier2_validx_d(), sizeof(T) * ctx->header->splen, stream);
  // clang-format on

  /* output of this function */
  *out = mem->compressed_d();
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

  // v1
  // auto d_spval = (T*)access(PSZ_SPFMT);
  // auto d_spidx = (M*)access(PSZ_SPFMT, header->splen * sizeof(T));
  // v2
  auto d_spval_idx = (_portable::compact_cell<DType, M>*)access(PSZ_SPFMT);

  auto d_anchor = (T*)access(PSZ_ANCHOR);
  auto d_space = out, d_xdata = out;  // aliases
  auto len3_std = MAKE_STDLEN3(header->x, header->y, header->z);

  eb = header->eb;
  eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

STEP_SCATTER:

  if (header->splen != 0)
    psz::module::GPU_scatter<T, M>::kernel_v2(d_spval_idx, header->splen, d_space, stream);

STEP_DECODING:

  phf_header h;
  memcpy_allkinds<D2H>((B*)&h, (B*)access(PSZ_ENCODED), sizeof(phf_header));
  phf::high_level<E>::decode(buf_hf, h, (B*)access(PSZ_ENCODED), mem->ectrl_d(), stream);

STEP_PREDICT:

  if (header->pred_type == Lorenzo)
    GPU_x_lorenzo_nd<T, Toggle::ZigZagDisabled>::kernel(
        mem->ectrl_d(), d_space, d_xdata, len3_std, eb, header->radius, stream);
  else if (header->pred_type == LorenzoZigZag)
    GPU_x_lorenzo_nd<T, Toggle::ZigZagEnabled>::kernel(
        mem->ectrl_d(), d_space, d_xdata, len3_std, eb, header->radius, stream);
  else if (header->pred_type == LorenzoProto)
    psz::module::GPU_PROTO_x_lorenzo_nd<T, E>::kernel(
        mem->ectrl_d(), d_space, d_xdata, len3_std, ebx2, ebx2_r, header->radius, stream);
  else if (header->pred_type == Spline)
    psz::module::GPU_reverse_predict_spline(
        mem->ectrl_d(), mem->ectrl_len3(), d_anchor, mem->anchor_len3(), d_xdata, len3_std, ebx2,
        eb_r, header->radius, stream);
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
    auto h_hist = MAKE_UNIQUE_HOST(Freq, mem->max_bklen);
    memcpy_allkinds<D2H>(h_hist.get(), mem->hist_d(), ctx->header->radius * 2, stream);
    _portable::utils::tofile(dump_name("u4", "ht"), h_hist.get(), ctx->header->radius * 2);
  }
  if (ctx->cli->dump_quantcode) {
    cout << "[psz::dump] dumping quantization codebook to file: " << dump_name("quant") << endl;
    auto h_ectrl = MAKE_UNIQUE_HOST(E, len);
    memcpy_allkinds<D2H>(h_ectrl.get(), mem->ectrl_d(), len, stream);
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
  auto mem = new Buf_Comp<T, E>(x, y, z, iscompression);
  mem->register_header(ctx->header);
  // buf_hf = new phf::Buf<E>(mem->len, mem->max_bklen);

  // optimize component(s)
  psz::module::GPU_histogram_generic<E>::init(
      mem->len, mem->max_bklen, mem->hist_generic_grid_dim, mem->hist_generic_block_dim,
      mem->hist_generic_shmem_use, mem->hist_generic_repeat);

  return mem;
}

PIPELINE(void*)::decompress_init(psz_header* header)
{
  constexpr auto iscompression = false;

  // extract context
  const auto pardeg = header->vle_pardeg;
  const auto x = header->x, y = header->y, z = header->z;

  // initialize internal buffers
  auto mem = new Buf_Comp<T, E>(x, y, z, iscompression);
  mem->register_header(header);
  return mem;
}

namespace {

template <typename T, typename E>
void compress_data_processing(pszctx* ctx, PSZ_BUF* mem, T* in, void* stream)
{
  auto len3_std = MAKE_STDLEN3(ctx->header->x, ctx->header->y, ctx->header->z);

  auto eb = ctx->header->eb, eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;
  const auto len = mem->len;

  if (ctx->header->pred_type == Lorenzo)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagDisabled>::kernel(
        in, len3_std, mem->ectrl_d(), (void*)mem->buf_outlier2(), mem->top1_d(), eb,
        ctx->header->radius, stream);
  else if (ctx->header->pred_type == LorenzoZigZag)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagEnabled>::kernel(
        in, len3_std, mem->ectrl_d(), (void*)mem->buf_outlier2(), mem->top1_d(), eb,
        ctx->header->radius, stream);
  else if (ctx->header->pred_type == LorenzoProto)
    psz::module::GPU_PROTO_c_lorenzo_nd_with_outlier<T, E>::kernel(
        in, len3_std, mem->ectrl_d(), (void*)mem->buf_outlier2(), ebx2, ebx2_r,
        ctx->header->radius, stream);
  else if (ctx->header->pred_type == Spline)
    psz::module::GPU_predict_spline(
        in, len3_std, mem->ectrl_d(), mem->ectrl_len3(), mem->anchor_d(), mem->anchor_len3(),
        (void*)mem->buf_outlier2(), ebx2, eb_r, ctx->header->radius, stream);

  /* make outlier count seen on host */
  sync_by_stream(stream);
  ctx->header->splen = mem->outlier2_host_get_num();
  if (ctx->header->splen == mem->buf_outlier2()->max_allowed_num()) {
    cerr << "[psz::warning::pipeline] max allowed num-outlier (" << mem->outlier_ratio()
         << " * input-len) exceeded, returning..." << endl;
    [[deprecated("need to return status")]] auto status = PSZ_WARN_OUTLIER_TOO_MANY;
    // return PSZ_WARN_OUTLIER_TOO_MANY;
  }

  if (ctx->header->codec1_type != Huffman) goto ENCODING_STEP;

  if (ctx->header->hist_type == psz_histotype::HistogramSparse)
    psz::module::GPU_histogram_Cauchy<E>::kernel(
        mem->ectrl_d(), len, mem->hist_d(), ctx->dict_size, stream);
  else if (ctx->header->hist_type == psz_histotype::HistogramGeneric)
    psz::module::GPU_histogram_generic<E>::kernel(
        mem->ectrl_d(), len, mem->hist_d(), ctx->dict_size, mem->hist_generic_grid_dim,
        mem->hist_generic_block_dim, mem->hist_generic_shmem_use, mem->hist_generic_repeat,
        stream);

ENCODING_STEP:

  memcpy_allkinds<D2H>(mem->hist_h(), mem->hist_d(), ctx->dict_size);
  phf::high_level<E>::build_book(mem->buf_hf(), mem->hist_h(), ctx->dict_size, stream);

  phf_header dummy_header;
  phf::high_level<E>::encode(
      mem->buf_hf(), mem->ectrl_d(), len, &mem->comp_codec_out, &mem->comp_codec_outlen,
      dummy_header, stream);
}

template <typename T, typename E>
void compress_merge_update_header(
    pszctx* ctx, PSZ_BUF* mem, u1** out, size_t* outlen, void* stream)
{
  auto pred_type = ctx->header->pred_type;

  mem->nbyte[PSZ_HEADER] = sizeof(psz_header);
  mem->nbyte[PSZ_ENCODED] = sizeof(u1) * mem->comp_codec_outlen;
  mem->nbyte[PSZ_ANCHOR] = pred_type == Spline ? sizeof(T) * mem->anchor_len() : 0;
  // mem->nbyte[PSZ_SPFMT] = (sizeof(T) + sizeof(psz::M)) * ctx->header->splen;
  mem->nbyte[PSZ_SPFMT] = sizeof(_portable::compact_cell<T, u4>) * ctx->header->splen;

  // clang-format off
  ctx->header->entry[0] = 0;
  // *.END + 1; need to know the ending position
  for (auto i = 1; i < PSZ_END + 1; i++) ctx->header->entry[i] = mem->nbyte[i - 1];
  for (auto i = 1; i < PSZ_END + 1; i++) ctx->header->entry[i] += ctx->header->entry[i - 1];

  CONCAT_ON_DEVICE(DST(PSZ_ANCHOR, 0), mem->anchor_d(), mem->nbyte[PSZ_ANCHOR], stream);
  CONCAT_ON_DEVICE(DST(PSZ_ENCODED, 0), mem->comp_codec_out, mem->nbyte[PSZ_ENCODED], stream);
  CONCAT_ON_DEVICE(DST(PSZ_SPFMT, 0), mem->outlier2_validx_d(), sizeof(_portable::compact_cell<T, u4>) * ctx->header->splen, stream);
  // clang-format on

  /* output of this function */
  *out = mem->compressed_d();
  *outlen = pszheader_filesize(ctx->header);
}

}  // namespace

PIPELINE(void)::compress(pszctx* ctx, PSZ_BUF* mem, T* in, u1** out, size_t* outlen, void* stream)
{
  compress_data_processing(ctx, mem, in, stream);
  compress_merge_update_header(ctx, mem, out, outlen, stream);
}

PIPELINE(void)::decompress(psz_header* header, PSZ_BUF* mem, u1* in, T* out, psz_stream_t stream)
{
  auto access = [&](int FIELD, szt offset_nbyte = 0) {
    return (void*)(in + header->entry[FIELD] + offset_nbyte);
  };

  auto d_anchor = (T*)access(PSZ_ANCHOR);
  // auto d_spval = (T*)access(PSZ_SPFMT);
  // auto d_spidx = (M*)access(PSZ_SPFMT, header->splen * sizeof(T));
  auto d_spval_idx = (_portable::compact_cell<T, M>*)access(PSZ_SPFMT);
  auto d_space = out, d_xdata = out;  // aliases
  auto len3_std = MAKE_STDLEN3(header->x, header->y, header->z);

  const auto eb = header->eb, eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

STEP_SCATTER:

  // float time_sp;

  if (header->splen != 0)
    psz::module::GPU_scatter<T, M>::kernel_v2(d_spval_idx, header->splen, d_space, stream);

STEP_DECODING:

  phf_header h;
  memcpy_allkinds<D2H>((B*)&h, (B*)access(PSZ_ENCODED), sizeof(phf_header));
  phf::high_level<E>::decode(mem->buf_hf(), h, (B*)access(PSZ_ENCODED), mem->ectrl_d(), stream);

STEP_PREDICT:

  if (header->pred_type == Lorenzo)
    GPU_x_lorenzo_nd<T, Toggle::ZigZagDisabled>::kernel(
        mem->ectrl_d(), d_space, d_xdata, len3_std, eb, header->radius, stream);
  else if (header->pred_type == LorenzoZigZag)
    GPU_x_lorenzo_nd<T, Toggle::ZigZagEnabled>::kernel(
        mem->ectrl_d(), d_space, d_xdata, len3_std, eb, header->radius, stream);
  else if (header->pred_type == LorenzoProto)
    psz::module::GPU_PROTO_x_lorenzo_nd<T, E>::kernel(
        mem->ectrl_d(), d_space, d_xdata, len3_std, ebx2, ebx2_r, header->radius, stream);
  else if (header->pred_type == Spline)
    psz::module::GPU_reverse_predict_spline(
        mem->ectrl_d(), mem->ectrl_len3(), d_anchor, mem->anchor_len3(), d_xdata, len3_std, ebx2,
        eb_r, header->radius, stream);
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
    auto h_hist = MAKE_UNIQUE_HOST(Freq, mem->max_bklen);
    memcpy_allkinds<D2H>(h_hist.get(), mem->hist_d(), ctx->header->radius * 2, stream);
    _portable::utils::tofile(dump_name("u4", "ht"), h_hist.get(), ctx->header->radius * 2);
  }
  if (ctx->cli->dump_quantcode) {
    cout << "[psz::dump] dumping quantization codebook to file: " << dump_name("quant") << endl;
    auto h_ectrl = MAKE_UNIQUE_HOST(E, mem->len);
    memcpy_allkinds<D2H>(h_ectrl.get(), mem->ectrl_d(), mem->len, stream);
    _portable::utils::tofile(dump_name("u" + to_string(sizeof(E)), "qt"), h_ectrl.get(), mem->len);
  }
}

#undef PIPELINE