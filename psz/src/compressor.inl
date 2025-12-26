#include <iostream>
#include <string>

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
#include "mem/sp_interface.h"
#include "utils/io.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

using Toggle = psz::Toggle;

template <typename T, Toggle ZigZag>
using GPU_c_lorenzo_nd =
    psz::module::GPU_c_lorenzo_nd<T, psz::PredConfig<T, psz::PredFunc<ZigZag>>, psz::Buf_Comp<T>>;

template <typename T, Toggle ZigZag>
using GPU_x_lorenzo_nd =
    psz::module::GPU_x_lorenzo_nd<T, psz::PredConfig<T, psz::PredFunc<ZigZag>>>;

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)

#define CONCAT_ON_DEVICE(dst, src, nbyte, stream) \
  if (nbyte != 0) cudaMemcpyAsync(dst, src, nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);

#elif defined(PSZ_USE_1API)

#define CONCAT_ON_DEVICE(dst, src, nbyte, stream) \
  if (nbyte != 0) ((sycl::queue*)stream)->memcpy(dst, src, nbyte);

#endif

#define DST(FIELD, OFFSET) ((void*)(mem->compressed_d() + ctx->header->entry[FIELD] + OFFSET))

#define PIPELINE ctx->header->pipeline
#define RC ctx->header->rc

namespace psz {

template <typename T>
Compressor<T>::Compressor(psz_ctx* ctx) : header_ref(ctx->header)
{
  constexpr auto iscompression = true;

  // extract context
  const auto pardeg = ctx->header->vle_pardeg;
  const auto x = ctx->header->len.x, y = ctx->header->len.y, z = ctx->header->len.z;
  len_linear = x * y * z;

  // optimize component(s)
  psz::module::GPU_histogram_generic<E>::init(
      len_linear, mem->max_bklen, hist_generic_grid_dim, hist_generic_block_dim,
      hist_generic_shmem_use, hist_generic_repeat);

  // initialize internal buffers
  mem = new Buf_Comp<T>(ctx->header->len, iscompression);
  buf_hf = new phf::Buf<E>(mem->len_linear, mem->max_bklen);
}

template <typename T>
Compressor<T>::Compressor(psz_header* header) : header_ref(header)
{
  constexpr auto iscompression = false;

  // extract context
  const auto pardeg = header->vle_pardeg;
  const auto x = header->len.x, y = header->len.y, z = header->len.z;
  len_linear = x * y * z;

  // initialize internal buffers
  mem = new Buf_Comp<T>(header->len, iscompression);
  buf_hf = new phf::Buf<E>(mem->len_linear, mem->max_bklen);
}

template <typename T>
Compressor<T>::~Compressor()
{
  if (mem) delete mem;
  if (buf_hf) delete buf_hf;
};

template <typename T>
void Compressor<T>::compress(psz_ctx* ctx, T* in, BYTE** out, size_t* outlen, void* stream)
{
  compress_data_processing(ctx, in, stream);
  compress_merge_update_header(ctx, out, outlen, stream);
}

template <typename T>
void Compressor<T>::compress_data_processing(psz_ctx* ctx, T* in, void* stream)
{
  auto len = ctx->header->len;

  eb = RC.eb, eb_r = 1 / eb;
  ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  const auto predictor = PIPELINE.predictor;

  if (predictor == Lorenzo)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagDisabled>::kernel(
        in, len, mem->ectrl_d(), (void*)mem->buf_outlier2(), mem->top1_d(), eb, RC.radius, stream);
  else if (predictor == LorenzoZigZag)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagEnabled>::kernel(
        in, len, mem->ectrl_d(), (void*)mem->buf_outlier2(), mem->top1_d(), eb, RC.radius, stream);
  else if (predictor == LorenzoProto)
    psz::module::GPU_PROTO_c_lorenzo_nd_with_outlier<T, E>::kernel(
        in, len, mem->ectrl_d(), (void*)mem->buf_outlier2(), ebx2, ebx2_r, RC.radius, stream);
  else if (predictor == Spline)
    psz::module::GPU_predict_spline(
        in, len, mem->ectrl_d(), mem->ectrl_len3(), mem->anchor_d(), mem->anchor_len3(),
        (void*)mem->buf_outlier(), ebx2, eb_r, RC.radius, stream);

  /* make outlier count seen on host */
  sync_by_stream(stream);
  [[deprecated("outlier handling not updated in OOD; use non-OOD instead")]] auto _splen =
      mem->outlier2_host_get_num();
  ctx->header->splen = _splen;

  if (PIPELINE.codec1 != Huffman) goto ENCODING_STEP;

  if (PIPELINE.hist == psz_hist::HistogramSparse)
    psz::module::GPU_histogram_Cauchy<E>::kernel(
        mem->ectrl_d(), len_linear, mem->hist_d(), ctx->dict_size, stream);
  else if (PIPELINE.hist == psz_hist::HistogramGeneric)
    psz::module::GPU_histogram_generic<E>::kernel(
        mem->ectrl_d(), len_linear, mem->hist_d(), ctx->dict_size, hist_generic_grid_dim,
        hist_generic_block_dim, hist_generic_shmem_use, hist_generic_repeat, stream);

ENCODING_STEP:

  memcpy_allkinds<D2H>(mem->hist_h(), mem->hist_d(), ctx->dict_size);
  phf::high_level<E>::build_book(buf_hf, mem->hist_h(), ctx->dict_size, stream);

  phf_header dummy_header;
  phf::high_level<E>::encode(
      buf_hf, mem->ectrl_d(), len_linear, &comp_codec_out, &comp_codec_outlen, dummy_header,
      stream);
}

template <typename T>
void Compressor<T>::compress_merge_update_header(
    psz_ctx* ctx, BYTE** out, size_t* outlen, void* stream)
{
  auto predictor = PIPELINE.predictor;

  nbyte[PSZ_HEADER] = sizeof(psz_header);
  nbyte[PSZ_ENCODED] = sizeof(BYTE) * comp_codec_outlen;
  nbyte[PSZ_ANCHOR] = predictor == Spline ? sizeof(T) * mem->anchor_len() : 0;
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

template <typename T>
void Compressor<T>::clear_buffer()
{
  mem->clear_buffer();
}

template <typename T>
void Compressor<T>::decompress(psz_header* header, BYTE* in, T* out, psz_stream_t stream)
{
  auto access = [&](int FIELD, szt offset_nbyte = 0) {
    return (void*)(in + header->entry[FIELD] + offset_nbyte);
  };

  // v1
  // auto d_spval = (T*)access(PSZ_SPFMT);
  // auto d_spidx = (M*)access(PSZ_SPFMT, header->splen * sizeof(T));
  // v2
  auto d_spval_idx = (_portable::compact_cell<T, M>*)access(PSZ_SPFMT);

  auto d_anchor = (T*)access(PSZ_ANCHOR);
  auto d_space = out, d_xdata = out;  // aliases
  auto len = header->len;

  eb = header->rc.eb;
  eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

STEP_SCATTER:

  if (header->splen != 0)
    psz::module::GPU_scatter<T, M>::kernel_v2(d_spval_idx, header->splen, d_space, stream);

STEP_DECODING:

  phf_header h;
  memcpy_allkinds<D2H>((BYTE*)&h, (BYTE*)access(PSZ_ENCODED), sizeof(phf_header));
  phf::high_level<E>::decode(buf_hf, h, (BYTE*)access(PSZ_ENCODED), mem->ectrl_d(), stream);

STEP_PREDICT:

  if (header->pipeline.predictor == Lorenzo)
    GPU_x_lorenzo_nd<T, Toggle::ZigZagDisabled>::kernel(
        mem->ectrl_d(), d_space, d_xdata, len, eb, header->rc.radius, stream);
  else if (header->pipeline.predictor == LorenzoZigZag)
    GPU_x_lorenzo_nd<T, Toggle::ZigZagEnabled>::kernel(
        mem->ectrl_d(), d_space, d_xdata, len, eb, header->rc.radius, stream);
  else if (header->pipeline.predictor == LorenzoProto)
    psz::module::GPU_PROTO_x_lorenzo_nd<T, E>::kernel(
        mem->ectrl_d(), d_space, d_xdata, len, ebx2, ebx2_r, header->rc.radius, stream);
  else if (header->pipeline.predictor == Spline)
    if constexpr (std::is_same_v<T, f4>)
      psz::module::GPU_spline_reconstruct<T, E>::null(/*     */);
}

// public getter
template <typename T>
void Compressor<T>::dump_compress_intermediate(psz_ctx* ctx, psz_stream_t stream)
{
  auto dump_name = [&](string t, string suffix = ".quant") -> string {
    return string(ctx->cli->file_input)                                                //
           + "." + string(ctx->cli->char_mode) + "_" + string(ctx->cli->char_meta_eb)  //
           + "." + "bk_" + to_string(RC.radius * 2)                                    //
           + "." + suffix + "_" + t;
  };

  cudaStreamSynchronize((cudaStream_t)stream);

  if (ctx->cli->dump_hist) {
    auto h_hist = MAKE_UNIQUE_HOST(Freq, mem->max_bklen);
    memcpy_allkinds<D2H>(h_hist.get(), mem->hist_d(), RC.radius * 2, stream);
    _portable::utils::tofile(dump_name("u4", "ht"), h_hist.get(), RC.radius * 2);
  }
  if (ctx->cli->dump_quantcode) {
    cout << "[psz::dump] dumping quantization codebook to file: " << dump_name("quant") << endl;
    auto h_ectrl = MAKE_UNIQUE_HOST(E, len_linear);
    memcpy_allkinds<D2H>(h_ectrl.get(), mem->ectrl_d(), len_linear, stream);
    _portable::utils::tofile(
        dump_name("u" + to_string(sizeof(E)), "qt"), h_ectrl.get(), len_linear);
  }
}

// public getter
template <typename T>
void Compressor<T>::export_header(psz_header& ext_header)
{
  ext_header = *header_ref;
}

}  // namespace psz

#define PPL_IMPL(RET_TYPE)          \
  template <typename T, typename E> \
  RET_TYPE psz::compression_pipeline<T, E>

PPL_IMPL(void*)::compress_init(psz_ctx* ctx)
{
  constexpr auto iscompression = true;

  // extract context
  const auto pardeg = ctx->header->vle_pardeg;
  const auto x = ctx->header->len.x, y = ctx->header->len.y, z = ctx->header->len.z;

  // initialize internal buffers
  auto mem = new Buf_Comp<T, E>(ctx->header->len, iscompression);
  mem->register_header(ctx->header);
  // buf_hf = new phf::Buf<E>(mem->len, mem->max_bklen);

  // optimize component(s)
  psz::module::GPU_histogram_generic<E>::init(
      mem->len_linear, mem->max_bklen, mem->hist_generic_grid_dim, mem->hist_generic_block_dim,
      mem->hist_generic_shmem_use, mem->hist_generic_repeat);

  return mem;
}

PPL_IMPL(void*)::decompress_init(psz_header* header)
{
  // extract context
  const auto pardeg = header->vle_pardeg;
  const auto x = header->len.x, y = header->len.y, z = header->len.z;

  // initialize internal buffers
  auto mem = new Buf_Comp<T, E>(header->len, false);
  mem->register_header(header);
  return mem;
}

namespace {

template <typename T, typename E>
void compress_data_processing(psz_ctx* ctx, PSZ_BUF* mem, T* in, void* stream)
{
  auto eb = RC.eb, eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  const auto len = ctx->header->len;
  const auto len_linear = mem->len_linear;
  const auto predictor = PIPELINE.predictor;
  const auto radius = RC.radius;

  if (predictor == Lorenzo)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagDisabled>::compressor_kernel(
        mem, in, len, eb, radius, stream);
  else if (predictor == LorenzoZigZag)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagEnabled>::compressor_kernel(
        mem, in, len, eb, radius, stream);
  else if (predictor == LorenzoProto)
    psz::module::GPU_PROTO_c_lorenzo_nd_with_outlier<T, E>::kernel(
        in, len, mem->ectrl_d(), (void*)mem->buf_outlier2(), ebx2, ebx2_r, RC.radius, stream);
  else if (predictor == Spline)
    if constexpr (std::is_same_v<T, f4>)
      psz::module::GPU_spline_construct<T, E>::null(/*       */);

  /* make outlier count seen on host */
  sync_by_stream(stream);
  ctx->header->splen = mem->outlier2_host_get_num();
  if (ctx->header->splen == mem->buf_outlier2()->max_allowed_num()) {
    cerr << "[psz::warning::pipeline] max allowed num-outlier (" << mem->outlier_ratio()
         << " * input-len) exceeded, returning..." << endl;
    [[deprecated("need to return status")]] auto status = PSZ_WARN_OUTLIER_TOO_MANY;
    // return PSZ_WARN_OUTLIER_TOO_MANY;
  }

  if (ctx->header->pipeline.codec1 != Huffman) goto ENCODING_STEP;

  if (ctx->header->pipeline.hist == psz_hist::HistogramSparse)
    psz::module::GPU_histogram_Cauchy<E>::kernel(
        mem->ectrl_d(), len_linear, mem->hist_d(), ctx->dict_size, stream);
  else if (ctx->header->pipeline.hist == psz_hist::HistogramGeneric)
    psz::module::GPU_histogram_generic<E>::kernel(
        mem->ectrl_d(), len_linear, mem->hist_d(), ctx->dict_size, mem->hist_generic_grid_dim,
        mem->hist_generic_block_dim, mem->hist_generic_shmem_use, mem->hist_generic_repeat,
        stream);

ENCODING_STEP:

  memcpy_allkinds<D2H>(mem->hist_h(), mem->hist_d(), ctx->dict_size);
  phf::high_level<E>::build_book(mem->buf_hf(), mem->hist_h(), ctx->dict_size, stream);

  phf_header dummy_header;
  phf::high_level<E>::encode(
      mem->buf_hf(), mem->ectrl_d(), len_linear, &mem->comp_codec_out, &mem->comp_codec_outlen,
      dummy_header, stream);
}

template <typename T, typename E>
void compress_merge_update_header(
    psz_ctx* ctx, PSZ_BUF* mem, u1** out, size_t* outlen, void* stream)
{
  auto predictor = PIPELINE.predictor;

  mem->nbyte[PSZ_HEADER] = sizeof(psz_header);
  mem->nbyte[PSZ_ENCODED] = sizeof(u1) * mem->comp_codec_outlen;
  mem->nbyte[PSZ_ANCHOR] = predictor == Spline ? sizeof(T) * mem->anchor_len() : 0;
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

PPL_IMPL(void)::compress_analysis(psz_ctx* ctx, PSZ_BUF* mem, T* in, u4* h_hist, void* stream)
{
  auto eb = RC.eb, eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  const auto len = ctx->header->len;
  const auto len_linear = mem->len_linear;
  const auto predictor = PIPELINE.predictor;
  const auto radius = RC.radius;

  if (PIPELINE.predictor == Lorenzo)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagDisabled>::compressor_kernel(
        mem, in, len, eb, radius, stream);
  else if (PIPELINE.predictor == LorenzoZigZag)
    GPU_c_lorenzo_nd<T, Toggle::ZigZagEnabled>::compressor_kernel(
        mem, in, len, eb, radius, stream);
  else if (PIPELINE.predictor == Spline)
    if constexpr (std::is_same_v<T, f4>)
      psz::module::GPU_spline_construct<T, E>::null(/*       */);

  /* make outlier count seen on host */
  sync_by_stream(stream);
  ctx->header->splen = mem->outlier_num();

  psz::module::GPU_histogram_Cauchy<E>::kernel(
      mem->ectrl_d(), len_linear, mem->hist_d(), ctx->dict_size, stream);

  memcpy_allkinds_async<D2H>(h_hist, mem->hist_d(), ctx->dict_size, stream);
  sync_by_stream(stream);

  memset_device(mem->hist_d(), ctx->dict_size, 0);
}

PPL_IMPL(void)::compress(psz_ctx* ctx, PSZ_BUF* mem, T* in, u1** out, size_t* outlen, void* stream)
{
  compress_data_processing(ctx, mem, in, stream);
  compress_merge_update_header(ctx, mem, out, outlen, stream);
}

PPL_IMPL(void)::decompress(psz_header* header, PSZ_BUF* mem, u1* in, T* out, psz_stream_t stream)
{
  auto access = [&](int FIELD, szt offset_nbyte = 0) {
    return (void*)(in + header->entry[FIELD] + offset_nbyte);
  };

  auto d_anchor = (T*)access(PSZ_ANCHOR);
  auto d_spval_idx = (_portable::compact_cell<T, M>*)access(PSZ_SPFMT);
  auto d_space = out, d_xdata = out;  // aliases
  auto len = header->len;

  const auto eb = header->rc.eb, eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

STEP_SCATTER:

  if (header->splen != 0)
    psz::module::GPU_scatter<T, M>::kernel_v2(d_spval_idx, header->splen, d_space, stream);

STEP_DECODING:

  phf_header h;
  memcpy_allkinds<D2H>((BYTE*)&h, (BYTE*)access(PSZ_ENCODED), sizeof(phf_header));
  phf::high_level<E>::decode(mem->buf_hf(), h, (BYTE*)access(PSZ_ENCODED), mem->ectrl_d(), stream);

STEP_PREDICT:

  if (header->pipeline.predictor == Lorenzo)
    GPU_x_lorenzo_nd<T, Toggle::ZigZagDisabled>::kernel(
        mem->ectrl_d(), d_space, d_xdata, len, eb, header->rc.radius, stream);
  else if (header->pipeline.predictor == LorenzoZigZag)
    GPU_x_lorenzo_nd<T, Toggle::ZigZagEnabled>::kernel(
        mem->ectrl_d(), d_space, d_xdata, len, eb, header->rc.radius, stream);
  else if (header->pipeline.predictor == LorenzoProto)
    psz::module::GPU_PROTO_x_lorenzo_nd<T, E>::kernel(
        mem->ectrl_d(), d_space, d_xdata, len, ebx2, ebx2_r, header->rc.radius, stream);
  else if (header->pipeline.predictor == Spline)
    if constexpr (std::is_same_v<T, f4>)
      psz::module::GPU_spline_reconstruct<T, E>::null(/*     */);
}

PPL_IMPL(void)::release(PSZ_BUF* mem)
{
  if (mem) delete mem;
}

PPL_IMPL(void)::compress_dump_internal_buf(psz_ctx* ctx, PSZ_BUF* mem, psz_stream_t stream)
{
  auto dump_name = [&](string t, string suffix = ".quant") -> string {
    return string(ctx->cli->file_input)                                                //
           + "." + string(ctx->cli->char_mode) + "_" + string(ctx->cli->char_meta_eb)  //
           + "." + "bk_" + to_string(RC.radius * 2)                                    //
           + "." + suffix + "_" + t;
  };

  sync_by_stream(stream);

  if (ctx->cli->dump_hist) {
    memcpy_allkinds<D2H>(mem->hist_h(), mem->hist_d(), RC.radius * 2, stream);
    _portable::utils::tofile(dump_name("u4", "ht"), mem->hist_h(), RC.radius * 2);
  }
  if (ctx->cli->dump_quantcode) {
    cout << "[psz::dump] dumping quantization codebook to file: " << dump_name("quant") << endl;
    auto h_ectrl = MAKE_UNIQUE_HOST(E, mem->len_linear);
    memcpy_allkinds<D2H>(h_ectrl.get(), mem->ectrl_d(), mem->len_linear, stream);
    _portable::utils::tofile(
        dump_name("u" + to_string(sizeof(E)), "qt"), h_ectrl.get(), mem->len_linear);
  }
}

#undef PPL_IMPL
#undef PIPELINE
#undef RC
