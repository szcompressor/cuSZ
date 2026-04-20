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
#include "lc_gen/lc_gen.h"
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
  buf_hf = new phf::Buf<E>(
      mem->len_linear, mem->max_bklen, -1,
      ctx->header->pipeline.codec1 == psz_codec::HuffmanRevisit);
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
  compress_predict_enc1(ctx, in, stream);
  compress_enc1_wrapup(ctx, out, outlen, stream);
}

template <typename T>
void Compressor<T>::compress_predict_enc1(psz_ctx* ctx, T* in, void* stream)
{
  auto len = ctx->header->len;

  eb = RC.eb, eb_r = 1 / eb;
  ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  const auto predictor = PIPELINE.predictor;

  if (predictor == Spline)
    memset_device(mem->buf_outlier()->num(), 1, 0);
  else
    memset_device(mem->buf_outlier2()->num_d(), 1, 0);

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
    if constexpr (std::is_same_v<T, f4>)
      psz::module::GPU_spline_construct<T, E>::kernel_v1(
          in, len, mem->anchor_d(), mem->anchor_len3(), mem->ectrl_d(), (void*)mem->buf_outlier(),
          eb, ctx->header->user_input_eb, ctx->header->rc.radius, ctx->header->intp_param,
          mem->profiled_errors_d(), mem->profiled_errors_h(), mem->profiled_errors_len(), stream);

  /* make outlier count seen on host */
  sync_by_stream(stream);
  if (predictor == Spline)
    ctx->header->splen = mem->buf_outlier()->num_outliers();
  else
    ctx->header->splen = mem->outlier2_host_get_num();

  if (PIPELINE.codec1 != Huffman and PIPELINE.codec1 != HuffmanRevisit) goto ENCODING_STEP;

  memset_device(mem->hist_d(), ctx->dict_size, 0);

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
  if (PIPELINE.codec1 == HuffmanRevisit)
    phf::high_level<E>::encode_HFR(
        buf_hf, mem->ectrl_d(), len_linear, &comp_codec_out, &comp_codec_outlen, dummy_header,
        stream);
  else
    phf::high_level<E>::encode(
        buf_hf, mem->ectrl_d(), len_linear, &comp_codec_out, &comp_codec_outlen, dummy_header,
        stream);

  // Keep outer archive metadata consistent with the actual PHF partitioning.
  ctx->header->vle_sublen = dummy_header.sublen;
  ctx->header->vle_pardeg = dummy_header.pardeg;
}

template <typename T>
void Compressor<T>::compress_enc1_wrapup(psz_ctx* ctx, BYTE** out, size_t* outlen, void* stream)
{
  auto predictor = PIPELINE.predictor;

  u4 nbyte[PSZ_ENC_PASS2_END] = {0};
  nbyte[PSZ_HEADER] = sizeof(psz_header);
  nbyte[PSZ_ENCODED] = sizeof(BYTE) * comp_codec_outlen;
  nbyte[PSZ_ANCHOR] = predictor == Spline ? sizeof(T) * mem->anchor_len() : 0;
  // Spline stores val[] then idx[] separately; Lorenzo/others store interleaved compact_cell
  nbyte[PSZ_SPFMT] = predictor == Spline
                         ? (sizeof(T) + sizeof(M)) * ctx->header->splen
                         : sizeof(_portable::compact_cell<T, M>) * ctx->header->splen;
  nbyte[PSZ_ENC_PASS1_END] = 0;

  // clang-format off
  ctx->header->entry[0] = 0;
  // *.END + 1; need to know the ending position
  for (auto i = 1; i < PSZ_ENC_PASS2_END + 1; i++) ctx->header->entry[i] = nbyte[i - 1];
  for (auto i = 1; i < PSZ_ENC_PASS2_END + 1; i++) ctx->header->entry[i] += ctx->header->entry[i - 1];

  CONCAT_ON_DEVICE(DST(PSZ_ANCHOR, 0), mem->anchor_d(), nbyte[PSZ_ANCHOR], stream);
  CONCAT_ON_DEVICE(DST(PSZ_ENCODED, 0), comp_codec_out, nbyte[PSZ_ENCODED], stream);
  if (predictor == Spline) {
    auto splen = ctx->header->splen;
    CONCAT_ON_DEVICE(DST(PSZ_SPFMT, 0), mem->outlier_val_d(), sizeof(T) * splen, stream);
    CONCAT_ON_DEVICE(DST(PSZ_SPFMT, sizeof(T) * splen), mem->outlier_idx_d(), sizeof(M) * splen, stream);
  } else {
    CONCAT_ON_DEVICE(DST(PSZ_SPFMT, 0), mem->outlier2_validx_d(), nbyte[PSZ_SPFMT], stream);
  }
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

  auto d_spval_idx = (_portable::compact_cell<T, M>*)access(PSZ_SPFMT);
  // Spline uses separate val/idx arrays; Lorenzo uses interleaved compact_cell
  auto d_spval = (T*)access(PSZ_SPFMT);
  auto d_spidx = (M*)access(PSZ_SPFMT, header->splen * sizeof(T));

  auto d_anchor = (T*)access(PSZ_ANCHOR);
  auto d_space = out, d_xdata = out;  // aliases
  auto len = header->len;

  eb = header->rc.eb;
  eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

STEP_SCATTER:

  if (header->pipeline.predictor == Spline) memset_device(d_space, len.x * len.y * len.z);

  if (header->splen != 0) {
    if (header->pipeline.predictor == Spline)
      psz::module::GPU_scatter<T, M>::kernel(
          d_spval, d_spidx, header->splen, d_space, nullptr, stream);
    else
      psz::module::GPU_scatter<T, M>::kernel_v2(d_spval_idx, header->splen, d_space, stream);
  }

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
      psz::module::GPU_spline_reconstruct<T, E>::kernel_v1(
          d_anchor, mem->anchor_len3(), mem->ectrl_d(), d_xdata, mem->ectrl_len3(), d_space, eb,
          header->rc.radius, header->intp_param, stream);
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
  const auto use_HFR = ctx->header->pipeline.codec1 == psz_codec::HuffmanRevisit;
  auto mem = new Buf_Comp<T, E>(ctx->header->len, iscompression, use_HFR);
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
  // initialize internal buffers
  const auto use_HFR = header->pipeline.codec1 == psz_codec::HuffmanRevisit;
  auto mem = new Buf_Comp<T, E>(header->len, false, use_HFR);
  mem->register_header(header);
  return mem;
}

PPL_IMPL(int)::compress_analysis(psz_ctx* ctx, PSZ_BUF* mem, T* in, u4* h_hist, void* stream)
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
  else if (PIPELINE.predictor == Spline) {
    memset_device(mem->buf_outlier()->num(), 1, 0);
    if constexpr (std::is_same_v<T, f4>)
      psz::module::GPU_spline_construct<T, E>::kernel_v1(
          in, len, mem->anchor_d(), mem->anchor_len3(), mem->ectrl_d(), (void*)mem->buf_outlier(),
          eb, ctx->header->user_input_eb, ctx->header->rc.radius, ctx->header->intp_param,
          mem->profiled_errors_d(), mem->profiled_errors_h(), mem->profiled_errors_len(), stream);
  }

  /* make outlier count seen on host */
  sync_by_stream(stream);
  ctx->header->splen = (PIPELINE.predictor == Spline) ? mem->buf_outlier()->num_outliers()
                                                      : mem->outlier2_host_get_num();

  psz::module::GPU_histogram_Cauchy<E>::kernel(
      mem->ectrl_d(), len_linear, mem->hist_d(), ctx->dict_size, stream);

  memcpy_allkinds_async<D2H>(h_hist, mem->hist_d(), ctx->dict_size, stream);
  sync_by_stream(stream);

  memset_device(mem->hist_d(), ctx->dict_size, 0);

  return PSZ_SUCCESS;
}

PPL_IMPL(int)::compress(psz_ctx* ctx, PSZ_BUF* mem, T* in, u1** out, size_t* outlen, void* stream)
{
  auto eb = RC.eb, eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  const auto len = ctx->header->len;
  const auto len_linear = mem->len_linear;
  const auto predictor = PIPELINE.predictor;
  const auto radius = RC.radius;

  auto compress_predict = [&]() -> int {
    if (predictor == Lorenzo)
      GPU_c_lorenzo_nd<T, Toggle::ZigZagDisabled>::compressor_kernel(
          mem, in, len, eb, radius, stream);
    else if (predictor == LorenzoZigZag)
      GPU_c_lorenzo_nd<T, Toggle::ZigZagEnabled>::compressor_kernel(
          mem, in, len, eb, radius, stream);
    else if (predictor == LorenzoProto)
      psz::module::GPU_PROTO_c_lorenzo_nd_with_outlier<T, E>::kernel(
          in, len, mem->ectrl_d(), (void*)mem->buf_outlier2(), ebx2, ebx2_r, RC.radius, stream);
    else if (predictor == Spline) {
      memset_device(mem->buf_outlier()->num(), 1, 0);
      if constexpr (std::is_same_v<T, f4>)
        psz::module::GPU_spline_construct<T, E>::kernel_v1(
            in, len, mem->anchor_d(), mem->anchor_len3(), mem->ectrl_d(),
            (void*)mem->buf_outlier(), eb, ctx->header->user_input_eb, ctx->header->rc.radius,
            ctx->header->intp_param, mem->profiled_errors_d(), mem->profiled_errors_h(),
            mem->profiled_errors_len(), stream);
    }
    else
      return PSZ_ABORT_NO_SUCH_PREDICTOR;

    /* make outlier count seen on host */
    sync_by_stream(stream);
    if (predictor == Spline) { ctx->header->splen = mem->buf_outlier()->num_outliers(); }
    else {
      ctx->header->splen = mem->outlier2_host_get_num();
      if (ctx->header->splen == mem->buf_outlier2()->max_allowed_num()) {
        cerr << "[psz::warning::pipeline] max allowed num-outlier (" << mem->outlier_ratio()
             << " * input-len) exceeded, returning..." << endl;
        return PSZ_WARN_OUTLIER_TOO_MANY;
      }
    }

    return PSZ_SUCCESS;
  };

  // shared for HF and HFR
  auto compress_histogram_and_build_book = [&]() {
    memset_device(mem->hist_d(), ctx->dict_size, 0);

    if (PIPELINE.hist == psz_hist::HistogramSparse)
      psz::module::GPU_histogram_Cauchy<E>::kernel(
          mem->ectrl_d(), len_linear, mem->hist_d(), ctx->dict_size, stream);
    else if (PIPELINE.hist == psz_hist::HistogramGeneric)
      psz::module::GPU_histogram_generic<E>::kernel(
          mem->ectrl_d(), len_linear, mem->hist_d(), ctx->dict_size, mem->hist_generic_grid_dim,
          mem->hist_generic_block_dim, mem->hist_generic_shmem_use, mem->hist_generic_repeat,
          stream);

    memcpy_allkinds<D2H>(mem->hist_h(), mem->hist_d(), ctx->dict_size);
    phf::high_level<E>::build_book(mem->buf_hf(), mem->hist_h(), ctx->dict_size, stream);
  };

  auto compress_encode_pass1_Huffman = [&]() -> int {
    compress_histogram_and_build_book();

    phf_header dummy_header;
    phf::high_level<E>::encode(
        mem->buf_hf(), mem->ectrl_d(), len_linear, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream);
    ctx->header->vle_sublen = dummy_header.sublen;
    ctx->header->vle_pardeg = dummy_header.pardeg;
    sync_by_stream(stream);
    return PSZ_SUCCESS;
  };

  // HFR: reduce-shuffle-merge encode with sparse breaking-point buffer.
  auto compress_encode_pass1_HFR = [&]() -> int {
    compress_histogram_and_build_book();

    phf_header dummy_header;
    phf::high_level<E>::encode_HFR(
        mem->buf_hf(), mem->ectrl_d(), len_linear, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream);
    ctx->header->vle_sublen = dummy_header.sublen;
    ctx->header->vle_pardeg = dummy_header.pardeg;
    sync_by_stream(stream);
    return PSZ_SUCCESS;
  };

  auto compress_encode_pass1_wrapup = [&]() {
    memset(mem->nbyte, 0, sizeof(mem->nbyte));
    mem->nbyte[PSZ_HEADER] = sizeof(psz_header);
    mem->nbyte[PSZ_ENCODED] = sizeof(u1) * mem->comp_codec_outlen;
    mem->nbyte[PSZ_ANCHOR] = predictor == Spline ? sizeof(T) * mem->anchor_len() : 0;
    mem->nbyte[PSZ_SPFMT] = predictor == Spline
                                ? (sizeof(T) + sizeof(u4)) * ctx->header->splen
                                : sizeof(_portable::compact_cell<T, u4>) * ctx->header->splen;
    mem->nbyte[PSZ_ENC_PASS1_END] = 0;

    // clang-format off
  ctx->header->entry[0] = 0;
  // *.END + 1; need to know the ending position
  for (auto i = 1; i < PSZ_ENC_PASS2_END + 1; i++) ctx->header->entry[i] = mem->nbyte[i - 1];
  for (auto i = 1; i < PSZ_ENC_PASS2_END + 1; i++) ctx->header->entry[i] += ctx->header->entry[i - 1];

  CONCAT_ON_DEVICE(DST(PSZ_ANCHOR, 0), mem->anchor_d(), mem->nbyte[PSZ_ANCHOR], stream);
  CONCAT_ON_DEVICE(DST(PSZ_ENCODED, 0), mem->comp_codec_out, mem->nbyte[PSZ_ENCODED], stream);
  if (predictor == Spline) {
    auto splen = ctx->header->splen;
    CONCAT_ON_DEVICE(DST(PSZ_SPFMT, 0), mem->outlier_val_d(), sizeof(T) * splen, stream);
    CONCAT_ON_DEVICE(DST(PSZ_SPFMT, sizeof(T) * splen), mem->outlier_idx_d(), sizeof(u4) * splen, stream);
  } else {
    CONCAT_ON_DEVICE(DST(PSZ_SPFMT, 0), mem->outlier2_validx_d(), mem->nbyte[PSZ_SPFMT], stream);
  }
    // clang-format on

    /* output of this function */
    *out = mem->compressed_d();
    *outlen = pszheader_filesize(ctx->header);
  };

  auto compress_encode_pass1_LC_TCMS = [&]() -> int {
#ifdef PSZ_USE_LC_FIXED
    // Hi-TP mode: TCMS replaces histogram+Huffman; mark hist as null in header
    ctx->header->pipeline.hist = psz_hist::NullHistogram;
    float time_tcms;
    TCMS_COMPRESS(
        (uint8_t*)mem->ectrl_d(), len_linear * sizeof(E), &mem->comp_codec_out,
        &mem->comp_codec_outlen, &time_tcms, stream);
    return PSZ_SUCCESS;
#else
    return PSZ_ABORT_NO_SUCH_CODEC;
#endif
  };

  auto compress_encode_pass2_LC_RTR = [&]() -> int {
#ifdef PSZ_USE_LC_FIXED
    // 0 HEADER
    // ---------  ENC2-RTR: start
    // 1 ENC1-HF
    // 2 ANCHOR
    // 3 SPFMT
    // ---------  ENC2-RTR: end
    // 4 END
    [[deprecated("TODO: remove CPU-side time_rtr")]] float time_rtr;
    [[deprecated(
        "TODO: should not be a modified pointer inside RTR_COMPRESS where an allocation "
        "happens")]] byte_t* comp_rtr_out;
    size_t comp_rtr_outlen;

    RTR_COMPRESS(
        (uint8_t*)DST(PSZ_ENCODED, 0),
        mem->nbyte[PSZ_ENCODED] + mem->nbyte[PSZ_ANCHOR] + mem->nbyte[PSZ_SPFMT], &comp_rtr_out,
        &comp_rtr_outlen, &time_rtr, stream);

    // reuse PSZ_ENCODED buf
    cudaMemcpyAsync(
        DST(PSZ_ENCODED, 0), (void*)comp_rtr_out, comp_rtr_outlen, cudaMemcpyDeviceToDevice,
        (cudaStream_t)stream);
    sync_by_stream(stream);
    ctx->header->entry[PSZ_ENC_PASS2_END] = ctx->header->entry[PSZ_ENCODED] + comp_rtr_outlen;

    *out = mem->compressed_d();
    *outlen = pszheader_filesize(ctx->header);
    return PSZ_SUCCESS;
#else
    return PSZ_ABORT_NO_SUCH_CODEC;
#endif
  };
  auto compress_encode_pass2_LC_BITR = [&]() -> int {

#ifdef PSZ_USE_LC_FIXED
    [[deprecated("TODO: remove CPU-side time_bitr")]] float time_bitr;
    [[deprecated(
        "TODO: should not be a modified pointer inside BITR_COMPRESS where an allocation "
        "happens")]] byte_t* comp_bitr_out;
    size_t comp_bitr_outlen;

    // Sync stream: wrapup async-copied anchor/spfmt to compressed_d; BITR uses default stream
    cudaStreamSynchronize((cudaStream_t)stream);
    BITR_COMPRESS(
        (uint8_t*)DST(PSZ_ANCHOR, 0), mem->nbyte[PSZ_ANCHOR] + mem->nbyte[PSZ_SPFMT],
        &comp_bitr_out, &comp_bitr_outlen, &time_bitr, stream);
    cudaMemcpyAsync(
        DST(PSZ_ANCHOR, 0), (void*)comp_bitr_out, comp_bitr_outlen, cudaMemcpyDeviceToDevice,
        (cudaStream_t)stream);
    sync_by_stream(stream);
    ctx->header->entry[PSZ_ENC_PASS2_END] = ctx->header->entry[PSZ_ANCHOR] + comp_bitr_outlen;

    *out = mem->compressed_d();
    *outlen = pszheader_filesize(ctx->header);
    return PSZ_SUCCESS;
#else
    return PSZ_ABORT_NO_SUCH_CODEC;
#endif
  };

  //// pipelines

  // Tian et al. 2020; Tian et al. 2021
  auto compress_encode_default = [&]() -> int {
    auto status = (PIPELINE.codec1 == HuffmanRevisit) ? compress_encode_pass1_HFR()
                                                      : compress_encode_pass1_Huffman();
    if (status != PSZ_SUCCESS) return status;
    compress_encode_pass1_wrapup();
    return PSZ_SUCCESS;
  };

  // Liu, Tian, Wu et al. 2024; Wu and Pan et al. 2025
  auto compress_encode_HiCR = [&]() -> int {
    auto status1 = compress_encode_pass1_Huffman();
    if (status1 != PSZ_SUCCESS) return status1;
    compress_encode_pass1_wrapup();
    auto status2 = compress_encode_pass2_LC_RTR();
    return PSZ_SUCCESS;
  };

  // Liu, Tian, Wu et al. 2024; Wu and Pan et al. 2025
  // HiTP ectrl-only: TCMS for ectrl, raw anchor+spfmt (no BITR, fallback)
  auto compress_encode_HiTP_ectrl = [&]() -> int {
    auto status1 = compress_encode_pass1_LC_TCMS();
    if (status1 != PSZ_SUCCESS) return status1;
    compress_encode_pass1_wrapup();
    return PSZ_SUCCESS;
  };

  // Liu, Tian, Wu et al. 2024; Wu and Pan et al. 2025
  auto compress_encode_HiTP = [&]() -> int {
    auto status1 = compress_encode_pass1_LC_TCMS();
    if (status1 != PSZ_SUCCESS) return status1;
    compress_encode_pass1_wrapup();
    auto status2 = compress_encode_pass2_LC_BITR();
    if (status2 != PSZ_SUCCESS) return status2;
    return PSZ_SUCCESS;
  };

  //// execution

  auto status_pred = compress_predict();
  if (status_pred != PSZ_SUCCESS) return status_pred;

#ifdef PSZ_USE_LC_FIXED
  // default:  HF(ec-quant) + raw(anchor/spfmt)
  // HiCR:     default + RTR(full block)
  // HiTP:     TCMS(ec-quant) + BITR(anchor/spfmt)
  // fallback: TCMS(ec-quant) + raw(anchor/spfmt)
  auto status_encode = (PIPELINE.codec2 == LC) ? ((PIPELINE.codec1 == LC) ? compress_encode_HiTP()
                                                                          : compress_encode_HiCR())
                       : (PIPELINE.codec1 == LC) ? compress_encode_HiTP_ectrl()
                                                 : compress_encode_default();
#else
  // LC not compiled: fall back to Huffman regardless of requested codec1/codec2.
  // Correct the header pipeline so the tag and CR reflect what actually ran.
  ctx->header->pipeline.codec1 = Huffman;
  ctx->header->pipeline.codec2 = NullCodec;
  auto status_encode = compress_encode_default();
#endif
  if (status_encode != PSZ_SUCCESS) return status_encode;

  return PSZ_SUCCESS;
}

PPL_IMPL(int)::decompress(psz_header* header, PSZ_BUF* mem, u1* in, T* out, psz_stream_t stream)
{
  auto access = [&](int FIELD, szt offset_nbyte = 0) {
    return (void*)(in + header->entry[FIELD] + offset_nbyte);
  };

  auto d_anchor = (T*)access(PSZ_ANCHOR);
  auto d_spval_idx = (_portable::compact_cell<T, M>*)access(PSZ_SPFMT);
  auto d_spval = (T*)access(PSZ_SPFMT);
  auto d_spidx = (M*)access(PSZ_SPFMT, header->splen * sizeof(T));
  auto d_space = out, d_xdata = out;  // aliases
  auto len = header->len;
  phf_header h;  // declared early so goto over STEP_DECODING is valid

  const auto eb = header->rc.eb, eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

#ifdef PSZ_USE_LC_FIXED
  if (header->pipeline.codec1 == LC and header->pipeline.codec2 != LC) {
    // TCMS-only: ectrl is TCMS-compressed, anchor/spfmt are raw in archive
    void* decomp_lc1 = nullptr;
    float time_lc1 = 0;
    TCMS_DECOMPRESS((uint8_t*)access(PSZ_ENCODED), &decomp_lc1, &time_lc1);
    cudaMemcpyAsync(
        mem->ectrl_d(), decomp_lc1, len.x * len.y * len.z * sizeof(E), cudaMemcpyDeviceToDevice,
        (cudaStream_t)stream);
    // d_anchor, d_spval, d_spidx already initialized to access(PSZ_ANCHOR/PSZ_SPFMT)
    if (header->pipeline.predictor == Spline) memset_device(d_space, len.x * len.y * len.z);
    if (header->splen != 0)
      psz::module::GPU_scatter<T, M>::kernel(
          d_spval, d_spidx, header->splen, d_space, nullptr, stream);
    goto STEP_PREDICT;
  }
  if (header->pipeline.codec2 == LC) {
    void* decomp_lc1 = nullptr;
    void* decomp_lc2 = nullptr;
    float time_lc1 = 0, time_lc2 = 0;

    if (header->pipeline.codec1 == Huffman) {
      // HiCR: RTR_DECOMPRESS over [ENCODED][ANCHOR][SPFMT]
      RTR_DECOMPRESS((uint8_t*)access(PSZ_ENCODED), &decomp_lc1, &time_lc1);
      // after decompress: decomp_lc1 = [HF][ANCHOR][SPFMT]
      d_anchor =
          (T*)((byte_t*)decomp_lc1 + (header->entry[PSZ_ANCHOR] - header->entry[PSZ_ENCODED]));
      d_spval =
          (T*)((byte_t*)decomp_lc1 + (header->entry[PSZ_SPFMT] - header->entry[PSZ_ENCODED]));
      d_spidx = (M*)(d_spval + header->splen);
      // HF decode from start of decompressed block
      memcpy_allkinds<D2H>((BYTE*)&h, (BYTE*)decomp_lc1, sizeof(phf_header));
      // scatter first (ectrl not yet needed), decode after
      if (header->pipeline.predictor == Spline) memset_device(d_space, len.x * len.y * len.z);
      if (header->splen != 0)
        psz::module::GPU_scatter<T, M>::kernel(
            d_spval, d_spidx, header->splen, d_space, nullptr, stream);
      phf::high_level<E>::decode(mem->buf_hf(), h, (BYTE*)decomp_lc1, mem->ectrl_d(), stream);
    }
    else {
      // HiTP: TCMS_DECOMPRESS ectrl + BITR_DECOMPRESS [ANCHOR][SPFMT]
      TCMS_DECOMPRESS((uint8_t*)access(PSZ_ENCODED), &decomp_lc1, &time_lc1);
      cudaMemcpyAsync(
          mem->ectrl_d(), decomp_lc1, len.x * len.y * len.z * sizeof(E), cudaMemcpyDeviceToDevice,
          (cudaStream_t)stream);
      BITR_DECOMPRESS((uint8_t*)access(PSZ_ANCHOR), &decomp_lc2, &time_lc2);
      d_anchor = (T*)decomp_lc2;
      d_spval = (T*)((byte_t*)decomp_lc2 + (header->entry[PSZ_SPFMT] - header->entry[PSZ_ANCHOR]));
      d_spidx = (M*)(d_spval + header->splen);
      if (header->pipeline.predictor == Spline) memset_device(d_space, len.x * len.y * len.z);
      if (header->splen != 0)
        psz::module::GPU_scatter<T, M>::kernel(
            d_spval, d_spidx, header->splen, d_space, nullptr, stream);
      // ectrl already placed in mem->ectrl_d() above
    }

    goto STEP_PREDICT;
  }
#endif

STEP_SCATTER:

  if (header->pipeline.predictor == Spline) memset_device(d_space, len.x * len.y * len.z);

  if (header->splen != 0) {
    if (header->pipeline.predictor == Spline)
      psz::module::GPU_scatter<T, M>::kernel(
          d_spval, d_spidx, header->splen, d_space, nullptr, stream);
    else
      psz::module::GPU_scatter<T, M>::kernel_v2(d_spval_idx, header->splen, d_space, stream);
  }

STEP_DECODING:

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
      psz::module::GPU_spline_reconstruct<T, E>::kernel_v1(
          d_anchor, mem->anchor_len3(), mem->ectrl_d(), d_xdata, mem->ectrl_len3(), d_space, eb,
          header->rc.radius, header->intp_param, stream);

  return PSZ_SUCCESS;
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
