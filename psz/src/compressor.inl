#include <iostream>
#include <string>

#include "compressor.hh"
#include "kernel.hh"
#include "lc_gen/lc_gen.h"
#include "phf.hh"
#include "ptb.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

using _ptb::make_view;
using Toggle = psz::Toggle;

using psz::PredictorFeature;
using psz::PredictorTyping;
using psz::module::GPU_c_lorenzo_nd;
using psz::module::GPU_c_spline_y24;
using psz::module::GPU_c_spline_y25;
using psz::module::GPU_x_lorenzo_nd;
using psz::module::GPU_x_spline_y24;
using psz::module::GPU_x_spline_y25;

template <typename T, Toggle ZigZag, Toggle H1L = Toggle::H1L_Off, Toggle H1G = Toggle::H1G_Off>
using pred_lrz_c = GPU_c_lorenzo_nd<PredictorTyping<T>, PredictorFeature<ZigZag>>;

template <typename T, Toggle ZigZag, Toggle H1L = Toggle::H1L_Off, Toggle H1G = Toggle::H1G_Off>
using pred_lrz_x = GPU_x_lorenzo_nd<PredictorTyping<T>, PredictorFeature<ZigZag>>;

template <typename T, typename E>
using spl_c_y24 = GPU_c_spline_y24<PredictorTyping<T, E>>;
template <typename T, typename E>
using spl_c_y25 = GPU_c_spline_y25<PredictorTyping<T, E>>;
template <typename T, typename E>
using spl_x_y24 = GPU_x_spline_y24<PredictorTyping<T, E>>;
template <typename T, typename E>
using spl_x_y25 = GPU_x_spline_y25<PredictorTyping<T, E>>;

#define c_lrz pred_lrz_c<T, Toggle::ZigZag_Off>::kernel
#define x_lrz pred_lrz_x<T, Toggle::ZigZag_Off>::kernel
#define x_lrz_zz pred_lrz_x<T, Toggle::ZigZag_On>::kernel
#define c_lrz_zz pred_lrz_c<T, Toggle::ZigZag_On>::kernel

#if defined(PSZ_USE_CUDA)

#define CONCAT_ON_DEVICE(dst, src, nbyte, stream) \
  if (nbyte != 0) cudaMemcpyAsync(dst, src, nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);

#elif defined(PSZ_USE_1API)

#define CONCAT_ON_DEVICE(dst, src, nbyte, stream) \
  if (nbyte != 0) ((sycl::queue*)stream)->memcpy(dst, src, nbyte);

#endif

#define DST(FIELD, OFFSET) ((void*)(mem->compressed_d() + ctx->header->entry[FIELD] + OFFSET))

#define PIPELINE ctx->header->pipeline
#define RC ctx->header->rc

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
  const auto _c1 = ctx->header->pipeline.codec1;
  const auto use_HFR = (_c1 == psz_codec::HFR) or (_c1 == psz_codec::HFR_PBKC) or
                       (_c1 == psz_codec::HFR_PBKGO) or (_c1 == psz_codec::HFR_V3);
  auto mem = new Buf_Comp<T, E>(ctx->header->len, iscompression, use_HFR);
  mem->register_header(ctx->header);
  mem->set_spline_variant(ctx->spline_variant);       // anchor sizing
  ctx->header->spline_variant = ctx->spline_variant;  // decompress dispatch

  // optimize component(s)
  psz::module::GPU_histogram_generic<E>::init(
      mem->len_linear, mem->max_bklen, mem->hist_generic_grid_dim, mem->hist_generic_block_dim,
      mem->hist_generic_shmem_use, mem->hist_generic_repeat);

  return mem;
}

PPL_IMPL(void*)::decompress_init(psz_header* header)
{
  // initialize internal buffers
  const auto _c1 = header->pipeline.codec1;
  const auto use_HFR = (_c1 == psz_codec::HFR) or (_c1 == psz_codec::HFR_PBKC) or
                       (_c1 == psz_codec::HFR_PBKGO) or (_c1 == psz_codec::HFR_V3);
  auto mem = new Buf_Comp<T, E>(header->len, false, use_HFR);
  mem->register_header(header);
  return mem;
}

PPL_IMPL(int)::comp_predict(psz_ctx* ctx, PSZ_BUF* mem, T* in, void* stream)
{
  const auto eb = RC.eb;
  const auto len = ctx->header->len;
  const auto radius = RC.radius;
  const auto predictor = PIPELINE.predictor;

  if (predictor == Lorenzo)
    c_lrz(mem, make_view(in, len), eb, radius, stream);
  else if (predictor == LorenzoZigZag)
    c_lrz_zz(mem, make_view(in, len), eb, radius, stream);
  else if (predictor == Spline) {
    if constexpr (std::is_same_v<T, f4>) {
      if (ctx->spline_variant == 1)
        spl_c_y24<T, E>::kernel(
            mem, make_view(in, len), eb, ctx->header->user_input_eb, ctx->header->rc.radius,
            ctx->header->intp_param, stream);
      else
        spl_c_y25<T, E>::kernel(
            mem, make_view(in, len), eb, ctx->header->user_input_eb, ctx->header->rc.radius,
            ctx->header->intp_param, stream);
    }
  }
  else
    return PSZ_ABORT_NO_SUCH_PREDICTOR;

  return PSZ_SUCCESS;
}

PPL_IMPL(int)::compress_analysis(psz_ctx* ctx, PSZ_BUF* mem, T* in, u4* h_hist, void* stream)
{
  const auto len_linear = mem->len_linear;

  if (auto stat = comp_predict(ctx, mem, in, stream); stat != PSZ_SUCCESS) return stat;

  /* make outlier count seen on host */
  sync_by_stream(stream);
  ctx->header->splen = mem->outlier2_host_get_num();

  psz::module::GPU_histogram_Cauchy<E>::kernel(
      mem->eq_d(), len_linear, mem->hist_d(), ctx->bklen, stream);

  memcpy_allkinds_async<D2H>(h_hist, mem->hist_d(), ctx->bklen, stream);
  sync_by_stream(stream);

  memset_device(mem->hist_d(), ctx->bklen, 0);

  return PSZ_SUCCESS;
}

PPL_IMPL(int)::compress(psz_ctx* ctx, PSZ_BUF* mem, T* in, u1** out, size_t* outlen, void* stream)
{
  const auto len_linear = mem->len_linear;
  const auto predictor = PIPELINE.predictor;
  // HFR family reduce-merge pass count from CLI (--rmerge-count); 3 for API callers (no cli).
  const HFR_Opts hfr_opts{ctx->cli ? ctx->cli->hfr_rmerge_count : 3};

  auto compress_predict = [&]() -> int {
    if (auto stat = comp_predict(ctx, mem, in, stream); stat != PSZ_SUCCESS) return stat;

    // HFR family defers outlier read until after pass1's own sync.
    const auto defer_outlier_read = (PIPELINE.codec1 == HFR) or (PIPELINE.codec1 == HFR_PBKC) or
                                    (PIPELINE.codec1 == HFR_PBKGO) or (PIPELINE.codec1 == HFR_V3);
    if (not defer_outlier_read) {
      /* make outlier count seen on host */
      sync_by_stream(stream);
      ctx->header->splen = mem->outlier2_host_get_num();
      if (ctx->header->splen == mem->buf_outlier2()->max_allowed_num()) {
        cerr << "[psz::warning::pipeline] max allowed num-outlier (" << mem->outlier_ratio()
             << " * input-len) exceeded, returning..." << endl;
        return PSZ_WARN_OUTLIER_TOO_MANY;
      }
    }

    return PSZ_SUCCESS;
  };

  // device histogram into mem->hist_d() (shared for HF, HFR, HFR-v3).
  auto compress_histogram = [&]() {
    memset_device(mem->hist_d(), ctx->bklen, 0);

    if (PIPELINE.hist == psz_hist::HistSp)
      psz::module::GPU_histogram_Cauchy<E>::kernel(
          mem->eq_d(), len_linear, mem->hist_d(), ctx->bklen, stream);
    else if (PIPELINE.hist == psz_hist::HistGeneric)
      psz::module::GPU_histogram_generic<E>::kernel(
          mem->eq_d(), len_linear, mem->hist_d(), ctx->bklen, mem->hist_generic_grid_dim,
          mem->hist_generic_block_dim, mem->hist_generic_shmem_use, mem->hist_generic_repeat,
          stream);
  };

  // shared for HF and HFR
  auto compress_histogram_and_build_book = [&]() {
    compress_histogram();
    memcpy_allkinds<D2H>(mem->hist_h(), mem->hist_d(), ctx->bklen);
    phf::high_level<E>::HF_build_book(mem->buf_hf(), mem->hist_h(), ctx->bklen, stream);
  };

  auto compress_encode_pass1_Huffman = [&]() -> int {
    compress_histogram_and_build_book();

    phf_header dummy_header;
    phf::high_level<E>::HF_encode(
        mem->buf_hf(), mem->eq_d(), len_linear, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HF);
    ctx->header->vle_sublen = dummy_header.sublen;
    ctx->header->vle_pardeg = dummy_header.pardeg;
    sync_by_stream(stream);
    return PSZ_SUCCESS;
  };

  // HFr1: ph1+ph2 same as HF; LAGO-concat in place of ph3+ph4.
  auto compress_encode_pass1_Huffman_rev1 = [&]() -> int {
    compress_histogram_and_build_book();

    phf_header dummy_header;
    phf::high_level<E>::HF_encode(
        mem->buf_hf(), mem->eq_d(), len_linear, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HFr1);
    ctx->header->vle_sublen = dummy_header.sublen;
    ctx->header->vle_pardeg = dummy_header.pardeg;
    sync_by_stream(stream);
    // Post-encode scan-state reset (LAGO pay-forward).
    mem->buf_hf()->reset(stream);
    return PSZ_SUCCESS;
  };

  // HFr2: same as _r1 but ships per-block metadata as AoS bheader_backport[].
  auto compress_encode_pass1_Huffman_rev2 = [&]() -> int {
    compress_histogram_and_build_book();

    phf_header dummy_header;
    phf::high_level<E>::HF_encode(
        mem->buf_hf(), mem->eq_d(), len_linear, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HFr2);
    ctx->header->vle_sublen = dummy_header.sublen;
    ctx->header->vle_pardeg = dummy_header.pardeg;
    sync_by_stream(stream);
    // Post-encode scan-state reset (LAGO pay-forward).
    mem->buf_hf()->reset(stream);
    return PSZ_SUCCESS;
  };

  // HFR reference (HFReVISIT base): shuffle-merge encode + sparse breaks.
  auto compress_encode_pass1_HFR = [&]() -> int {
    // low-rmerge preset (same as v3): higher RT -> more merge-breaks -> lower CR.
    const HFR_Opts v2_opts{/*reduce_times=*/1};

    compress_histogram_and_build_book();

    phf_header dummy_header;
    phf::high_level<E>::HFR_encode(
        mem->buf_hf(), mem->eq_d(), len_linear, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HFR, nullptr, nullptr, v2_opts);
    ctx->header->vle_sublen = dummy_header.sublen;
    ctx->header->vle_pardeg = dummy_header.pardeg;
    sync_by_stream(stream);
    // Post-encode scan-state reset; for future multistream coordination.
    mem->buf_hf()->reset_HFR(stream);
    return PSZ_SUCCESS;
  };

  // HFR-PBKC; also as alt default codec1
  // low-rmerge preset: RT=1 (2 pts/thread); 0 = zero-merge (regressed speed).
  auto compress_encode_pass1_HFR_PBK_Compat = [&]() -> int {
    const HFR_Opts pbkc_opts{/*reduce_times=*/1};
    phf_header dummy_header;
    phf::high_level<E>::HFR_encode(
        mem->buf_hf(), mem->eq_d(), len_linear, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HFR_PBKC, nullptr, nullptr, pbkc_opts);
    ctx->header->vle_sublen = dummy_header.sublen;
    ctx->header->vle_pardeg = dummy_header.pardeg;
    sync_by_stream(stream);
    // Post-encode scan-state reset; for future multistream coordination.
    mem->buf_hf()->reset_HFR(stream);
    return PSZ_SUCCESS;
  };

  // HFR-PBKGO: globally-ordered slot; no LAGO concat
  auto compress_encode_pass1_HFR_PBK_GO = [&]() -> int {
    const HFR_Opts hfr_opts{/*reduce_times=*/1};
    phf_header dummy_header;
    phf::high_level<E>::HFR_encode(
        mem->buf_hf(), mem->eq_d(), len_linear, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HFR_PBKGO, nullptr, nullptr, hfr_opts);
    ctx->header->vle_sublen = dummy_header.sublen;
    ctx->header->vle_pardeg = dummy_header.pardeg;
    sync_by_stream(stream);
    mem->buf_hf()->reset_HFR(stream);
    return PSZ_SUCCESS;
  };

  // HFR-v3: GPU-picked global PBK book + low-rmerge preset (scalable default).
  auto compress_encode_pass1_HFR_v3 = [&]() -> int {
    // low-rmerge preset: 1 = 2 points/thread (balanced); 0 = zero-merge (cruel).
    const HFR_Opts v3_opts{/*reduce_times=*/1};

    compress_histogram();
    phf::high_level<E>::HFR_pick_pbk(mem->buf_hf(), mem->hist_d(), ctx->bklen, len_linear, stream);

    phf_header dummy_header;
    phf::high_level<E>::HFR_encode(
        mem->buf_hf(), mem->eq_d(), len_linear, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HFR_V3, nullptr, nullptr, v3_opts);
    ctx->header->vle_sublen = dummy_header.sublen;
    ctx->header->vle_pardeg = dummy_header.pardeg;
    sync_by_stream(stream);
    mem->buf_hf()->reset_HFR(stream);
    return PSZ_SUCCESS;
  };

  auto compress_encode_pass1_wrapup = [&]() {
    memset(mem->nbyte, 0, sizeof(mem->nbyte));
    mem->nbyte[PSZ_HEADER] = sizeof(psz_header);
    mem->nbyte[PSZ_ENCODED] = sizeof(u1) * mem->comp_codec_outlen;
    mem->nbyte[PSZ_ANCHOR] = predictor == Spline ? sizeof(T) * mem->anchor_len() : 0;
    mem->nbyte[PSZ_SPFMT] = sizeof(_ptb::compact_cell<T, u4>) * ctx->header->splen;
    mem->nbyte[PSZ_ENC_PASS1_END] = 0;

    // clang-format off
  ctx->header->entry[0] = 0;
  // *.END + 1; need to know the ending position
  for (auto i = 1; i < PSZ_ENC_PASS2_END + 1; i++) ctx->header->entry[i] = mem->nbyte[i - 1];
  for (auto i = 1; i < PSZ_ENC_PASS2_END + 1; i++) ctx->header->entry[i] += ctx->header->entry[i - 1];

  CONCAT_ON_DEVICE(DST(PSZ_ANCHOR, 0), mem->anchor_d(), mem->nbyte[PSZ_ANCHOR], stream);
  CONCAT_ON_DEVICE(DST(PSZ_ENCODED, 0), mem->comp_codec_out, mem->nbyte[PSZ_ENCODED], stream);
  CONCAT_ON_DEVICE(DST(PSZ_SPFMT, 0), mem->outlier2_validx_d(), mem->nbyte[PSZ_SPFMT], stream);
    // clang-format on

    /* output of this function */
    *out = mem->compressed_d();
    *outlen = pszheader_filesize(ctx->header);
  };

  auto compress_encode_pass1_LC_TCMS = [&]() -> int {
#ifdef PSZ_USE_LC_FIXED
    // Hi-TP mode: TCMS replaces histogram+HF; mark hist as null in header
    ctx->header->pipeline.hist = psz_hist::HistNull;
    lc_c::TCMS_COMPRESS(
        (uint8_t*)mem->eq_d(), len_linear * sizeof(E), mem->buf_lc(), &mem->comp_codec_outlen,
        stream);
    mem->comp_codec_out = mem->buf_lc()->encoded_d();
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
    size_t comp_rtr_outlen;

    lc_c::RTR_COMPRESS(
        (uint8_t*)DST(PSZ_ENCODED, 0),
        mem->nbyte[PSZ_ENCODED] + mem->nbyte[PSZ_ANCHOR] + mem->nbyte[PSZ_SPFMT], mem->buf_lc(),
        &comp_rtr_outlen, stream);

    // reuse PSZ_ENCODED buf
    cudaMemcpyAsync(
        DST(PSZ_ENCODED, 0), (void*)mem->buf_lc()->encoded_d(), comp_rtr_outlen,
        cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
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
    size_t comp_bitr_outlen;

    lc_c::BITR_COMPRESS(
        (uint8_t*)DST(PSZ_ANCHOR, 0), mem->nbyte[PSZ_ANCHOR] + mem->nbyte[PSZ_SPFMT],
        mem->buf_lc(), &comp_bitr_outlen, stream);
    cudaMemcpyAsync(
        DST(PSZ_ANCHOR, 0), (void*)mem->buf_lc()->encoded_d(), comp_bitr_outlen,
        cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
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
    int status;
    if (PIPELINE.codec1 == HFR)
      status = compress_encode_pass1_HFR();
    else if (PIPELINE.codec1 == HFR_PBKC)
      status = compress_encode_pass1_HFR_PBK_Compat();
    else if (PIPELINE.codec1 == HFR_PBKGO)
      status = compress_encode_pass1_HFR_PBK_GO();
    else if (PIPELINE.codec1 == HFR_V3)
      status = compress_encode_pass1_HFR_v3();
    else if (PIPELINE.codec1 == HFr1)
      status = compress_encode_pass1_Huffman_rev1();
    else if (PIPELINE.codec1 == HFr2)
      status = compress_encode_pass1_Huffman_rev2();
    else
      status = compress_encode_pass1_Huffman();
    if (status != PSZ_SUCCESS) return status;

    // Deferred outlier-count read (pass1 already synced for HFR family).
    if ((PIPELINE.codec1 == HFR) or (PIPELINE.codec1 == HFR_PBKC) or
        (PIPELINE.codec1 == HFR_PBKGO) or (PIPELINE.codec1 == HFR_V3)) {
      ctx->header->splen = mem->outlier2_host_get_num();
      if (ctx->header->splen == mem->buf_outlier2()->max_allowed_num()) {
        cerr << "[psz::warning::pipeline] max allowed num-outlier (" << mem->outlier_ratio()
             << " * input-len) exceeded, returning..." << endl;
        return PSZ_WARN_OUTLIER_TOO_MANY;
      }
    }

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
  // HiTP eq-only: TCMS for eq, raw anchor+spfmt (no BITR, fallback)
  auto compress_encode_HiTP_eq = [&]() -> int {
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
                       : (PIPELINE.codec1 == LC) ? compress_encode_HiTP_eq()
                                                 : compress_encode_default();
#else
  // In the case of LC being not compiled --> fallback
  if (PIPELINE.codec1 == LC or PIPELINE.codec2 == LC) {
    ctx->header->pipeline.codec1 = DEFAULT_CODEC;
    ctx->header->pipeline.codec2 = CodecNull;
  }
  auto status_encode = compress_encode_default();
#endif
  if (status_encode != PSZ_SUCCESS) return status_encode;

  return PSZ_SUCCESS;
}

PPL_IMPL(void)::decomp_scatter(
    psz_header* header, _ptb::compact_cell<T, M>* d_spval_idx, T* d_space, void* stream)
{
  const auto len = header->len;
  if (header->pipeline.predictor == Spline) memset_device(d_space, len.x * len.y * len.z);
  if (header->splen != 0)
    psz::module::GPU_scatter<T, M>::kernel_v2(d_spval_idx, header->splen, d_space, stream);
}

PPL_IMPL(void)::decomp_predict(
    psz_header* header, PSZ_BUF* mem, T* d_anchor, T* d_xdata, void* stream)
{
  const auto eb = header->rc.eb;
  const auto len = header->len;

  if (header->pipeline.predictor == Lorenzo)
    x_lrz(mem, d_xdata, eb, header->rc.radius, stream);
  else if (header->pipeline.predictor == LorenzoZigZag)
    x_lrz_zz(mem, d_xdata, eb, header->rc.radius, stream);
  else if (header->pipeline.predictor == Spline) {
    mem->set_spline_variant(header->spline_variant);  // anchor sizing
    if constexpr (std::is_same_v<T, f4>) {
      if (header->spline_variant == 1)
        spl_x_y24<T, E>::kernel(
            mem, d_anchor, make_view(d_xdata, len), eb, header->rc.radius, header->intp_param,
            stream);
      else
        spl_x_y25<T, E>::kernel(
            mem, d_anchor, make_view(d_xdata, len), eb, header->rc.radius, header->intp_param,
            stream);
    }
  }
}

PPL_IMPL(int)::decompress(psz_header* header, PSZ_BUF* mem, u1* in, T* out, psz_stream_t stream)
{
  auto access = [&](int FIELD, szt offset_nbyte = 0) {
    return (void*)(in + header->entry[FIELD] + offset_nbyte);
  };

  auto d_anchor = (T*)access(PSZ_ANCHOR);
  auto d_spval_idx = (_ptb::compact_cell<T, M>*)access(PSZ_SPFMT);
  auto d_space = out, d_xdata = out;  // aliases
  auto len = header->len;
  phf_header h;  // declared early so goto over STEP_DECODING is valid

#ifdef PSZ_USE_LC_FIXED
  if (header->pipeline.codec1 == LC and header->pipeline.codec2 != LC) {
    // TCMS-only: eq is TCMS-compressed, anchor/spfmt are raw in archive
    lc_c::TCMS_DECOMPRESS((uint8_t*)access(PSZ_ENCODED), mem->buf_lc(), stream);
    cudaMemcpyAsync(
        mem->eq_d(), mem->buf_lc()->decoded_d(), len.x * len.y * len.z * sizeof(E),
        cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    // d_anchor and d_spval_idx already initialized to access(PSZ_ANCHOR/PSZ_SPFMT)
    decomp_scatter(header, d_spval_idx, d_space, stream);
    goto STEP_PREDICT;
  }
  if (header->pipeline.codec2 == LC) {
    if (header->pipeline.codec1 == HF) {
      // HiCR: RTR_DECOMPRESS over [ENCODED][ANCHOR][SPFMT]
      lc_c::RTR_DECOMPRESS((uint8_t*)access(PSZ_ENCODED), mem->buf_lc(), stream);
      auto decomp_lc1 = mem->buf_lc()->decoded_d();
      // after decompress: decomp_lc1 = [HF][ANCHOR][SPFMT]
      d_anchor =
          (T*)((byte_t*)decomp_lc1 + (header->entry[PSZ_ANCHOR] - header->entry[PSZ_ENCODED]));
      d_spval_idx =
          (_ptb::compact_cell<T, M>*)((byte_t*)decomp_lc1 +
                                      (header->entry[PSZ_SPFMT] - header->entry[PSZ_ENCODED]));
      // HF decode from start of decompressed block
      memcpy_allkinds<D2H>((BYTE*)&h, (BYTE*)decomp_lc1, sizeof(phf_header));
      // scatter first (eq not yet needed), decode after
      decomp_scatter(header, d_spval_idx, d_space, stream);
      phf::high_level<E>::HF_decode(
          mem->buf_hf(), h, (BYTE*)decomp_lc1, mem->eq_d(), stream, psz_codec::HF);
    }
    else {
      // HiTP: TCMS_DECOMPRESS eq + BITR_DECOMPRESS [ANCHOR][SPFMT]
      lc_c::TCMS_DECOMPRESS((uint8_t*)access(PSZ_ENCODED), mem->buf_lc(), stream);
      cudaMemcpyAsync(
          mem->eq_d(), mem->buf_lc()->decoded_d(), len.x * len.y * len.z * sizeof(E),
          cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
      lc_c::BITR_DECOMPRESS((uint8_t*)access(PSZ_ANCHOR), mem->buf_lc(), stream);
      auto decomp_lc2 = mem->buf_lc()->decoded_d();
      d_anchor = (T*)decomp_lc2;
      d_spval_idx = (_ptb::compact_cell<T, M>*)((byte_t*)decomp_lc2 + (header->entry[PSZ_SPFMT] -
                                                                       header->entry[PSZ_ANCHOR]));
      decomp_scatter(header, d_spval_idx, d_space, stream);
      // eq already placed in mem->eq_d() above
    }

    goto STEP_PREDICT;
  }
#endif

STEP_SCATTER:

  decomp_scatter(header, d_spval_idx, d_space, stream);

STEP_DECODING:

  memcpy_allkinds<D2H>((BYTE*)&h, (BYTE*)access(PSZ_ENCODED), sizeof(phf_header));
  if (header->pipeline.codec1 == HFR_PBKC)
    phf::high_level<E>::HFR_decode(
        mem->buf_hf(), h, (BYTE*)access(PSZ_ENCODED), mem->eq_d(), stream, psz_codec::HFR_PBKC);
  else if (header->pipeline.codec1 == HFR_PBKGO)
    phf::high_level<E>::HFR_decode(
        mem->buf_hf(), h, (BYTE*)access(PSZ_ENCODED), mem->eq_d(), stream, psz_codec::HFR_PBKGO);
  else if (header->pipeline.codec1 == HFR)
    phf::high_level<E>::HFR_decode(
        mem->buf_hf(), h, (BYTE*)access(PSZ_ENCODED), mem->eq_d(), stream, psz_codec::HFR);
  else if (header->pipeline.codec1 == HFR_V3)
    phf::high_level<E>::HFR_decode(
        mem->buf_hf(), h, (BYTE*)access(PSZ_ENCODED), mem->eq_d(), stream, psz_codec::HFR_V3);
  else if (header->pipeline.codec1 == HFr2)
    phf::high_level<E>::HF_decode(
        mem->buf_hf(), h, (BYTE*)access(PSZ_ENCODED), mem->eq_d(), stream, psz_codec::HFr2);
  else
    // HF + HFr1 share the same on-disk layout / decoder.
    phf::high_level<E>::HF_decode(
        mem->buf_hf(), h, (BYTE*)access(PSZ_ENCODED), mem->eq_d(), stream, psz_codec::HF);

STEP_PREDICT:

  decomp_predict(header, mem, d_anchor, d_xdata, stream);

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
    _ptb::utils::tofile(dump_name("u4", "ht"), mem->hist_h(), RC.radius * 2);
  }
  if (ctx->cli->dump_quantcode) {
    cout << "[psz::dump] dumping quantization codebook to file: " << dump_name("quant") << endl;
    auto h_eq = MAKE_UNIQUE_HOST(E, mem->len_linear);
    memcpy_allkinds<D2H>(h_eq.get(), mem->eq_d(), mem->len_linear, stream);
    _ptb::utils::tofile(dump_name("u" + to_string(sizeof(E)), "qt"), h_eq.get(), mem->len_linear);
  }
}

#undef PPL_IMPL
#undef PIPELINE
#undef RC
