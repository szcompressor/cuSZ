#include "context_impl.h"
#include "pipeline.h"
#include <iostream>
#include <string>

#include "compressor.hh"
#include "compressor2.inl"
#include "fzg_hl.hh"
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

using psz::PredictorFeature;
using psz::PredictorTyping;
using psz::module::GPU_c_lorenzo_nd;
using psz::module::GPU_c_spline_y24;
using psz::module::GPU_c_spline_y25;
using psz::module::GPU_x_lorenzo_nd;
using psz::module::GPU_x_spline_y24;
using psz::module::GPU_x_spline_y25;

template <typename T, typename E, int ZigZag>
using pred_lrz_c = GPU_c_lorenzo_nd<PredictorTyping<T, E>, PredictorFeature<ZigZag>>;

template <typename T, typename E, int ZigZag>
using pred_lrz_x = GPU_x_lorenzo_nd<PredictorTyping<T, E>, PredictorFeature<ZigZag>>;

template <typename T, typename E>
using spl_c_y24 = GPU_c_spline_y24<PredictorTyping<T, E>, PredictorFeature<0b0>>;
template <typename T, typename E>
using spl_c_y25 = GPU_c_spline_y25<PredictorTyping<T, E>, PredictorFeature<0b0>>;
template <typename T, typename E>
using spl_x_y24 = GPU_x_spline_y24<PredictorTyping<T, E>>;
template <typename T, typename E>
using spl_x_y25 = GPU_x_spline_y25<PredictorTyping<T, E>>;

#define c_lrz pred_lrz_c<T, E, 0b0>::kernel
#define x_lrz pred_lrz_x<T, E, 0b0>::kernel
#define x_lrz_zz pred_lrz_x<T, E, 0b1>::kernel
#define c_lrz_zz pred_lrz_c<T, E, 0b1>::kernel

#if defined(PSZ_USE_CUDA)

#define CONCAT_ON_DEVICE(dst, src, nbyte, stream) \
  if (nbyte != 0) cudaMemcpyAsync(dst, src, nbyte, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);

#elif defined(PSZ_USE_1API)

#define CONCAT_ON_DEVICE(dst, src, nbyte, stream) \
  if (nbyte != 0) ((sycl::queue*)stream)->memcpy(dst, src, nbyte);

#endif

#define DST(FIELD, OFFSET) ((void*)(mem->compressed_d() + ctx->header->entry[FIELD] + OFFSET))

#define PIPELINE ctx->header->pipeline

#define PPL_IMPL(RET_TYPE)                     \
  template <typename T, typename E, class PPL> \
  RET_TYPE psz::compression_pipeline<T, E, PPL>


PPL_IMPL(void*)::compress_init(psz_ctx* ctx, bool skip_hf)
{
  constexpr auto iscompression = true;

  // extract context
  const auto x = ctx->header->len.x, y = ctx->header->len.y, z = ctx->header->len.z;

  // init internal buffers
  const auto _c1 = ctx->header->pipeline.codec1;
  // codec2==LC precedes HF_rev2: Buf_HF takes the generic chunk size
  const auto use_HFR =
      ((_c1 == psz_codec::HFR) or (_c1 == psz_codec::HFR_PBKC) or (_c1 == psz_codec::HFR_PBKGO) or
       (_c1 == psz_codec::HFR_V3) or (_c1 == psz_codec::HFR_V4));
  const auto use_FZG = (_c1 == psz_codec::FZG);
  auto const _pred = ctx->header->pipeline.predictor;
  int const _nd = (z > 1) ? 3 : (y > 1) ? 2 : 1;
  bool const y25_tile = _pred == psz_predictor::SplineY25 and _nd >= 2;
  bool const tile_order =
      (_nd >= 2 and (_pred == psz_predictor::Lorenzo or _pred == psz_predictor::LorenzoZigZag or
                     _pred == psz_predictor::SplineY24)) or
      y25_tile;
  Buf_Comp<T, E>* mem;
  if (skip_hf) {
    BufToggle_Comp toggle{
        /*use_quant=*/true, /*use_outlier=*/true,     /*use_anchor=*/true,
        /*use_hist=*/true,  /*use_compressed=*/false, /*use_top1=*/true,
        /*use_lc=*/true};
    mem = new Buf_Comp<T, E>(ctx->header->len, &toggle);
  }
  else {
    mem = new Buf_Comp<T, E>(
        ctx->header->len, iscompression, use_HFR, true, _c1 == psz_codec::HF_r2, tile_order,
        y25_tile, use_FZG, _c1, ctx->header->pipeline.codec2);
  }
  mem->register_header(ctx->header);
  mem->set_predictor(_pred);  // anchor sizing

  // optimize component(s)
  psz::module::GPU_histogram_generic<E>::init(
      mem->len_linear, mem->max_bklen, mem->hist_generic_grid_dim, mem->hist_generic_block_dim,
      mem->hist_generic_shmem_use, mem->hist_generic_repeat);

  return mem;
}

PPL_IMPL(void*)::decompress_init(psz_header* header)
{
  const auto _c1 = header->pipeline.codec1;
  const auto use_HFR =
      ((_c1 == psz_codec::HFR) or (_c1 == psz_codec::HFR_PBKC) or (_c1 == psz_codec::HFR_PBKGO) or
       (_c1 == psz_codec::HFR_V3) or (_c1 == psz_codec::HFR_V4));
  // Spl-y25 decodes into d_eq (by interp level); Lrz and Spl-y24 decode into the output buffer.
  const auto _pred = header->pipeline.predictor;
  const auto alloc_eq = (_pred == psz_predictor::SplineY25);
  auto const _l = header->len;
  int const _nd = (_l.z > 1) ? 3 : (_l.y > 1) ? 2 : 1;
  bool const y25_tile = _pred == psz_predictor::SplineY25 and _nd >= 2;
  bool const tile_order =
      (_nd >= 2 and (_pred == psz_predictor::Lorenzo or _pred == psz_predictor::LorenzoZigZag or
                     _pred == psz_predictor::SplineY24)) or
      y25_tile;
  auto mem = new Buf_Comp<T, E>(
      header->len, false, use_HFR, alloc_eq, _c1 == psz_codec::HF_r2, tile_order, y25_tile,
      _c1 == psz_codec::FZG, _c1, header->pipeline.codec2);
  mem->register_header(header);
  return mem;
}

PPL_IMPL(int)::comp_predict(psz_ctx* ctx, PSZ_BUF* mem, T* in, void* stream, bool force_global)
{
  const auto eb = ctx->header->eb;
  const auto len = ctx->header->len;
  const auto radius = ctx->header->radius;
  const auto predictor = PIPELINE.predictor;
  // unpred-incomp (enc_id=31) in (use_HFR) encoder
  const bool enable_localized =
      (not pszcodec_is_pass2(PIPELINE.codec2)) and
      ((PIPELINE.codec1 == HFR_PBKC) or (PIPELINE.codec1 == HFR_PBKGO) or
       (PIPELINE.codec1 == HFR) or (PIPELINE.codec1 == HFR_V3) or (PIPELINE.codec1 == HFR_V4));
  // HF and HF-rev2, FZG, and LC (TCMS/HiCR/HiTP) codecs use the global compact.
  // TCMS is not compat with HFR for now.
  const bool enable_global = (PIPELINE.codec1 == HF) or (PIPELINE.codec1 == HF_r2) or
                             (PIPELINE.codec1 == FZG) or (PIPELINE.codec1 == LC_TCMS) or
                             (PIPELINE.codec1 == LC_DRH) or pszcodec_is_pass2(PIPELINE.codec2) or
                             force_global;

  if (predictor == Lorenzo)
    c_lrz(mem, make_view(in, len), eb, radius, enable_localized, enable_global, stream);
  else if (predictor == LorenzoZigZag)
    c_lrz_zz(mem, make_view(in, len), eb, radius, enable_localized, enable_global, stream);
  else if (pszpredictor_is_spline(predictor)) {
    if constexpr (std::is_same_v<T, f4>) {
      if (predictor == SplineY24)
        spl_c_y24<T, E>::kernel(
            mem, make_view(in, len), eb, ctx->header->user_input_eb, ctx->header->radius,
            ctx->header->intp_param, enable_global, stream);
      else
        spl_c_y25<T, E>::kernel(
            mem, make_view(in, len), eb, ctx->header->user_input_eb, ctx->header->radius,
            ctx->header->intp_param, enable_global, stream);
    }
  }
  else
    return PSZ_ABORT_NO_SUCH_PREDICTOR;

  return PSZ_SUCCESS;
}

PPL_IMPL(int)::compress_analysis(psz_ctx* ctx, PSZ_BUF* mem, T* in, u4* h_hist, void* stream)
{
  const auto len_linear = mem->len_linear;

  // predictor-only analysis: force the global compact so decomp_scatter restores outliers.
  if (auto stat = comp_predict(ctx, mem, in, stream, /*force_global=*/true); stat != PSZ_SUCCESS)
    return stat;

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
  if (not dispatch<T, E>::pipeline_supported(PIPELINE)) return PSZ_ABORT_UNSUPPORTED_PIPELINE;

  int routed_status;
  if (dispatch<T, E>::route_predictor(PIPELINE, [&](auto ppl) {
        routed_status =
            compression_pipeline<T, E, decltype(ppl)>::compress(ctx, mem, in, out, outlen, stream);
      }))
    return routed_status;

  const auto len_linear = mem->len_linear;
  const auto predictor = PIPELINE.predictor;
  // pass `--rmerge-count` (internal tuning use)
  const HFR_Opts hfr_opts{
      (ctx->cli and ctx->cli->hfr_rmerge_count > 0) ? ctx->cli->hfr_rmerge_count : 3};

  // lrz-1/2D: 1Ki; lrz-3D/spl-y24: 2Ki; spl-y25-3D: 4Ki
  // Codecs subsequently sees paddded eq and several nblock of (1Ki).
  auto const l3 = ctx->header->len;
  int const nd = (l3.z > 1) ? 3 : (l3.y > 1) ? 2 : 1;
  auto const c1 = PIPELINE.codec1;
  bool const is_hfr_family =
      (c1 == HFR or c1 == HFR_PBKC or c1 == HFR_PBKGO or c1 == HFR_V3 or c1 == HFR_V4);
  bool const y25_tile_nd = nd >= 2 and predictor == SplineY25;
  bool const tile_order_nd =
      (y25_tile_nd or (nd >= 2 and (predictor == Lorenzo or predictor == LorenzoZigZag or
                                    predictor == SplineY24)));
  auto const cdiv = [](u4 v, u4 d) -> size_t { return (v + d - 1u) / d; };
  size_t len_eq = len_linear;
  if (tile_order_nd)
    len_eq = y25_tile_nd ? (nd == 3 ? cdiv(l3.x, 16) * cdiv(l3.y, 16) * cdiv(l3.z, 16) * 4096
                                    : cdiv(l3.x, 64) * cdiv(l3.y, 64) * 4096)  // y25 2D = 64x64
             : (nd == 3) ? cdiv(l3.x, 32) * cdiv(l3.y, 8) * cdiv(l3.z, 8) * 2048
                         : cdiv(l3.x, 32) * cdiv(l3.y, 32) * 1024;

  int const magnitude = not tile_order_nd                              ? 10
                        : y25_tile_nd                                  ? 12
                        : (nd == 3 or pszpredictor_is_spline(predictor)) ? 11
                                                                       : 10;

  auto compress_predict = [&]() -> int {
    if (auto stat = comp_predict(ctx, mem, in, stream); stat != PSZ_SUCCESS) return stat;

    const auto defer_outlier_read =
        (not pszcodec_is_pass2(PIPELINE.codec2)) and ((PIPELINE.codec1 == HFR) or (PIPELINE.codec1 == HFR_PBKC) or
                                     (PIPELINE.codec1 == HFR_PBKGO) or
                                     (PIPELINE.codec1 == HFR_V3) or (PIPELINE.codec1 == HFR_V4));
    if (not defer_outlier_read) {
      // HF/HF-rev2 ~ LC path: use global compact
      const bool keep_global = (PIPELINE.codec1 == HF) or (PIPELINE.codec1 == HF_r2) or
                               (PIPELINE.codec1 == LC_TCMS) or (PIPELINE.codec1 == LC_DRH) or
                               pszcodec_is_pass2(PIPELINE.codec2);
      sync_by_stream(stream);
      ctx->header->splen =
          keep_global
              ? std::min<size_t>(mem->outlier2_host_get_num(), mem->outlier2_max_allowed_num())
              : 0;
    }

    return PSZ_SUCCESS;
  };

  // device histogram into mem->hist_d() (shared for HF, HFR, HFR-v3).
  auto compress_histogram = [&]() {
    memset_device(mem->hist_d(), ctx->bklen, 0);

    if (PIPELINE.hist == psz_hist::HistSp)
      psz::module::GPU_histogram_Cauchy<E>::kernel(
          mem->eq_d(), len_eq, mem->hist_d(), ctx->bklen, stream);
    else if (PIPELINE.hist == psz_hist::HistGeneric)
      psz::module::GPU_histogram_generic<E>::kernel(
          mem->eq_d(), len_eq, mem->hist_d(), ctx->bklen, mem->hist_generic_grid_dim,
          mem->hist_generic_block_dim, mem->hist_generic_shmem_use, mem->hist_generic_repeat,
          stream);
  };

  // shared for HF and HFR
  auto compress_histogram_and_build_book = [&]() {
    compress_histogram();
    memcpy_allkinds<D2H>(mem->hist_h(), mem->hist_d(), ctx->bklen);
    phf::high_level<E>::HF_build_book(mem->buf_hf(), mem->hist_h(), ctx->bklen, stream);
  };

  // HF_r2: same as _r1 but ships per-block metadata as AoS bheader_backport[].
  auto compress_encode_pass1_Huffman_rev2 = [&]() -> int {
    compress_histogram_and_build_book();

    phf_header dummy_header{};
    phf::high_level<E>::HF_encode(
        mem->buf_hf(), mem->eq_d(), len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HF_r2);
    sync_by_stream(stream);
    // Post-encode scan-state reset (LAGO pay-forward).
    mem->buf_hf()->reset(stream);
    return PSZ_SUCCESS;
  };

  // HFR reference (HFReVISIT base): shuffle-merge encode + sparse breaks.
  auto compress_encode_pass1_HFR = [&]() -> int {
    // low-rmerge preset (same as v3): higher RT -> more merge-breaks -> lower CR.
    HFR_Opts v2_opts{/*reduce_times=*/1};
    v2_opts.magnitude = magnitude;
    v2_opts.block_outliers = mem->block_outliers_d();

    compress_histogram_and_build_book();

    phf_header dummy_header{};
    phf::high_level<E>::HFR_encode(
        mem->buf_hf(), mem->eq_d(), len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HFR, nullptr, nullptr, v2_opts);
    sync_by_stream(stream);
    // Post-encode scan-state reset; for future multistream coordination.
    mem->buf_hf()->reset_HFR(stream);
    return PSZ_SUCCESS;
  };

  // HFR-PBKC; also as alt default codec1
  // low-rmerge preset: RT=1 (2 pts/thread); 0 = zero-merge (regressed speed).
  auto compress_encode_pass1_HFR_PBK_Compat = [&]() -> int {
    HFR_Opts pbkc_opts{/*reduce_times=*/1};
    pbkc_opts.magnitude = magnitude;
    pbkc_opts.block_outliers = mem->block_outliers_d();
    phf_header dummy_header{};
    phf::high_level<E>::HFR_encode(
        mem->buf_hf(), mem->eq_d(), len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HFR_PBKC, nullptr, nullptr, pbkc_opts);
    sync_by_stream(stream);
    // Post-encode scan-state reset
    mem->buf_hf()->reset_HFR(stream);
    return PSZ_SUCCESS;
  };

  // exclude RT=1
  auto compress_encode_pass1_HFR_PBK_GO = [&]() -> int {
    HFR_Opts hfr_opts{
        (ctx->cli and ctx->cli->hfr_rmerge_count > 0) ? ctx->cli->hfr_rmerge_count : 2};
    hfr_opts.magnitude = magnitude;
    hfr_opts.block_outliers = mem->block_outliers_d();
    phf_header dummy_header{};
    phf::high_level<E>::HFR_encode(
        mem->buf_hf(), mem->eq_d(), len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HFR_PBKGO, nullptr, nullptr, hfr_opts);
    sync_by_stream(stream);
    mem->buf_hf()->reset_HFR(stream);
    return PSZ_SUCCESS;
  };

  // HFR-v3: GPU-picked global PBK book + low-rmerge preset (scalable default).
  auto compress_encode_pass1_HFR_v3 = [&]() -> int {
    // low-rmerge preset: 1 = 2 points/thread (balanced); 0 = zero-merge (cruel).
    HFR_Opts v3_opts{/*reduce_times=*/1};
    v3_opts.magnitude = magnitude;
    v3_opts.block_outliers = mem->block_outliers_d();

    compress_histogram();
    phf::high_level<E>::HFR_pick_pbk(mem->buf_hf(), mem->hist_d(), ctx->bklen, len_eq, stream);

    phf_header dummy_header{};
    phf::high_level<E>::HFR_encode(
        mem->buf_hf(), mem->eq_d(), len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HFR_V3, nullptr, nullptr, v3_opts);
    sync_by_stream(stream);
    mem->buf_hf()->reset_HFR(stream);
    return PSZ_SUCCESS;
  };

  auto compress_encode_pass1_HFR_v4 = [&]() -> int {  // PBKC but single-book
    HFR_Opts v4_opts{/*reduce_times=*/1};
    v4_opts.magnitude = magnitude;
    v4_opts.block_outliers = mem->block_outliers_d();

    compress_histogram();
    phf::high_level<E>::HFR_pick_pbk(mem->buf_hf(), mem->hist_d(), ctx->bklen, len_eq, stream);

    phf_header dummy_header{};
    phf::high_level<E>::HFR_encode(
        mem->buf_hf(), mem->eq_d(), len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen,
        dummy_header, stream, psz_codec::HFR_V4, nullptr, nullptr, v4_opts);
    sync_by_stream(stream);
    mem->buf_hf()->reset_HFR(stream);
    return PSZ_SUCCESS;
  };

  // Zhang, Tian, et al. 2023
  auto compress_encode_pass1_FZG = [&]() -> int {
    // fzg::E fixed at u2
    if constexpr (std::is_same_v<E, u2>) {
      if (predictor != LorenzoZigZag) return PSZ_ABORT_NO_SUCH_PREDICTOR;
      fzg_header dummy_header{};
      auto status = fzg::high_level::encode(
          mem->buf_fzg(), mem->eq_d(), len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen,
          dummy_header, stream);
      sync_by_stream(stream);
      return status == 0 ? PSZ_SUCCESS : PSZ_ABORT_NO_SUCH_CODEC;
    }
    else
      return PSZ_ABORT_NO_SUCH_CODEC;
  };

  auto compress_encode_pass1_wrapup = [&]() -> int {
    memset(mem->nbyte, 0, sizeof(mem->nbyte));
    mem->nbyte[PSZ_HEADER] = sizeof(psz_header);
    mem->nbyte[PSZ_ENCODED] = sizeof(u1) * mem->comp_codec_outlen;
    mem->nbyte[PSZ_ANCHOR] = pszpredictor_is_spline(predictor) ? sizeof(T) * mem->anchor_len() : 0;
    mem->nbyte[PSZ_SPFMT] = sizeof(_ptb::compact_cell<T, u4>) * ctx->header->splen;
    mem->nbyte[PSZ_ENC_PASS1_END] = 0;

    // clang-format off
  ctx->header->entry[0] = 0;
  // *.END + 1; need to know the ending position
  for (auto i = 1; i < PSZ_ENC_PASS2_END + 1; i++) ctx->header->entry[i] = mem->nbyte[i - 1];
  for (auto i = 1; i < PSZ_ENC_PASS2_END + 1; i++) ctx->header->entry[i] += ctx->header->entry[i - 1];

  if (pszheader_filesize(ctx->header) > mem->compressed_max_bytes()) {
    cerr << "[psz::error::pipeline] compressed size (" << pszheader_filesize(ctx->header)
         << " B) exceeds buffer (" << mem->compressed_max_bytes() << " B), returning..." << endl;
    return PSZ_ABORT_COMPRESSED_TOO_LARGE;
  }

  CONCAT_ON_DEVICE(DST(PSZ_ANCHOR, 0), mem->anchor_d(), mem->nbyte[PSZ_ANCHOR], stream);
  CONCAT_ON_DEVICE(DST(PSZ_ENCODED, 0), mem->comp_codec_out, mem->nbyte[PSZ_ENCODED], stream);
  CONCAT_ON_DEVICE(DST(PSZ_SPFMT, 0), mem->outlier2_validx_d(), mem->nbyte[PSZ_SPFMT], stream);
    // clang-format on

    /* output of this function */
    *out = mem->compressed_d();
    *outlen = pszheader_filesize(ctx->header);
    return PSZ_SUCCESS;
  };

  auto compress_encode_pass1_LC_TCMS = [&]() -> int {
    // Hi-TP mode: TCMS (cuSZ-Hi), or DRH for an ad hoc fix
    ctx->header->pipeline.hist = psz_hist::HistNull;
    size_t const lc_eq_n = tile_order_nd ? len_eq : len_linear;
    if (PIPELINE.codec1 == LC_DRH)
      lc_c::DRH_COMPRESS(
          (uint8_t*)mem->eq_d(), lc_eq_n * sizeof(E), mem->buf_lc(), &mem->comp_codec_outlen,
          stream);
    else
      lc_c::TCMS_COMPRESS(
          (uint8_t*)mem->eq_d(), lc_eq_n * sizeof(E), mem->buf_lc(), &mem->comp_codec_outlen,
          stream);
    mem->comp_codec_out = mem->buf_lc()->encoded_d();
    return PSZ_SUCCESS;
  };

  auto compress_encode_pass2_LC_RTR = [&]() -> int {
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
  };
  auto compress_encode_pass2_LC_BITR = [&]() -> int {
    size_t comp_bitr_outlen = 0;
    size_t const bitr_in_n = mem->nbyte[PSZ_ANCHOR] + mem->nbyte[PSZ_SPFMT];

    if (bitr_in_n != 0) {
      lc_c::BITR_COMPRESS(
          (uint8_t*)DST(PSZ_ANCHOR, 0), bitr_in_n, mem->buf_lc(), &comp_bitr_outlen, stream);
      cudaMemcpyAsync(
          DST(PSZ_ANCHOR, 0), (void*)mem->buf_lc()->encoded_d(), comp_bitr_outlen,
          cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
      sync_by_stream(stream);
    }
    ctx->header->entry[PSZ_ENC_PASS2_END] = ctx->header->entry[PSZ_ANCHOR] + comp_bitr_outlen;

    *out = mem->compressed_d();
    *outlen = pszheader_filesize(ctx->header);
    return PSZ_SUCCESS;
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
    else if (PIPELINE.codec1 == HFR_V4)
      status = compress_encode_pass1_HFR_v4();
    else if (PIPELINE.codec1 == FZG)
      status = compress_encode_pass1_FZG();
    else  // HF_r2, and HF (now an alias for HF_r2)
      status = compress_encode_pass1_Huffman_rev2();
    if (status != PSZ_SUCCESS) return status;

    if (PIPELINE.codec1 == FZG)  // just in case for the next round
      memset_device_async(mem->buf_fzg()->offset_counter_d(), 1, 0, stream);

    // HF/HF-rev2, FZG and LC are separeted from HFR* + HFD26.
    const bool hfr_fam =
        (PIPELINE.codec1 == HFR_PBKC or PIPELINE.codec1 == HFR_PBKGO or PIPELINE.codec1 == HFR or
         PIPELINE.codec1 == HFR_V3 or PIPELINE.codec1 == HFR_V4);
    const bool keep_global = (PIPELINE.codec1 == HF) or (PIPELINE.codec1 == HF_r2) or
                             (PIPELINE.codec1 == FZG) or hfr_fam;
    if (keep_global) {
      sync_by_stream(stream);
      ctx->header->splen =
          std::min<size_t>(mem->outlier2_host_get_num(), mem->outlier2_max_allowed_num());
    }
    else
      ctx->header->splen = 0;

    return compress_encode_pass1_wrapup();
  };

  // Liu, Tian, Wu et al. 2024; Wu and Pan et al. 2025
  auto compress_encode_HiCR = [&]() -> int {
    auto status1 = compress_encode_pass1_Huffman_rev2();
    if (status1 != PSZ_SUCCESS) return status1;
    if (auto s = compress_encode_pass1_wrapup(); s != PSZ_SUCCESS) return s;
    auto status2 = compress_encode_pass2_LC_RTR();
    return PSZ_SUCCESS;
  };

  // Liu, Tian, Wu et al. 2024; Wu and Pan et al. 2025
  // HiTP eq-only: TCMS for eq, raw anchor+spfmt (no BITR, fallback)
  auto compress_encode_HiTP_eq = [&]() -> int {
    auto status1 = compress_encode_pass1_LC_TCMS();
    if (status1 != PSZ_SUCCESS) return status1;
    return compress_encode_pass1_wrapup();
  };

  // Liu, Tian, Wu et al. 2024; Wu and Pan et al. 2025
  auto compress_encode_HiTP = [&]() -> int {
    auto status1 = compress_encode_pass1_LC_TCMS();
    if (status1 != PSZ_SUCCESS) return status1;
    if (auto s = compress_encode_pass1_wrapup(); s != PSZ_SUCCESS) return s;
    auto status2 = compress_encode_pass2_LC_BITR();
    if (status2 != PSZ_SUCCESS) return status2;
    return PSZ_SUCCESS;
  };

  //// execution

  auto status_pred = compress_predict();
  if (status_pred != PSZ_SUCCESS) return status_pred;

  // default:  HF(ec-quant) + raw(anchor/spfmt)
  // HiCR:     default + RTR(full block)
  // HiTP:     TCMS(ec-quant) + BITR(anchor/spfmt)
  // fallback: TCMS(ec-quant) + raw(anchor/spfmt)
  const bool codec1_is_lc_eq = (PIPELINE.codec1 == LC_TCMS) or (PIPELINE.codec1 == LC_DRH);
  auto status_encode = pszcodec_is_pass2(PIPELINE.codec2)
                           ? (codec1_is_lc_eq ? compress_encode_HiTP() : compress_encode_HiCR())
                       : codec1_is_lc_eq ? compress_encode_HiTP_eq()
                                         : compress_encode_default();
  if (status_encode != PSZ_SUCCESS) return status_encode;

  return PSZ_SUCCESS;
}

PPL_IMPL(void)::decomp_scatter(
    psz_header* header, _ptb::compact_cell<T, M>* d_spval_idx, T* d_space, void* stream)
{
  const auto len = header->len;
  // spl-y25 keeps eq in eq_d: output buffer start at zero
  // lrz and spl-y24: decode eq in the output buffer.
  if (header->pipeline.predictor == SplineY25)
    memset_device(d_space, len.x * len.y * len.z);
  if (header->splen != 0)
    psz::module::GPU_scatter<T, M>::kernel_v3_fuse(d_spval_idx, header->splen, d_space, stream);
}

PPL_IMPL(void)::decomp_predict(
    psz_header* header, PSZ_BUF* mem, T* d_anchor, T* d_xdata, void* stream)
{
  const auto eb = header->eb;
  const auto len = header->len;

  if (header->pipeline.predictor == Lorenzo)
    x_lrz(mem, d_xdata, eb, header->radius, stream);
  else if (header->pipeline.predictor == LorenzoZigZag)
    x_lrz_zz(mem, d_xdata, eb, header->radius, stream);
  else if (pszpredictor_is_spline(header->pipeline.predictor)) {
    mem->set_predictor(header->pipeline.predictor);  // anchor sizing
    if constexpr (std::is_same_v<T, f4>) {
      if (header->pipeline.predictor == SplineY24)
        spl_x_y24<T, E>::kernel(
            mem, d_anchor, make_view(d_xdata, len), eb, header->radius, header->intp_param,
            stream);
      else
        spl_x_y25<T, E>::kernel(
            mem, d_anchor, make_view(d_xdata, len), eb, header->radius, header->intp_param,
            stream);
    }
  }
}

// `--hfd-coarse` to select fallback coarse HFD
enum coarse_decoder { HF_coarse, HFR_coarse };

PPL_IMPL(int)::decompress(
    psz_header* header, PSZ_BUF* mem, u1* in, T* out, psz_stream_t stream, bool use_hfd_coarse)
{
  if (not dispatch<T, E>::pipeline_supported(header->pipeline))
    return PSZ_ABORT_UNSUPPORTED_PIPELINE;

  int routed_status;
  if (dispatch<T, E>::route_predictor(
          header->pipeline, [&](auto ppl) {
            routed_status = compression_pipeline<T, E, decltype(ppl)>::decompress(
                header, mem, in, out, stream, use_hfd_coarse);
          }))
    return routed_status;

  auto access = [&](int FIELD, szt offset_nbyte = 0) {
    return (void*)(in + header->entry[FIELD] + offset_nbyte);
  };

  auto d_anchor = (T*)access(PSZ_ANCHOR);
  auto d_spval_idx = (_ptb::compact_cell<T, M>*)access(PSZ_SPFMT);
  auto d_space = out, d_xdata = out;  // aliases
  auto len = header->len;
  phf_header h{};  // declared early so goto over STEP_DECODING is valid

  // One chunk (non-1Ki) can contain multiple ND tiles.
  int const nd = (len.z > 1) ? 3 : (len.y > 1) ? 2 : 1;
  auto const c1 = header->pipeline.codec1;
  bool const is_hfr_family =
      (c1 == HFR or c1 == HFR_PBKC or c1 == HFR_PBKGO or c1 == HFR_V3 or c1 == HFR_V4);
  bool const y25_tile_nd = nd >= 2 and header->pipeline.predictor == SplineY25;
  bool const tile_nd = (nd >= 2);  // tile-order is the only 2D/3D eq layout
  // spl-y25 needs mem->eq_d() (per-level clustering);
  // lrz*, and spl-y24 decode in place (to output directly)
  bool const eq_in_out = header->pipeline.predictor != SplineY25;

  if ((header->pipeline.codec1 == LC_TCMS or header->pipeline.codec1 == LC_DRH) and
      not pszcodec_is_pass2(header->pipeline.codec2)) {
    // TCMS/DRH-only: eq is LC-compressed, anchor/spfmt are raw in archive
    if (header->pipeline.codec1 == LC_DRH)
      lc_c::DRH_DECOMPRESS((uint8_t*)access(PSZ_ENCODED), mem->buf_lc(), stream);
    else
      lc_c::TCMS_DECOMPRESS((uint8_t*)access(PSZ_ENCODED), mem->buf_lc(), stream);
    // compat with spl-y25's mem->eq_d()
    size_t const lc_eq_n = tile_nd ? mem->eq_len() : (size_t)len.x * len.y * len.z;
    auto lc_eq_decoded = (E*)mem->buf_lc()->decoded_d();
    if (tile_nd)
      psz::module::GPU_cast<E, T>::kernel(lc_eq_decoded, mem->decode_fused_d(), lc_eq_n, stream);
    else if (eq_in_out)
      psz::module::GPU_cast<E, T>::kernel(lc_eq_decoded, d_space, lc_eq_n, stream);
    else
      cudaMemcpyAsync(
          mem->eq_d(), lc_eq_decoded, lc_eq_n * sizeof(E), cudaMemcpyDeviceToDevice,
          (cudaStream_t)stream);
    // d_anchor and d_spval_idx already initialized to access(PSZ_ANCHOR/PSZ_SPFMT)
    if (tile_nd and header->splen != 0)
      psz::module::GPU_scatter<T, M>::kernel_v3_fuse(
          d_spval_idx, header->splen, mem->decode_fused_d(), stream);
    else
      decomp_scatter(header, d_spval_idx, d_space, stream);
    goto STEP_PREDICT;
  }
  if (pszcodec_is_pass2(header->pipeline.codec2)) {
    if (header->pipeline.codec1 != LC_TCMS and header->pipeline.codec1 != LC_DRH) {
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
      // eq must be resolved before scatter
      if (tile_nd)
        phf::high_level<E>::HF_decode(
            mem->buf_hf(), h, (BYTE*)decomp_lc1, mem->decode_fused_d(), stream, psz_codec::HF);
      else if (eq_in_out)
        phf::high_level<E>::HF_decode(
            mem->buf_hf(), h, (BYTE*)decomp_lc1, d_space, stream, psz_codec::HF);
      else
        phf::high_level<E>::HF_decode(
            mem->buf_hf(), h, (BYTE*)decomp_lc1, mem->eq_d(), stream, psz_codec::HF);
      if (tile_nd and header->splen != 0)
        psz::module::GPU_scatter<T, M>::kernel_v3_fuse(
            d_spval_idx, header->splen, mem->decode_fused_d(), stream);
      else
        decomp_scatter(header, d_spval_idx, d_space, stream);
    }
    else {
      // HiTP: TCMS/DRH_DECOMPRESS eq + BITR_DECOMPRESS [ANCHOR][SPFMT]
      if (header->pipeline.codec1 == LC_DRH)
        lc_c::DRH_DECOMPRESS((uint8_t*)access(PSZ_ENCODED), mem->buf_lc(), stream);
      else
        lc_c::TCMS_DECOMPRESS((uint8_t*)access(PSZ_ENCODED), mem->buf_lc(), stream);
      size_t const lc_eq_n = tile_nd ? mem->eq_len() : (size_t)len.x * len.y * len.z;
      auto lc_eq_decoded = (E*)mem->buf_lc()->decoded_d();
      if (tile_nd)
        psz::module::GPU_cast<E, T>::kernel(lc_eq_decoded, mem->decode_fused_d(), lc_eq_n, stream);
      else if (eq_in_out)
        psz::module::GPU_cast<E, T>::kernel(lc_eq_decoded, d_space, lc_eq_n, stream);
      else
        cudaMemcpyAsync(
            mem->eq_d(), lc_eq_decoded, lc_eq_n * sizeof(E), cudaMemcpyDeviceToDevice,
            (cudaStream_t)stream);
      // A zero-byte anchor+spfmt region was never BITR-encoded.
      if (header->entry[PSZ_ENC_PASS2_END] != header->entry[PSZ_ANCHOR]) {
        lc_c::BITR_DECOMPRESS((uint8_t*)access(PSZ_ANCHOR), mem->buf_lc(), stream);
        auto decomp_lc2 = mem->buf_lc()->decoded_d();
        d_anchor = (T*)decomp_lc2;
        d_spval_idx =
            (_ptb::compact_cell<T, M>*)((byte_t*)decomp_lc2 +
                                        (header->entry[PSZ_SPFMT] - header->entry[PSZ_ANCHOR]));
      }
      if (tile_nd and header->splen != 0)
        psz::module::GPU_scatter<T, M>::kernel_v3_fuse(
            d_spval_idx, header->splen, mem->decode_fused_d(), stream);
      else
        decomp_scatter(header, d_spval_idx, d_space, stream);
      // eq already placed above
    }

    goto STEP_PREDICT;
  }

STEP_DECODING:

  memcpy_allkinds<D2H>((BYTE*)&h, (BYTE*)access(PSZ_ENCODED), sizeof(phf_header));
  if (header->pipeline.codec1 == FZG) {
    if constexpr (std::is_same_v<E, u2>) {
      bool const eq_in_out = header->pipeline.predictor != SplineY25;
      fzg_header h_fzg;
      fzg::high_level::decode(
          mem->buf_fzg(), h_fzg, (uint8_t*)access(PSZ_ENCODED), 0, mem->fzg_scratch_d(),
          mem->eq_len(), stream);
      if (tile_nd)
        psz::module::GPU_cast<E, T>::kernel(
            mem->fzg_scratch_d(), mem->decode_fused_d(), mem->eq_len(), stream);
      else if (eq_in_out)
        psz::module::GPU_cast<E, T>::kernel(mem->fzg_scratch_d(), d_space, mem->eq_len(), stream);
      else
        memcpy_allkinds<D2D>(mem->eq_d(), mem->fzg_scratch_d(), mem->eq_len());
    }
  }
  else {
    auto enc = (BYTE*)access(PSZ_ENCODED);
    // predictor chunksize == encoder chunksize
    auto const _pd = header->pipeline.predictor;
    int const _nd = (len.z > 1) ? 3 : (len.y > 1) ? 2 : 1;
    bool const _y25t = _nd >= 2 and _pd == SplineY25;
    bool const _tile =
        _y25t or (_nd >= 2 and (_pd == Lorenzo or _pd == LorenzoZigZag or _pd == SplineY24));
    int const magnitude =
        not _tile ? 10 : _y25t ? 12 : (_nd == 3 or pszpredictor_is_spline(_pd)) ? 11 : 10;
    // HFD26 works with HFR family
    auto decode_eq = [&](auto* dst) -> int {
      using Eout = std::remove_pointer_t<decltype(dst)>;
      auto const c = header->pipeline.codec1;
      auto const coarse =
          (c == HFR or c == HFR_PBKC or c == HFR_PBKGO or c == HFR_V3 or c == HFR_V4) ? HFR_coarse
                                                                                      : HF_coarse;
      if (coarse == HFR_coarse and not use_hfd_coarse)
        return phf::high_level<E>::template HFD26_decode<Eout>(
            mem->buf_hf(), h, enc, dst, stream, c, magnitude);
      if (coarse == HFR_coarse)
        return phf::high_level<E>::template HFR_decode<Eout>(
            mem->buf_hf(), h, enc, dst, stream, c, magnitude);
      return phf::high_level<E>::template HF_decode<Eout>(
          mem->buf_hf(), h, enc, dst, stream, (c == HF_r2) ? HF_r2 : HF);
    };
    int const decode_stat = tile_nd     ? decode_eq(mem->decode_fused_d())
                            : eq_in_out ? decode_eq(d_space)
                                        : decode_eq(mem->eq_d());
    if (decode_stat != PHF_SUCCESS) return PSZ_ABORT_NO_SUCH_CODEC;
  }

STEP_SCATTER:

  if (tile_nd and header->splen != 0)
    psz::module::GPU_scatter<T, M>::kernel_v3_fuse(
        d_spval_idx, header->splen, mem->decode_fused_d(), stream);
  else
    decomp_scatter(header, d_spval_idx, d_space, stream);

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
           + "." + "bk_" + to_string(ctx->header->radius * 2)                                    //
           + "." + suffix + "_" + t;
  };

  sync_by_stream(stream);

  if (ctx->cli->dump_hist) {
    memcpy_allkinds<D2H>(mem->hist_h(), mem->hist_d(), ctx->header->radius * 2, stream);
    _ptb::utils::tofile(dump_name("u4", "ht"), mem->hist_h(), ctx->header->radius * 2);
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
