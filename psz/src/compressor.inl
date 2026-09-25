#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>

#include "compressor.hh"
#include "context_impl.h"
#include "extrema.hh"
#include "fzg_hl.hh"
#include "kernel.hh"
#include "lc_gen/lc_gen.h"
#include "module.hh"
#include "phf.hh"
#include "ptb.hh"

namespace psz {

using _ptb::make_view;
using std::string;
using std::to_string;

using _2609::is_lc;
using _2609::ModuleCodec1;
using _2609::ModuleCodec2;
using _2609::ModuleLorenzo;
using _2609::ModuleSplineY24;
using _2609::ModuleSplineY25;
using _2609::needs_eq4;
using _2609::pass2_head;
using _2609::Pipeline;
using _2609::seg_anchor;
using _2609::seg_encoded;
using _2609::seg_header;
using _2609::seg_pass1_end;
using _2609::seg_pass2_end;
using _2609::seg_spfmt;
using _2609::Segment;
using _2609::valid;

static_assert(
    seg_header == PSZ_HEADER and seg_encoded == PSZ_ENCODED and seg_anchor == PSZ_ANCHOR and
    seg_spfmt == PSZ_SPFMT and seg_pass1_end == PSZ_ENC_PASS1_END and
    seg_pass2_end == PSZ_ENC_PASS2_END);
static_assert(
    seg_header == PSZHEADER_HEADER and seg_encoded == PSZHEADER_ENCODED and
    seg_anchor == PSZHEADER_ANCHOR and seg_spfmt == PSZHEADER_SPFMT and
    seg_pass1_end == PSZHEADER_ENC_PASS1_END and seg_pass2_end == PSZHEADER_ENC_PASS2_END);

template <typename T, class P, class C1, class C2>
struct compressor_cpp<T, Pipeline<P, C1, C2>> {
  static_assert(Pipeline<P, C1, C2>::valid, "this pipeline has no walk through the machine");

  using Buf = Buf_Comp<T>;
  using E = std::conditional_t<needs_eq4(C1::kind), u4, u2>;
  using Lorenzo_c = module::GPU_c_lorenzo_nd<PredictorTyping<T, E>, typename P::Features>;
  using Lorenzo_x = module::GPU_x_lorenzo_nd<PredictorTyping<T, E>, typename P::Features>;
  using Spline_c = std::conditional_t<
      P::kind == SplineY24, module::GPU_c_spline_y24<PredictorTyping<T, E>, PredictorFeature<0b0>>,
      module::GPU_c_spline_y25<PredictorTyping<T, E>, PredictorFeature<0b0>>>;
  using Spline_x = std::conditional_t<
      P::kind == SplineY24, module::GPU_x_spline_y24<PredictorTyping<T, E>>,
      module::GPU_x_spline_y25<PredictorTyping<T, E>>>;

  static constexpr bool spline = P::spline;
  static constexpr bool has_codec2 = (C2::kind != CodecNull);
  static constexpr psz_codec pass1 = C1::kind;
  static constexpr bool hfr = _2609::is_hfr(pass1);
  static constexpr bool enable_localized = _2609::unpred_localized(pass1);
  static constexpr bool enable_global = _2609::unpred_spill(pass1);
  // spl-y25 still need standalone eq
  static constexpr bool eq_in_out = P::kind != SplineY25;

  static void concat_on_device(void* dst, void* src, size_t nbyte, void* stream)
  {
    if (nbyte != 0) memcpy_allkinds_async<D2D>((u1*)dst, (u1*)src, nbyte, stream);
  }

  static u1* archive_at(psz_ctx* ctx, Buf* mem, Segment seg)
  { return mem->compressed_d() + ctx->header->entry[seg]; }

  static void comp_predict(psz_ctx* ctx, Buf* mem, T* in, void* stream)
  {
    const auto eb = ctx->header->eb;
    const auto radius = ctx->header->radius;
    const auto len = ctx->header->len;

    if constexpr (spline) {
      if constexpr (std::is_same_v<T, f4>)  // FIXME spl-f8
        Spline_c::kernel(
            mem, make_view(in, len), eb, ctx->header->user_input_eb, radius,
            ctx->header->intp_param, enable_global, stream);
    }
    else
      Lorenzo_c::kernel(
          mem, make_view(in, len), eb, radius, enable_localized, enable_global, stream);
  }

  static size_t eq_len(psz_len len)
  {
    auto const cdiv = [](u4 v, u4 d) -> size_t { return (v + d - 1u) / d; };
    int const nd = (len.z > 1) ? 3 : (len.y > 1) ? 2 : 1;
    if (nd < 2) return (size_t)len.x * len.y * len.z;
    if constexpr (spline) {
      if constexpr (P::kind == SplineY25)
        return nd == 3 ? cdiv(len.x, 16) * cdiv(len.y, 16) * cdiv(len.z, 16) * 4096
                       : cdiv(len.x, 64) * cdiv(len.y, 64) * 4096;
      else
        return cdiv(len.x, 32) * cdiv(len.y, 8) * cdiv(len.z, 8) * 2048;
    }
    else
      return nd == 3 ? cdiv(len.x, 32) * cdiv(len.y, 8) * cdiv(len.z, 8) * 2048
                     : cdiv(len.x, 32) * cdiv(len.y, 32) * 1024;
  }

  static int block_magnitude(psz_len len)
  {
    int const nd = (len.z > 1) ? 3 : (len.y > 1) ? 2 : 1;
    if (nd < 2) return 10;
    if constexpr (spline) {
      if constexpr (P::kind == SplineY25)
        return 12;
      else
        return 11;
    }
    else
      return nd == 3 ? 11 : 10;
  }

  static HFR_Opts encode_opts(psz_ctx* ctx, Buf* mem)
  {
    HFR_Opts opts;
    opts.reduce_times = 1;  // PBKGO restructed to be >1
    // --rmerge-count to ovrride
    if (ctx->cli and ctx->cli->hfr_rmerge_count > 0)
      opts.reduce_times = ctx->cli->hfr_rmerge_count;
    opts.magnitude = block_magnitude(ctx->header->len);
    opts.block_outliers = mem->template block_outliers_d<E>();
    return opts;
  }

  static int comp_encode_pass1(psz_ctx* ctx, Buf* mem, void* stream)
  {
    const auto len_eq = eq_len(ctx->header->len);

    if constexpr (is_lc(pass1)) {
      mem->lc_wire_encoded(mem->compressed_d() + sizeof(psz_header));
      if constexpr (pass1 == LC_DRH)
        lc_c::DRH_COMPRESS(
            (uint8_t*)mem->template eq_d<E>(), len_eq * sizeof(E), mem->buf_lc1(),
            &mem->comp_codec_outlen, stream);
      else
        lc_c::TCMS_COMPRESS(
            (uint8_t*)mem->template eq_d<E>(), len_eq * sizeof(E), mem->buf_lc1(),
            &mem->comp_codec_outlen, stream);
      mem->comp_codec_out = mem->buf_lc1()->encoded_d();
      mem->lc_wire_encoded(nullptr);
    }
    else if constexpr (pass1 == FZG) {
      if constexpr (std::is_same_v<E, u2>) {  // fzg::E is fixed at u2
        fzg_header dummy_header{};
        auto const status = fzg::high_level::encode(
            mem->buf_fzg(), mem->template eq_d<E>(), len_eq, &mem->comp_codec_out,
            &mem->comp_codec_outlen, dummy_header, stream);
        sync_by_stream(stream);
        return status == 0 ? PSZ_SUCCESS : PSZ_ABORT_NO_SUCH_CODEC;
      }
      else
        return PSZ_ABORT_NO_SUCH_CODEC;
    }
    else {
      phf_header dummy_header{};
      phf::Buf<E>* hf = nullptr;
      if constexpr (hfr)
        hf = mem->buf_hfr();
      else
        hf = mem->buf_hf();
      auto const eq = mem->template eq_d<E>();
      auto const hist = ctx->header->pipeline.hist;

      if constexpr (pass1 == HFR or not hfr) {
        phf::high_level<E>::make_book(hf, eq, len_eq, ctx->bklen, stream, hist);
        if constexpr (pass1 == HFR)
          phf::high_level<E>::HFR_RTBK_encode(
              hf, eq, len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen, dummy_header, stream,
              nullptr, nullptr, encode_opts(ctx, mem));
        else
          phf::high_level<E>::HF_encode(
              hf, eq, len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen, dummy_header, stream,
              pass1);
      }
      else {
        if constexpr (pass1 == HFR_V3 or pass1 == HFR_V4)
          phf::high_level<E>::HFR_pick_pbk(hf, eq, len_eq, ctx->bklen, stream, hist);
        phf::high_level<E>::HFR_PBK_encode(
            hf, eq, len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen, dummy_header, stream,
            pass1, nullptr, nullptr, encode_opts(ctx, mem));
      }
    }
    return PSZ_SUCCESS;
  }

  static void comp_encode_pass2(psz_ctx* ctx, Buf* mem, void* stream)
  {
    constexpr auto head = pass2_head(C2::kind);
    auto span = (uint8_t*)archive_at(ctx, mem, head);
    auto span_nbyte = ctx->header->entry[seg_pass1_end] - ctx->header->entry[head];

    size_t nbyte;
    if constexpr (C2::kind == LC_BITR)
      lc_c::BITR_COMPRESS(span, span_nbyte, mem->buf_lc2(), &nbyte, stream);
    else
      lc_c::RTR_COMPRESS(span, span_nbyte, mem->buf_lc2(), &nbyte, stream);

    concat_on_device((void*)span, mem->buf_lc2()->encoded_d(), nbyte, stream);
    ctx->header->entry[seg_pass2_end] = ctx->header->entry[head] + nbyte;
  }

  static int comp_process(psz_ctx* ctx, Buf* mem, T* in, void* stream)
  {
    comp_predict(ctx, mem, in, stream);
    if constexpr (pass1 == CodecNull) {
      mem->comp_codec_out = (u1*)mem->template eq_d<E>();
      mem->comp_codec_outlen = eq_len(ctx->header->len) * sizeof(E);
      return PSZ_SUCCESS;
    }
    else {
      int status = comp_encode_pass1(ctx, mem, stream);
      if constexpr (has_codec2) {
        if (status != PSZ_SUCCESS) return status;
        status = comp_concat_segments(ctx, mem, stream);
        if (status != PSZ_SUCCESS) return status;
        comp_encode_pass2(ctx, mem, stream);
      }
      return status;
    }
  }

  static int comp_archive(psz_ctx* ctx, Buf* mem, u1** out, size_t* out_bytes, void* stream)
  {
    if constexpr (not has_codec2) {
      int const status = comp_concat_segments(ctx, mem, stream);
      if (status != PSZ_SUCCESS) return status;
    }
    sync_by_stream(stream);
    comp_publish(ctx, mem, out, out_bytes);
    return PSZ_SUCCESS;
  }

  static int set_splen(psz_ctx* ctx, Buf* mem, void* stream)
  {
    sync_by_stream(stream);
    ctx->header->splen = mem->outlier2_host_get_num();
    if (ctx->header->splen == mem->buf_outlier2()->max_allowed_num())
      return PSZ_WARN_OUTLIER_TOO_MANY;
    return PSZ_SUCCESS;
  }

  static void comp_publish(psz_ctx* ctx, Buf* mem, u1** out, size_t* out_bytes)
  {
    *out = mem->compressed_d();
    *out_bytes = pszheader_filesize(ctx->header);
  }

  static size_t anchor_nbyte(Buf* mem) { return spline ? sizeof(T) * mem->anchor_len() : 0; }
  static size_t spfmt_nbyte(psz_ctx* ctx)
  { return sizeof(_ptb::compact_cell<T, u4>) * ctx->header->splen; }

  static int set_entries(psz_ctx* ctx, Buf* mem)
  {
    mem->nbyte[seg_header] = sizeof(psz_header);
    mem->nbyte[seg_encoded] = sizeof(u1) * mem->comp_codec_outlen;
    mem->nbyte[seg_anchor] = anchor_nbyte(mem);
    mem->nbyte[seg_spfmt] = spfmt_nbyte(ctx);
    mem->nbyte[seg_pass1_end] = 0;

    // Every stage starts 8-aligned.
    ctx->header->entry[0] = 0;
    for (auto i = 1; i <= seg_pass2_end; i++)
      ctx->header->entry[i] = ctx->header->entry[i - 1] + _2609::pad8(mem->nbyte[i - 1]);

    if (pszheader_filesize(ctx->header) > mem->compressed_max_bytes())
      return PSZ_ABORT_COMPRESSED_TOO_LARGE;
    return PSZ_SUCCESS;
  }

  // pad to 8 bytes for each archive segment
  static void clear_pad(psz_ctx* ctx, Buf* mem, Segment seg, void* stream)
  {
    auto const gap = _2609::pad8(mem->nbyte[seg]) - mem->nbyte[seg];
    if (gap != 0) memset_device_async(archive_at(ctx, mem, seg) + mem->nbyte[seg], gap, 0, stream);
  }

  static int comp_concat_segments(psz_ctx* ctx, Buf* mem, void* stream)
  {
    int status = set_splen(ctx, mem, stream);
    if (status != PSZ_SUCCESS) return status;
    status = set_entries(ctx, mem);
    if (status != PSZ_SUCCESS) return status;

    concat_on_device(
        archive_at(ctx, mem, seg_anchor), mem->anchor_d(), mem->nbyte[seg_anchor], stream);
    if ((void*)mem->comp_codec_out != archive_at(ctx, mem, seg_encoded))
      concat_on_device(
          archive_at(ctx, mem, seg_encoded), mem->comp_codec_out, mem->nbyte[seg_encoded], stream);
    concat_on_device(
        archive_at(ctx, mem, seg_spfmt), mem->outlier2_validx_d(), mem->nbyte[seg_spfmt], stream);

    clear_pad(ctx, mem, seg_encoded, stream);
    clear_pad(ctx, mem, seg_anchor, stream);
    clear_pad(ctx, mem, seg_spfmt, stream);
    return PSZ_SUCCESS;
  }

  static void decomp_predict(psz_header* header, Buf* mem, T* d_anchor, T* out, void* stream)
  {
    const auto eb = header->eb;
    const auto radius = header->radius;
    const auto len = header->len;

    if constexpr (spline) {
      if constexpr (std::is_same_v<T, f4>)
        Spline_x::kernel(
            mem, d_anchor, make_view(out, len), eb, radius, header->intp_param, stream);
    }
    else
      Lorenzo_x::kernel(mem, out, eb, radius, stream);
  }

  static int decomp_decode(
      psz_header* header, Buf* mem, u1* in, T* out, T** d_anchor, void* stream)
  {
    auto access = [&](Segment seg) { return (void*)(in + header->entry[seg]); };
    auto const len = header->len;
    auto const len_linear = (size_t)len.x * len.y * len.z;
    int const nd = (len.z > 1) ? 3 : (len.y > 1) ? 2 : 1;
    bool const tile_nd = nd >= 2;

    *d_anchor = (T*)access(seg_anchor);
    auto d_spvi = (_ptb::compact_cell<T, M>*)access(seg_spfmt);
    auto d_space = out;

    // eq lands wherever the predictor reads it from
    auto place_eq = [&](E* src, size_t n) {
      if (tile_nd)
        module::GPU_cast<E, T>::kernel(src, mem->decode_fused_d(), n, stream);
      else if constexpr (eq_in_out)
        module::GPU_cast<E, T>::kernel(src, d_space, n, stream);
      else
        concat_on_device(mem->template eq_d<E>(), src, n * sizeof(E), stream);
    };

    auto encoded = (BYTE*)access(seg_encoded);

    if constexpr (is_lc(pass1)) {
      if constexpr (pass1 == LC_DRH)
        lc_c::DRH_DECOMPRESS((uint8_t*)access(seg_encoded), mem->buf_lc1(), stream);
      else
        lc_c::TCMS_DECOMPRESS((uint8_t*)access(seg_encoded), mem->buf_lc1(), stream);
      place_eq((E*)mem->buf_lc1()->decoded_d(), tile_nd ? mem->eq_len() : len_linear);
    }

    // a zero-byte span was never encoded
    if constexpr (has_codec2) {
      constexpr auto head = pass2_head(C2::kind);
      if (header->entry[seg_pass2_end] != header->entry[head]) {
        if constexpr (C2::kind == LC_BITR)
          lc_c::BITR_DECOMPRESS((uint8_t*)access(head), mem->buf_lc2(), stream);
        else
          lc_c::RTR_DECOMPRESS((uint8_t*)access(head), mem->buf_lc2(), stream);

        auto staged = (byte_t*)mem->buf_lc2()->decoded_d();
        auto decoded = [&](Segment seg) {
          return staged + (header->entry[seg] - header->entry[head]);
        };
        if constexpr (head == seg_encoded) encoded = (BYTE*)decoded(seg_encoded);
        *d_anchor = (T*)decoded(seg_anchor);
        d_spvi = (_ptb::compact_cell<T, M>*)decoded(seg_spfmt);
      }
    }

    if constexpr (pass1 == FZG) {
      if constexpr (std::is_same_v<E, u2>) {
        fzg_header h_fzg;
        fzg::high_level::decode(
            mem->buf_fzg(), h_fzg, (uint8_t*)encoded, 0, mem->buf_fzg()->out_d(), mem->eq_len(),
            stream);
        place_eq(mem->buf_fzg()->out_d(), mem->eq_len());
      }
    }
    else if constexpr (pass1 == CodecNull)
      place_eq((E*)encoded, tile_nd ? mem->eq_len() : len_linear);
    else if constexpr (not is_lc(pass1)) {
      phf_header hdr{};
      memcpy_allkinds<D2H>((BYTE*)&hdr, encoded, sizeof(phf_header));

      auto decode_eq = [&](auto* dst) -> int {
        using Eout = std::remove_pointer_t<decltype(dst)>;
        if constexpr (hfr)
          return phf::high_level<E>::template HFD26_decode<Eout>(
              mem->buf_hfr(), hdr, encoded, dst, stream, pass1, block_magnitude(len));
        else  // HF_r2 fully supersedes HF.
          return phf::high_level<E>::template HF_decode<Eout>(
              mem->buf_hf(), hdr, encoded, dst, stream, HF_r2);
      };

      int stat;
      if (tile_nd)
        stat = decode_eq(mem->decode_fused_d());
      else if constexpr (eq_in_out)
        stat = decode_eq(d_space);
      else
        stat = decode_eq(mem->template eq_d<E>());
      if (stat != PHF_SUCCESS) return PSZ_ABORT_NO_SUCH_CODEC;
    }

    if (header->splen != 0)
      module::GPU_scatter<T, M>::kernel_v3_fuse(
          d_spvi, header->splen, tile_nd ? mem->decode_fused_d() : d_space, stream);

    return PSZ_SUCCESS;
  }

  static int decomp_process(psz_header* header, Buf* mem, u1* in, T* out, void* stream)
  {
    T* d_anchor = nullptr;
    auto const stat = decomp_decode(header, mem, in, out, &d_anchor, stream);
    if (stat != PSZ_SUCCESS) return stat;

    decomp_predict(header, mem, d_anchor, out, stream);

    return PSZ_SUCCESS;
  }
};

template <typename T, class PPL>
struct compressor_cpp<T, PPL>::dispatch {
  template <class P, class C1, class F>
  static bool route_codec2(psz_ppl const& p, F&& f)
  {
    auto go = [&f](auto ppl) {
      if constexpr (decltype(ppl)::valid) {
        f(ppl);
        return true;
      }
      else
        return false;
    };

    switch (p.codec2) {
      case CodecNull: return go(Pipeline<P, C1>{});
      case LC_BITR: return go(Pipeline<P, C1, ModuleCodec2<LC_BITR>>{});
      case LC_RTR: return go(Pipeline<P, C1, ModuleCodec2<LC_RTR>>{});
      default: return false;
    }
  }

  template <class P, class F>
  static bool route_codec1(psz_ppl const& p, F&& f)
  {
    switch (p.codec1) {
      case CodecNull: return route_codec2<P, ModuleCodec1<CodecNull>>(p, f);
      case HF:
      case HF_r2: return route_codec2<P, ModuleCodec1<HF_r2>>(p, f);
      case HFR: return route_codec2<P, ModuleCodec1<HFR>>(p, f);
      case HFR_V4: return route_codec2<P, ModuleCodec1<HFR_V4>>(p, f);
      case HFR_V3: return route_codec2<P, ModuleCodec1<HFR_V3>>(p, f);
      case HFR_PBKC: return route_codec2<P, ModuleCodec1<HFR_PBKC>>(p, f);
      case HFR_PBKGO: return route_codec2<P, ModuleCodec1<HFR_PBKGO>>(p, f);
      case LC_TCMS: return route_codec2<P, ModuleCodec1<LC_TCMS>>(p, f);
      case LC_DRH: return route_codec2<P, ModuleCodec1<LC_DRH>>(p, f);
      case FZG: return route_codec2<P, ModuleCodec1<FZG>>(p, f);
      default: return false;
    }
  }

  template <class F>
  static bool route_predictor(psz_ppl const& p, F&& f)
  {
    switch (p.predictor) {
      case Lorenzo: return route_codec1<ModuleLorenzo<>>(p, f);
      case LorenzoZigZag: return route_codec1<ModuleLorenzo<PredictorFeature<1>>>(p, f);
      case SplineY24: return route_codec1<ModuleSplineY24<>>(p, f);
      case SplineY25: return route_codec1<ModuleSplineY25<>>(p, f);
      default: return false;
    }
  }
};

#define PIPELINE ctx->header->pipeline

#define PPL_IMPL(RET_TYPE)         \
  template <typename T, class PPL> \
  RET_TYPE compressor_cpp<T, PPL>

PPL_IMPL(void*)::compress_init(psz_ctx* ctx)
{
  auto mem = new Buf_Comp<T>(ctx->header->len, true, 2);
  mem->register_header(ctx->header);
  return mem;
}

PPL_IMPL(void*)::compress_init_2stage(psz_ctx* ctx) { return compress_init(ctx); }

PPL_IMPL(void*)::compress_init_3stage(psz_ctx* ctx)
{
  auto mem = new Buf_Comp<T>(ctx->header->len, true, 3);
  mem->register_header(ctx->header);
  return mem;
}

PPL_IMPL(void*)::decompress_init(psz_header* header)
{
  auto mem = new Buf_Comp<T>(header->len, false, _2609::nstage_of(header->pipeline));
  mem->register_header(header);
  return mem;
}

PPL_IMPL(psz_data_summary)::compress_extrema(psz_ctx* ctx, T* in, psz_stream_t stream)
{
  auto const [min_val, max_val, avg_val, rng] =
      psz::cuda::GPU_get_extrema<T>::kernel(in, ctx->len_linear, stream);
  ctx->header->min_val = min_val;
  ctx->header->max_val = max_val;
  return {min_val, max_val, rng, std::numeric_limits<double>::quiet_NaN(), avg_val};
}

PPL_IMPL(int)::compress_analysis(psz_ctx* ctx, Buf_Comp<T>* mem, T* in, u4* h_hist, void* stream)
{
  psz_ppl const saved = PIPELINE;
  auto const status = compress_process(
      ctx, psz_ppl{saved.predictor, saved.hist, CodecNull, CodecNull}, mem, in, stream);
  PIPELINE = saved;
  if (status != PSZ_SUCCESS) return status;

  sync_by_stream(stream);
  ctx->header->splen = mem->outlier2_host_get_num();

  auto d_hist = MAKE_UNIQUE_DEVICE(u4, ctx->bklen);
  module::GPU_histogram_Cauchy<u2>::kernel(
      mem->template eq_d<u2>(), mem->len_linear, d_hist.get(), ctx->bklen, stream);
  memcpy_allkinds_async<D2H>(h_hist, d_hist.get(), ctx->bklen, stream);
  sync_by_stream(stream);

  return PSZ_SUCCESS;
}

PPL_IMPL(int)::compress_process(
    psz_ctx* ctx, psz_ppl pipeline, Buf_Comp<T>* mem, T* in, void* stream)
{
  if (pipeline.codec2 != CodecNull and not mem->buf_lc2()) {
    fprintf(
        stderr, "[psz::warning] 2-stage compressor: pass 2 dropped; use compress_init_3stage\n");
    pipeline.codec2 = CodecNull;
  }
  ctx->header->pipeline = pipeline;
  if (not valid(pipeline)) return PSZ_ABORT_UNSUPPORTED_PIPELINE;
  if (_2609::is_spline(pipeline.predictor) and not std::is_same_v<T, f4>)
    return PSZ_ABORT_UNSUPPORTED_TYPE;
  if (not mem->select(pipeline)) return PSZ_ABORT_UNSUPPORTED_PIPELINE;

  int status = PSZ_ABORT_UNSUPPORTED_PIPELINE;
  dispatch::route_predictor(pipeline, [&](auto ppl) {
    status = compressor_cpp<T, decltype(ppl)>::comp_process(ctx, mem, in, stream);
  });
  return status;
}

PPL_IMPL(int)::compress_archive(
    psz_ctx* ctx, Buf_Comp<T>* mem, psz_header* out_header, u1** out, size_t* out_bytes,
    void* stream)
{
  if (not mem->compressed_d()) return PSZ_ABORT_UNSUPPORTED_PIPELINE;
  int status = PSZ_ABORT_UNSUPPORTED_PIPELINE;
  dispatch::route_predictor(PIPELINE, [&](auto ppl) {
    status = compressor_cpp<T, decltype(ppl)>::comp_archive(ctx, mem, out, out_bytes, stream);
  });
  if (ctx->cli and status == PSZ_SUCCESS) compress_dump_frame(ctx, mem, stream);
  if (status != PSZ_SUCCESS) return status;
  *out_header = *(ctx->header);
  memcpy_allkinds<H2D>(*out, (u1*)ctx->header, sizeof(psz_header));
  return PSZ_SUCCESS;
}

PPL_IMPL(int)::compress_reset(psz_ctx*, Buf_Comp<T>* mem, psz_stream_t stream)
{
  mem->reset(stream);
  return PSZ_SUCCESS;
}

PPL_IMPL(int)::decompress_process(
    psz_ctx* ctx, Buf_Comp<T>* mem, u1* in, T* out, psz_stream_t stream)
{
  auto const header = ctx->header;
  if (not valid(header->pipeline)) return PSZ_ABORT_UNSUPPORTED_PIPELINE;
  if (_2609::is_spline(header->pipeline.predictor) and not std::is_same_v<T, f4>)
    return PSZ_ABORT_UNSUPPORTED_TYPE;
  if (not mem->select(header->pipeline)) return PSZ_ABORT_UNSUPPORTED_PIPELINE;

  int status = PSZ_ABORT_UNSUPPORTED_PIPELINE;
  dispatch::route_predictor(header->pipeline, [&](auto ppl) {
    status = compressor_cpp<T, decltype(ppl)>::decomp_process(header, mem, in, out, stream);
  });
  return status;
}

PPL_IMPL(int)::decompress_reset(psz_ctx*, Buf_Comp<T>*, psz_stream_t) { return PSZ_SUCCESS; }

PPL_IMPL(void)::compress_dump_frame(psz_ctx* ctx, Buf_Comp<T>* mem, psz_stream_t stream)
{
  auto dump_name = [&](string t, string suffix = ".quant") -> string {
    return string(ctx->cli->file_input)                                                //
           + "." + string(ctx->cli->char_mode) + "_" + string(ctx->cli->char_meta_eb)  //
           + "." + "bk_" + to_string(ctx->header->radius * 2)                          //
           + "." + suffix + "_" + t;
  };

  sync_by_stream(stream);

  auto go = [&](auto e) {
    using E = decltype(e);
    phf::Buf<E>* hf = nullptr;
    if constexpr (sizeof(E) == 4)
      hf = mem->buf_hfr();
    else
      hf = mem->buf_hf();
    if (ctx->cli->dump_hist and hf) {
      memcpy_allkinds<D2H>(hf->hist_h(), hf->hist_d(), ctx->header->radius * 2, stream);
      _ptb::utils::tofile(dump_name("u4", "ht"), hf->hist_h(), ctx->header->radius * 2);
    }
    if (ctx->cli->dump_quantcode) {
      printf("[psz::dump] dump quant to file: %s\n", dump_name("quant").c_str());
      auto h_eq = MAKE_UNIQUE_HOST(E, mem->len_linear);
      memcpy_allkinds<D2H>(h_eq.get(), mem->template eq_d<E>(), mem->len_linear, stream);
      _ptb::utils::tofile(
          dump_name("u" + to_string(sizeof(E)), "qt"), h_eq.get(), mem->len_linear);
    }
  };
  if (needs_eq4(PIPELINE))
    go(u4{});
  else
    go(u2{});
}

#undef PPL_IMPL
#undef PIPELINE

}  // namespace psz
