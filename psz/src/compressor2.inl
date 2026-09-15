#ifndef PSZ_COMPRESSOR2_INL
#define PSZ_COMPRESSOR2_INL

#include "pipeline.h"
#include <type_traits>

#include "compressor.hh"
#include "fzg_hl.hh"
#include "kernel.hh"
#include "lc_gen/lc_gen.h"
#include "module.hh"
#include "phf.hh"
#include "ptb.hh"

namespace psz {

using _ptb::make_view;

// device-to-device concat; an empty segment is a no-op
inline void concat_on_device(void* dst, void* src, size_t nbyte, void* stream)
{
  if (nbyte != 0) memcpy_allkinds_async<D2D>((u1*)dst, (u1*)src, nbyte, stream);
}

using namespace _2609;

static_assert(
    seg_header == PSZ_HEADER and seg_encoded == PSZ_ENCODED and seg_anchor == PSZ_ANCHOR and
    seg_spfmt == PSZ_SPFMT and seg_pass1_end == PSZ_ENC_PASS1_END and
    seg_pass2_end == PSZ_ENC_PASS2_END);

template <class P, typename T, typename E, bool IsSpline = P::spline>
struct spline_kernels {
  using c = psz::module::GPU_c_spline_y25<PredictorTyping<T, E>, PredictorFeature<0b0>>;
  using x = psz::module::GPU_x_spline_y25<PredictorTyping<T, E>>;
};

template <class P, typename T, typename E>
struct spline_kernels<P, T, E, true> {
  static constexpr bool y24 = (P::kind == SplineY24);
  using c = std::conditional_t<
      y24, psz::module::GPU_c_spline_y24<PredictorTyping<T, E>, PredictorFeature<0b0>>,
      psz::module::GPU_c_spline_y25<PredictorTyping<T, E>, PredictorFeature<0b0>>>;
  using x = std::conditional_t<
      y24, psz::module::GPU_x_spline_y24<PredictorTyping<T, E>>,
      psz::module::GPU_x_spline_y25<PredictorTyping<T, E>>>;
};

template <class P, bool IsSpline = P::spline>
struct predicts_y25 {
  static constexpr bool value = false;
};

template <class P>
struct predicts_y25<P, true> {
  static constexpr bool value = (P::kind == SplineY25);
};

namespace pipeline_routine {

template <typename T, typename E>
void histogram(psz_ctx* ctx, Buf_Comp<T, E>* mem, size_t len_eq, void* stream)
{
  if (ctx->header->pipeline.hist == HistSp)
    psz::module::GPU_histogram_Cauchy<E>::kernel(
        mem->eq_d(), len_eq, mem->hist_d(), ctx->bklen, stream);
  else
    psz::module::GPU_histogram_generic<E>::kernel(
        mem->eq_d(), len_eq, mem->hist_d(), ctx->bklen, mem->hist_generic_grid_dim,
        mem->hist_generic_block_dim, mem->hist_generic_shmem_use, mem->hist_generic_repeat,
        stream);
}

template <typename T, typename E>
int set_splen(psz_ctx* ctx, Buf_Comp<T, E>* mem, void* stream)
{
  sync_by_stream(stream);
  ctx->header->splen = mem->outlier2_host_get_num();
  if (ctx->header->splen == mem->buf_outlier2()->max_allowed_num())
    return PSZ_WARN_OUTLIER_TOO_MANY;
  return PSZ_SUCCESS;
}

template <typename T, typename E>
void publish(psz_ctx* ctx, Buf_Comp<T, E>* mem, u1** out, size_t* outlen)
{
  *out = mem->compressed_d();
  *outlen = pszheader_filesize(ctx->header);
}

}  // namespace pipeline_routine

template <typename T, typename E, class P, class C1, class C2>
struct compression_pipeline<T, E, Pipeline<P, C1, C2>> {
  static_assert(Pipeline<P, C1, C2>::valid, "this pipeline has no walk through the machine");

  using Buf = Buf_Comp<T, E>;
  using Lorenzo_c = psz::module::GPU_c_lorenzo_nd<PredictorTyping<T, E>, typename P::Features>;
  using Lorenzo_x = psz::module::GPU_x_lorenzo_nd<PredictorTyping<T, E>, typename P::Features>;
  using Spline_c = typename spline_kernels<P, T, E>::c;
  using Spline_x = typename spline_kernels<P, T, E>::x;

  static constexpr bool spline = P::spline;
  static constexpr bool lc_pass2 = (C2::kind == LC_BITR or C2::kind == LC_RTR);
  static constexpr psz_codec pass1 = C1::kind;
  static constexpr bool hfr = is_hfr(pass1);
  static constexpr bool enable_localized = unpred_localized(pass1);
  static constexpr bool enable_global = unpred_spill(pass1);

  static u1* archive_at(psz_ctx* ctx, Buf* mem, Segment seg)
  { return mem->compressed_d() + ctx->header->entry[seg]; }

  static void* compress_init(psz_ctx* ctx)
  {
    auto mem = new Buf(ctx->header->len, true, hfr);
    mem->register_header(ctx->header);
    if (ctx->header->pipeline.hist == HistGeneric)
      psz::module::GPU_histogram_generic<E>::init(
          mem->len_linear, Buf::max_bklen, mem->hist_generic_grid_dim, mem->hist_generic_block_dim,
          mem->hist_generic_shmem_use, mem->hist_generic_repeat);
    return mem;
  }

  static void* decompress_init(psz_header* header)
  {
    auto mem = new Buf(header->len, false, hfr);
    mem->register_header(header);
    return mem;
  }

  static void predict(psz_ctx* ctx, Buf* mem, T* in, void* stream)
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

  static void build_runtime_book(psz_ctx* ctx, Buf* mem, void* stream)
  {
    memcpy_allkinds<D2H>(mem->hist_h(), mem->hist_d(), ctx->bklen);
    phf::high_level<E>::HF_build_book(mem->buf_hf(), mem->hist_h(), ctx->bklen, stream);
  }

  static void pick_pbk(psz_ctx* ctx, Buf* mem, void* stream)
  {
    phf::high_level<E>::HFR_pick_pbk(
        mem->buf_hf(), mem->hist_d(), ctx->bklen, eq_len(ctx->header->len), stream);
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

  static void make_book(psz_ctx* ctx, Buf* mem, size_t len_eq, void* stream)
  {
    if constexpr (needs_book(pass1)) {
      pipeline_routine::histogram(ctx, mem, len_eq, stream);
      if constexpr (pass1 == HFR_V3 or pass1 == HFR_V4)
        pick_pbk(ctx, mem, stream);
      else
        build_runtime_book(ctx, mem, stream);
    }
  }

  static HFR_Opts encode_opts(psz_ctx* ctx, Buf* mem)
  {
    HFR_Opts opts;
    opts.reduce_times = 1;  // PBKGO restructed to be >1
    // --rmerge-count to ovrride
    if (ctx->cli and ctx->cli->hfr_rmerge_count > 0)
      opts.reduce_times = ctx->cli->hfr_rmerge_count;
    opts.magnitude = block_magnitude(ctx->header->len);
    opts.block_outliers = mem->block_outliers_d();
    return opts;
  }

  static int encode_pass1(psz_ctx* ctx, Buf* mem, void* stream)
  {
    const auto len_eq = eq_len(ctx->header->len);

    if constexpr (is_lc_pass1(pass1)) {
      mem->lc_wire_encoded(mem->compressed_d() + sizeof(psz_header));
      if constexpr (pass1 == LC_DRH)
        lc_c::DRH_COMPRESS(
            (uint8_t*)mem->eq_d(), len_eq * sizeof(E), mem->buf_lc(), &mem->comp_codec_outlen,
            stream);
      else
        lc_c::TCMS_COMPRESS(
            (uint8_t*)mem->eq_d(), len_eq * sizeof(E), mem->buf_lc(), &mem->comp_codec_outlen,
            stream);
      mem->comp_codec_out = mem->buf_lc()->encoded_d();
      mem->lc_wire_encoded(nullptr);
    }
    else if constexpr (pass1 == FZG) {
      if constexpr (std::is_same_v<E, u2>) {  // fzg::E is fixed at u2
        fzg_header dummy_header{};
        auto const status = fzg::high_level::encode(
            mem->buf_fzg(), mem->eq_d(), len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen,
            dummy_header, stream);
        sync_by_stream(stream);
        return status == 0 ? PSZ_SUCCESS : PSZ_ABORT_NO_SUCH_CODEC;
      }
      else
        return PSZ_ABORT_NO_SUCH_CODEC;
    }
    else {
      make_book(ctx, mem, len_eq, stream);

      phf_header dummy_header{};
      if constexpr (hfr)
        phf::high_level<E>::HFR_encode(
            mem->buf_hf(), mem->eq_d(), len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen,
            dummy_header, stream, pass1, nullptr, nullptr, encode_opts(ctx, mem));
      else
        phf::high_level<E>::HF_encode(
            mem->buf_hf(), mem->eq_d(), len_eq, &mem->comp_codec_out, &mem->comp_codec_outlen,
            dummy_header, stream, pass1);
    }
    return PSZ_SUCCESS;
  }

  static int compress(psz_ctx* ctx, Buf* mem, T* in, u1** out, size_t* outlen, void* stream)
  {
    int status = PSZ_SUCCESS;

    predict(ctx, mem, in, stream);

    status = encode_pass1(ctx, mem, stream);
    if (status != PSZ_SUCCESS) return status;

    status = pipeline_routine::set_splen(ctx, mem, stream);
    if (status != PSZ_SUCCESS) return status;

    status = set_entries(ctx, mem);
    if (status != PSZ_SUCCESS) return status;

    concat_segments(ctx, mem, stream);

    if constexpr (lc_pass2)
      ctx->header->entry[seg_pass2_end] =
          ctx->header->entry[pass2_head(C2::kind)] + encode_pass2(ctx, mem, stream);
    sync_by_stream(stream);

    pipeline_routine::publish(ctx, mem, out, outlen);

    return PSZ_SUCCESS;
  }

  static size_t encode_pass2(psz_ctx* ctx, Buf* mem, void* stream)
  {
    auto span = (uint8_t*)archive_at(ctx, mem, pass2_head(C2::kind));
    auto span_nbyte = ctx->header->entry[seg_pass1_end] - ctx->header->entry[pass2_head(C2::kind)];

    size_t nbyte;
    if constexpr (C2::kind == LC_BITR)
      lc_c::BITR_COMPRESS(span, span_nbyte, mem->buf_lc(), &nbyte, stream);
    else
      lc_c::RTR_COMPRESS(span, span_nbyte, mem->buf_lc(), &nbyte, stream);

    concat_on_device((void*)span, mem->buf_lc()->encoded_d(), nbyte, stream);
    return nbyte;
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

    ctx->header->entry[0] = 0;
    for (auto i = 1; i <= seg_pass2_end; i++)
      ctx->header->entry[i] = ctx->header->entry[i - 1] + mem->nbyte[i - 1];

    if (pszheader_filesize(ctx->header) > mem->compressed_max_bytes())
      return PSZ_ABORT_COMPRESSED_TOO_LARGE;
    return PSZ_SUCCESS;
  }

  static void concat_segments(psz_ctx* ctx, Buf* mem, void* stream)
  {
    concat_on_device(
        archive_at(ctx, mem, seg_anchor), mem->anchor_d(), mem->nbyte[seg_anchor], stream);
    if ((void*)mem->comp_codec_out != archive_at(ctx, mem, seg_encoded))
      concat_on_device(
          archive_at(ctx, mem, seg_encoded), mem->comp_codec_out, mem->nbyte[seg_encoded], stream);
    concat_on_device(
        archive_at(ctx, mem, seg_spfmt), mem->outlier2_validx_d(), mem->nbyte[seg_spfmt], stream);
  }

  // spl-y25 keeps eq in its own buffer; every other predictor decodes into the
  // output. 2D/3D eq is tile-ordered, and that decodes into decode_fused.
  static constexpr bool eq_in_out = not predicts_y25<P>::value;

  using Base = psz::compression_pipeline<T, E>;

  static int decompress(
      psz_header* header, Buf* mem, u1* in, T* out, void* stream, bool use_hfd_coarse = false)
  {
    auto access = [&](Segment seg) { return (void*)(in + header->entry[seg]); };
    auto const len = header->len;
    auto const len_linear = (size_t)len.x * len.y * len.z;
    int const nd = (len.z > 1) ? 3 : (len.y > 1) ? 2 : 1;
    bool const tile_nd = nd >= 2;

    auto d_anchor = (T*)access(seg_anchor);
    auto d_spvi = (_ptb::compact_cell<T, M>*)access(seg_spfmt);
    auto d_space = out;

    // eq lands wherever the predictor reads it from
    auto place_eq = [&](E* src, size_t n) {
      if (tile_nd)
        psz::module::GPU_cast<E, T>::kernel(src, mem->decode_fused_d(), n, stream);
      else if constexpr (eq_in_out)
        psz::module::GPU_cast<E, T>::kernel(src, d_space, n, stream);
      else
        concat_on_device(mem->eq_d(), src, n * sizeof(E), stream);
    };

    auto encoded = (BYTE*)access(seg_encoded);

    if constexpr (is_lc_pass1(pass1)) {
      if constexpr (pass1 == LC_DRH)
        lc_c::DRH_DECOMPRESS((uint8_t*)access(seg_encoded), mem->buf_lc(), stream);
      else
        lc_c::TCMS_DECOMPRESS((uint8_t*)access(seg_encoded), mem->buf_lc(), stream);
      place_eq((E*)mem->buf_lc()->decoded_d(), tile_nd ? mem->eq_len() : len_linear);
    }

    // a zero-byte span was never encoded
    if constexpr (lc_pass2) {
      constexpr auto head = pass2_head(C2::kind);
      if (header->entry[seg_pass2_end] != header->entry[head]) {
        if constexpr (C2::kind == LC_BITR)
          lc_c::BITR_DECOMPRESS((uint8_t*)access(head), mem->buf_lc(), stream);
        else
          lc_c::RTR_DECOMPRESS((uint8_t*)access(head), mem->buf_lc(), stream);

        auto staged = (byte_t*)mem->buf_lc()->decoded_d();
        auto decoded = [&](Segment seg) {
          return staged + (header->entry[seg] - header->entry[head]);
        };
        if constexpr (head == seg_encoded) encoded = (BYTE*)decoded(seg_encoded);
        d_anchor = (T*)decoded(seg_anchor);
        d_spvi = (_ptb::compact_cell<T, M>*)decoded(seg_spfmt);
      }
    }

    if constexpr (pass1 == FZG) {
      if constexpr (std::is_same_v<E, u2>) {
        fzg_header h_fzg;
        fzg::high_level::decode(
            mem->buf_fzg(), h_fzg, (uint8_t*)encoded, 0, mem->fzg_scratch_d(), mem->eq_len(),
            stream);
        place_eq(mem->fzg_scratch_d(), mem->eq_len());
      }
    }
    else if constexpr (not is_lc_pass1(pass1)) {
      phf_header hdr{};
      memcpy_allkinds<D2H>((BYTE*)&hdr, encoded, sizeof(phf_header));

      auto decode_eq = [&](auto* dst) -> int {
        using Eout = std::remove_pointer_t<decltype(dst)>;
        if constexpr (hfr) {
          auto const magnitude = block_magnitude(len);
          if (not use_hfd_coarse)
            return phf::high_level<E>::template HFD26_decode<Eout>(
                mem->buf_hf(), hdr, encoded, dst, stream, pass1, magnitude);
          return phf::high_level<E>::template HFR_decode<Eout>(
              mem->buf_hf(), hdr, encoded, dst, stream, pass1, magnitude);
        }
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
        stat = decode_eq(mem->eq_d());
      if (stat != PHF_SUCCESS) return PSZ_ABORT_NO_SUCH_CODEC;
    }

    if (tile_nd and header->splen != 0)
      psz::module::GPU_scatter<T, M>::kernel_v3_fuse(
          d_spvi, header->splen, mem->decode_fused_d(), stream);
    else
      Base::decomp_scatter(header, d_spvi, d_space, stream);

    Base::decomp_predict(header, mem, d_anchor, out, stream);

    return PSZ_SUCCESS;
  }
};

namespace _2609 {

template <typename T, typename E>
struct dispatch {
  static bool pipeline_supported(psz_ppl const& p) { return pszppl_supported(p); }

  template <class P, class C1, class F>
  static bool route_codec2(psz_ppl const& p, F&& f)
  {
    // a pipeline the machine cannot walk is never instantiated
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
      case HF:
      case HF_r2: return route_codec2<P, ModuleCodec1<HF_r2>>(p, f);
      case HFR: return route_codec2<P, ModuleCodec1<HFR>>(p, f);
      case HFR_V2: return route_codec2<P, ModuleCodec1<HFR_V2>>(p, f);
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

}  // namespace _2609

}  // namespace psz

#endif /* PSZ_COMPRESSOR2_INL */
