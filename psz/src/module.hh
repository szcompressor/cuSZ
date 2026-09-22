#ifndef PSZ_MODULE_HH
#define PSZ_MODULE_HH

#include "component.hh"
#include "cusz/type.h"

namespace psz::_2609 {

constexpr bool is_hfr(psz_codec c)
{
  switch (c) {
    case HFR:
    case HFR_V2:
    case HFR_V3:
    case HFR_V4:
    case HFR_PBKC:
    case HFR_PBKGO:
    case HFR_PBKF: return true;
    default: return false;
  }
}

constexpr bool is_lc(psz_codec c)
{
  switch (c) {
    case LC_TCMS:
    case LC_DRH:
    case LC_BITR:
    case LC_RTR: return true;
    default: return false;
  }
}

constexpr bool is_spline(psz_predictor p) { return p == SplineY24 or p == SplineY25; }

constexpr bool needs_book(psz_codec c)
{
  switch (c) {
    case HF:
    case HF_r2:
    case HFR:
    case HFR_V3:
    case HFR_V4: return true;
    default: return false;
  }
}

constexpr bool unpred_localized(psz_codec c)
{
  switch (c) {
    case HFR:
    case HFR_V3:
    case HFR_V4:
    case HFR_PBKC:
    case HFR_PBKGO: return true;
    default: return false;
  }
}

constexpr bool unpred_spill(psz_codec c)
{
  switch (c) {
    case HF:
    case HF_r2:
    case FZG:
    case LC_TCMS:
    case LC_DRH: return true;
    default: return false;
  }
}

enum Segment : int {
  seg_header,
  seg_encoded,
  seg_anchor,
  seg_spfmt,
  seg_pass1_end,
  seg_pass2_end,
};

constexpr size_t pad8(size_t n) { return (n + 7) & ~(size_t)7; }

constexpr Segment pass2_head(psz_codec codec2)
{ return codec2 == LC_RTR ? seg_encoded : seg_anchor; }

template <class _Features = PredictorFeature<0>>
struct ModuleLorenzo {
  using Features = _Features;
  static constexpr bool spline = false;
  static constexpr psz_predictor kind = Lorenzo;
};

template <class _Features = PredictorFeature<0>>
struct ModuleSplineY24 {
  using Features = _Features;
  static constexpr bool spline = true;
  static constexpr psz_predictor kind = SplineY24;
};

template <class _Features = PredictorFeature<0>>
struct ModuleSplineY25 {
  using Features = _Features;
  static constexpr bool spline = true;
  static constexpr psz_predictor kind = SplineY25;
};

template <psz_codec _Kind>
struct ModuleCodec1 {
  static constexpr psz_codec kind = _Kind;
};

template <psz_codec _Kind = CodecNull>
struct ModuleCodec2 {
  static constexpr psz_codec kind = _Kind;
};

template <class P>
constexpr psz_predictor predictor_kind()
{
  if constexpr (not P::spline and P::Features::UseZigZag == 1)
    return LorenzoZigZag;
  else
    return P::kind;
}

struct CodecEdge {
  psz_codec codec;
  bool after_zigzag;  // may follow lrz-zz
  bool after_plain;   // may follow lrz, spl-y24, spl-y25
  bool before_codec2;  // may be followed by a second codec
  bool as_codec2;      // may occupy codec 2
};

constexpr CodecEdge _codec_edges[] = {
    {HF, true, true, true, false},        {HF_r2, true, true, true, false},
    {HFR, false, true, true, false},      {HFR_V2, false, true, true, false},
    {HFR_V3, false, true, true, false},   {HFR_V4, false, true, true, false},
    {HFR_PBKC, false, true, true, false}, {HFR_PBKGO, false, true, true, false},
    {HFR_PBKF, false, true, true, false}, {LC_TCMS, true, true, true, false},
    {LC_DRH, true, true, true, false},    {LC_BITR, false, false, false, true},
    {LC_RTR, false, false, false, true},  {FZG, true, false, false, false},
    {CodecNull, true, true, false, false},
};

constexpr CodecEdge _egress_edge(psz_codec c)
{
  for (auto const& e : _codec_edges)
    if (e.codec == c) return e;
  return {c, false, false, false, false};
}

constexpr bool _p1_to_c1(psz_predictor p1, psz_codec c1)
{
  auto const e = _egress_edge(c1);
  return p1 == LorenzoZigZag ? e.after_zigzag : e.after_plain;
}

constexpr bool _c1_to_c2(psz_codec c1, psz_codec c2)
{
  if (c2 == CodecNull) return true;
  return _egress_edge(c2).as_codec2 and _egress_edge(c1).before_codec2;
}

constexpr bool valid(psz_ppl p)
{ return _p1_to_c1(p.predictor, p.codec1) and _c1_to_c2(p.codec1, p.codec2); }

constexpr psz_ppl compose(psz_predictor p1, psz_codec c1, psz_codec optional_c2)
{
  psz_ppl ppl{};
  ppl.predictor = p1;
  ppl.codec1 = c1;
  ppl.codec2 = optional_c2;

  if (optional_c2 == LC_TCMS) ppl.codec2 = is_lc(c1) ? LC_BITR : LC_RTR;

  ppl.hist = needs_book(ppl.codec1) ? HistGeneric : HistNull;

  return ppl;
}

constexpr psz_preset preset_of(psz_ppl p)
{
  if (p.codec1 == FZG) return PSZ_PRESET_LRZZZ_FZG;
  if (is_spline(p.predictor) and p.codec2 != CodecNull) {
    if (p.codec1 == LC_TCMS) return PSZ_PRESET_HITP;
    if (p.codec1 == LC_DRH) return PSZ_PRESET_HITP_R1;
    return PSZ_PRESET_HICR;
  }
  return p.codec2 != CodecNull ? PSZ_PRESET_P1_C1_C2 : PSZ_PRESET_P1_C1;
}

constexpr bool is_generic(psz_preset p) { return p == PSZ_PRESET_P1_C1 or p == PSZ_PRESET_P1_C1_C2; }

constexpr psz_ppl pipeline_of(psz_preset p)
{
  psz_ppl ppl{};

  ppl.predictor = (p == PSZ_PRESET_LRZZZ_FZG) ? LorenzoZigZag : SplineY25;
  ppl.hist = (p == PSZ_PRESET_HICR) ? HistGeneric : HistNull;

  switch (p) {
    case PSZ_PRESET_LRZZZ_FZG: ppl.codec1 = FZG; break;
    case PSZ_PRESET_HITP: ppl.codec1 = LC_TCMS; break;
    case PSZ_PRESET_HITP_R1: ppl.codec1 = LC_DRH; break;
    default: ppl.codec1 = HF_r2; break;  // HF_r2 supersedes HF
  }

  if (p == PSZ_PRESET_HICR)
    ppl.codec2 = LC_RTR;
  else if (p == PSZ_PRESET_HITP or p == PSZ_PRESET_HITP_R1)
    ppl.codec2 = LC_BITR;
  else
    ppl.codec2 = CodecNull;
  return ppl;
}

constexpr bool needs_eq4(psz_ppl p)
{
  switch (p.codec1) {
    case HFR:
    case HFR_PBKC:
    case HFR_PBKGO:
    case HFR_V3:
    case HFR_V4: return true;
    default: return false;
  }
}

constexpr int radius_of(psz_ppl p) { return needs_eq4(p) ? 128 : 512; }

constexpr int radius_of(psz_preset p) { return radius_of(pipeline_of(p)); }

constexpr int nstage_of(psz_ppl p) { return p.codec1 == CodecNull ? 1 : p.codec2 == CodecNull ? 2 : 3; }

template <class _Predictor, class _Codec1, class _Codec2 = ModuleCodec2<>>
struct Pipeline {
  using Predictor = _Predictor;
  using Codec1 = _Codec1;
  using Codec2 = _Codec2;

  static constexpr bool valid = _p1_to_c1(predictor_kind<_Predictor>(), Codec1::kind) and
                                _c1_to_c2(Codec1::kind, Codec2::kind);

  static constexpr bool modern = is_hfr(Codec1::kind);
  static constexpr bool legacy = not modern;
};

// clang-format off
// historical pipelines (_r1 is an ad hoc test)
using PresetLrzFZG  = Pipeline<ModuleLorenzo<PredictorFeature<1>>, ModuleCodec1<FZG>>;
using PresetHiCR    = Pipeline<ModuleSplineY25<>, ModuleCodec1<HF_r2>,   ModuleCodec2<LC_RTR>>;
using PresetHiTP    = Pipeline<ModuleSplineY25<>, ModuleCodec1<LC_TCMS>, ModuleCodec2<LC_BITR>>;
using PresetHiTP_r1 = Pipeline<ModuleSplineY25<>, ModuleCodec1<LC_DRH>,  ModuleCodec2<LC_BITR>>;
// alias
using PresetDefault = Pipeline<ModuleLorenzo<>, ModuleCodec1<HFR_PBKC>>;
// clang-format on

template <class... Stage>
struct component;

template <class P>
struct component<P> {
  template <class C1>
  constexpr component<P, C1> next() const
  {
    static_assert(Pipeline<P, C1>::valid, "no edge from this predictor to this codec");
    return {};
  }
};

template <class P, class C1>
struct component<P, C1> {
  using pipeline = Pipeline<P, C1>;

  template <class C2>
  constexpr component<P, C1, C2> next() const
  {
    static_assert(Pipeline<P, C1, C2>::valid, "no edge from this codec to this pass 2");
    return {};
  }
};

template <class P, class C1, class C2>
struct component<P, C1, C2> {
  using pipeline = Pipeline<P, C1, C2>;
};

}  // namespace psz::_2609

#endif /* PSZ_MODULE_HH */
