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

constexpr bool is_lc_pass1(psz_codec c)
{
  switch (c) {
    case LC_TCMS:
    case LC_DRH: return true;
    default: return false;
  }
}

constexpr bool is_lc_pass2(psz_codec c)
{
  switch (c) {
    case LC_BITR:
    case LC_RTR: return true;
    default: return false;
  }
}

constexpr bool is_lc(psz_codec c) { return is_lc_pass1(c) or is_lc_pass2(c); }

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
  bool before_pass2;  // may be followed by an LC pass 2
};

constexpr CodecEdge _codec_edges[] = {
    {HF, true, true, true},        {HF_r2, true, true, true},      {HFR, false, true, true},
    {HFR_V2, false, true, true},   {HFR_V3, false, true, true},    {HFR_V4, false, true, true},
    {HFR_PBKC, false, true, true}, {HFR_PBKGO, false, true, true}, {HFR_PBKF, false, true, true},
    {LC_TCMS, true, true, true},   {LC_DRH, true, true, true},     {LC_BITR, false, false, false},
    {LC_RTR, false, false, false}, {FZG, true, false, false},
};

constexpr CodecEdge _egress_edge(psz_codec c)
{
  for (auto const& e : _codec_edges)
    if (e.codec == c) return e;
  return {c, false, false, false};
}

constexpr bool _p1_to_c1(psz_predictor p1, psz_codec c1)
{
  auto const e = _egress_edge(c1);
  return p1 == LorenzoZigZag ? e.after_zigzag : e.after_plain;
}

constexpr bool _c1_to_c2(psz_codec c1, psz_codec c2)
{
  if (c2 == CodecNull) return true;
  if (not is_lc_pass2(c2)) return false;
  return _egress_edge(c1).before_pass2;
}

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
