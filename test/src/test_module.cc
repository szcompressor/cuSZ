#include <cstdio>
#include <type_traits>

#include "module.hh"

using psz::PredictorFeature;
using psz::_2609::compose;
using psz::_2609::is_generic;
using psz::_2609::ModuleCodec1;
using psz::_2609::ModuleCodec2;
using psz::_2609::ModuleLorenzo;
using psz::_2609::ModuleSplineY24;
using psz::_2609::ModuleSplineY25;
using psz::_2609::needs_book;
using psz::_2609::Pipeline;
using psz::_2609::preset_of;
using psz::_2609::valid;

static_assert(ModuleLorenzo<>::Features::UseZigZag == 0 and not ModuleLorenzo<>::spline);
static_assert(ModuleLorenzo<PredictorFeature<1>>::Features::UseZigZag == 1);
static_assert(ModuleSplineY24<>::spline and ModuleSplineY24<>::kind == SplineY24);
static_assert(ModuleSplineY25<>::spline and ModuleSplineY25<>::kind == SplineY25);
static_assert(not std::is_same_v<ModuleSplineY24<>, ModuleSplineY25<>>);

using Default = Pipeline<ModuleLorenzo<>, ModuleCodec1<HF>>;
static_assert(Default::Codec1::kind == HF and Default::Codec2::kind == CodecNull);
static_assert(Default::legacy and not Default::modern);

using HiTP = Pipeline<ModuleSplineY24<>, ModuleCodec1<LC_TCMS>, ModuleCodec2<LC_BITR>>;
static_assert(HiTP::Codec2::kind == LC_BITR and HiTP::legacy);

static_assert(Pipeline<ModuleLorenzo<>, ModuleCodec1<HFR>>::modern);
static_assert(Pipeline<ModuleLorenzo<>, ModuleCodec1<HFR_PBKC>>::modern);
static_assert(Pipeline<ModuleLorenzo<>, ModuleCodec1<HFR_PBKGO>>::modern);
static_assert(Pipeline<ModuleLorenzo<>, ModuleCodec1<HFR_V3>>::modern);
static_assert(Pipeline<ModuleLorenzo<>, ModuleCodec1<HFR_V4>>::modern);

template <class P>
static bool agrees(psz_preset e, char const* name)
{
  auto const ppl = psz::_2609::pipeline_of(e);
  bool ok = ppl.predictor == psz::_2609::predictor_kind<typename P::Predictor>() and
            ppl.hist == (needs_book(P::Codec1::kind) ? HistGeneric : HistNull) and
            ppl.codec1 == P::Codec1::kind and ppl.codec2 == P::Codec2::kind;

  if (not ok) fprintf(stderr, "preset does not match its alias: %s\n", name);
  return ok;
}

#define AGREES(ALIAS, ENUM) agrees<psz::_2609::ALIAS>(ENUM, #ALIAS)

using ZigZagHFR = Pipeline<ModuleLorenzo<PredictorFeature<1>>, ModuleCodec1<HFR_V4>>;
using LrzFZG = Pipeline<ModuleLorenzo<>, ModuleCodec1<FZG>>;
using ZigZagFZG = Pipeline<ModuleLorenzo<PredictorFeature<1>>, ModuleCodec1<FZG>>;
using HFRunderLC = Pipeline<ModuleSplineY25<>, ModuleCodec1<HFR_V4>, ModuleCodec2<LC_RTR>>;

static_assert(psz::_2609::PresetHiCR::valid);
static_assert(psz::_2609::PresetHiTP::valid);
static_assert(HFRunderLC::valid);
static_assert(ZigZagFZG::valid);
static_assert(not ZigZagHFR::valid);
static_assert(not LrzFZG::valid);

using psz::_2609::component;

using WalkHiCR = decltype(component<ModuleSplineY25<>>{}
                              .next<ModuleCodec1<HF_r2>>()
                              .next<ModuleCodec2<LC_RTR>>())::pipeline;
using WalkHiTP = decltype(component<ModuleSplineY25<>>{}
                              .next<ModuleCodec1<LC_TCMS>>()
                              .next<ModuleCodec2<LC_BITR>>())::pipeline;
using WalkFZG =
    decltype(component<ModuleLorenzo<PredictorFeature<1>>>{}.next<ModuleCodec1<FZG>>())::pipeline;

static_assert(std::is_same_v<WalkHiCR, psz::_2609::PresetHiCR>);
static_assert(std::is_same_v<WalkHiTP, psz::_2609::PresetHiTP>);
static_assert(WalkHiCR::valid and WalkHiTP::valid and WalkFZG::valid);

int main()
{
  bool ok = true;

  ok = AGREES(PresetLrzFZG, PSZ_PRESET_LRZZZ_FZG) and ok;
  ok = AGREES(PresetHiCR, PSZ_PRESET_HICR) and ok;
  ok = AGREES(PresetHiTP, PSZ_PRESET_HITP) and ok;
  ok = AGREES(PresetHiTP_r1, PSZ_PRESET_HITP_R1) and ok;

  ok = is_generic(PSZ_PRESET_P1_C1) and ok;
  ok = is_generic(PSZ_PRESET_P1_C1_C2) and ok;
  ok = (not is_generic(PSZ_PRESET_HICR)) and ok;

  ok = (preset_of(compose(Lorenzo, HFR_V4, CodecNull)) == PSZ_PRESET_P1_C1) and ok;
  ok = (preset_of(compose(SplineY25, HF_r2, LC_TCMS)) == PSZ_PRESET_HICR) and ok;
  ok = (preset_of(compose(SplineY25, LC_TCMS, LC_TCMS)) == PSZ_PRESET_HITP) and ok;
  ok = (preset_of(compose(SplineY25, LC_DRH, LC_TCMS)) == PSZ_PRESET_HITP_R1) and ok;
  ok = (preset_of(compose(LorenzoZigZag, FZG, CodecNull)) == PSZ_PRESET_LRZZZ_FZG) and ok;

  ok = (compose(Lorenzo, HFR_PBKC, CodecNull).hist == HistNull) and ok;
  ok = (compose(Lorenzo, HFR_V4, CodecNull).hist == HistGeneric) and ok;
  ok = (compose(SplineY25, HFR_V4, LC_TCMS).codec1 == HFR_V4) and ok;
  ok = valid(compose(SplineY25, HFR_V4, LC_TCMS)) and ok;

  ok = (not valid(compose(LorenzoZigZag, HFR_V4, CodecNull))) and ok;
  ok = (not valid(compose(Lorenzo, FZG, CodecNull))) and ok;
  ok = valid(compose(LorenzoZigZag, FZG, CodecNull)) and ok;

  ok = (ZigZagHFR::valid == valid(compose(LorenzoZigZag, HFR_V4, CodecNull))) and ok;
  ok = (LrzFZG::valid == valid(compose(Lorenzo, FZG, CodecNull))) and ok;
  ok = (ZigZagFZG::valid == valid(compose(LorenzoZigZag, FZG, CodecNull))) and ok;
  ok = (HFRunderLC::valid == valid(compose(SplineY25, HFR_V4, LC_TCMS))) and ok;

  ok = psz::_2609::is_lc(LC_BITR) and psz::_2609::is_lc(LC_RTR) and ok;
  ok = psz::_2609::is_lc(LC_TCMS) and psz::_2609::is_lc(LC_DRH) and ok;
  ok = psz::_2609::_egress_edge(LC_RTR).as_codec2 and
       (not psz::_2609::_egress_edge(LC_TCMS).as_codec2) and ok;
  ok = (not valid(compose(Lorenzo, LC_RTR, CodecNull))) and ok;
  ok = (not valid(compose(Lorenzo, LC_BITR, CodecNull))) and ok;

  {
    psz_predictor const p1s[] = {Lorenzo, LorenzoZigZag, SplineY24, SplineY25};
    psz_codec const cs[] = {HF,     HF_r2,    HFR,       HFR_V2,   HFR_V3,
                            HFR_V4, HFR_PBKC, HFR_PBKGO, HFR_PBKF, LC_TCMS,
                            LC_DRH, LC_BITR,  LC_RTR,    FZG,      CodecNull};
    int checked = 0;
    for (auto p1 : p1s)
      for (auto c1 : cs)
        for (auto c2 : cs) {
          bool const table = psz::_2609::_p1_to_c1(p1, c1) and psz::_2609::_c1_to_c2(c1, c2);
          psz_ppl ppl{p1, HistGeneric, c1, c2};
          bool const rule = valid(ppl);
          if (table != rule) {
            fprintf(stderr, "edge table and support rule differ: p1=%d c1=%d c2=%d\n", p1, c1, c2);
            ok = false;
          }
          checked++;
        }
    if (checked != 900) ok = false;
  }

  ok = (psz::_2609::radius_of(PSZ_PRESET_HICR) == 512) and ok;
  ok = (psz::_2609::radius_of(PSZ_PRESET_HITP) == 512) and ok;

  return ok ? 0 : 1;
}
