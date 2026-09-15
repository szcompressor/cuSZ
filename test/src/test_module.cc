#include "pipeline.h"
#include <cstdio>
#include <type_traits>

#include "module.hh"

using psz::PredictorFeature;
using psz::_2609::ModuleCodec1;
using psz::_2609::ModuleCodec2;
using psz::_2609::ModuleLorenzo;
using psz::_2609::ModuleSplineY24;
using psz::_2609::ModuleSplineY25;
using psz::_2609::needs_book;
using psz::_2609::Pipeline;

static_assert(ModuleLorenzo<>::Features::UseZigZag == 0 and not ModuleLorenzo<>::spline);
static_assert(ModuleLorenzo<PredictorFeature<1>>::Features::UseZigZag == 1);
static_assert(ModuleSplineY24<>::spline and ModuleSplineY24<>::kind == SplineY24);
static_assert(ModuleSplineY25<>::spline and ModuleSplineY25<>::kind == SplineY25);
static_assert(not std::is_same_v<ModuleSplineY24<>, ModuleSplineY25<>>);

using Default = Pipeline<ModuleLorenzo<>, ModuleCodec1<HF>>;
static_assert(Default::Codec1::kind == HF and Default::Codec2::kind == CodecNull);
static_assert(Default::legacy and not Default::modern);

using HiTP =
    Pipeline<ModuleSplineY24<>, ModuleCodec1<LC_TCMS>, ModuleCodec2<LC_BITR>>;
static_assert(HiTP::Codec2::kind == LC_BITR and HiTP::legacy);

static_assert(Pipeline<ModuleLorenzo<>, ModuleCodec1<HFR>>::modern);
static_assert(
    Pipeline<ModuleLorenzo<>, ModuleCodec1<HFR_PBKC>>::modern);
static_assert(
    Pipeline<ModuleLorenzo<>, ModuleCodec1<HFR_PBKGO>>::modern);
static_assert(
    Pipeline<ModuleLorenzo<>, ModuleCodec1<HFR_V3>>::modern);
static_assert(
    Pipeline<ModuleLorenzo<>, ModuleCodec1<HFR_V4>>::modern);

// psz_preset and the preset aliases must name the same pipelines
template <class P>
static bool agrees(psz_preset e, char const* name)
{
  auto const ppl = pszpreset_pipeline(e);
  bool ok = ppl.predictor == psz::_2609::predictor_kind<typename P::Predictor>() and
            ppl.hist == (needs_book(P::Codec1::kind) ? HistGeneric : HistNull) and ppl.codec1 == P::Codec1::kind and
            ppl.codec2 == P::Codec2::kind;

  if (not ok) fprintf(stderr, "preset does not match its alias: %s\n", name);
  return ok;
}

#define AGREES(ALIAS, ENUM) agrees<psz::_2609::ALIAS>(ENUM, #ALIAS)

// the walk, decided at compile time
using ZigZagHFR =
    Pipeline<ModuleLorenzo<PredictorFeature<1>>, ModuleCodec1<HFR_V4>>;
using LrzFZG = Pipeline<ModuleLorenzo<>, ModuleCodec1<FZG>>;
using ZigZagFZG =
    Pipeline<ModuleLorenzo<PredictorFeature<1>>, ModuleCodec1<FZG>>;
using HFRunderLC = Pipeline<
    ModuleSplineY25<>, ModuleCodec1<HFR_V4>, ModuleCodec2<LC_RTR>>;

static_assert(psz::_2609::PresetHiCR::valid);
static_assert(psz::_2609::PresetHiTP::valid);
static_assert(HFRunderLC::valid);
static_assert(ZigZagFZG::valid);
static_assert(not ZigZagHFR::valid);
static_assert(not LrzFZG::valid);

using psz::_2609::component;

// the walk: each next() is legal only where the machine has an edge
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

  // a generic shape names a stage count, not a pipeline
  ok = pszpreset_is_generic(PSZ_PRESET_P1_C1) and ok;
  ok = pszpreset_is_generic(PSZ_PRESET_P1_C1_C2) and ok;
  ok = (not pszpreset_is_generic(PSZ_PRESET_HICR)) and ok;

  // compose then classify lands back on the shape that was asked for
  ok = (pszppl_preset(pszppl_compose(Lorenzo, HFR_V4, CodecNull)) == PSZ_PRESET_P1_C1) and ok;
  ok = (pszppl_preset(pszppl_compose(SplineY25, HF_r2, LC_TCMS)) == PSZ_PRESET_HICR) and ok;
  ok = (pszppl_preset(pszppl_compose(SplineY25, LC_TCMS, LC_TCMS)) == PSZ_PRESET_HITP) and ok;
  ok = (pszppl_preset(pszppl_compose(SplineY25, LC_DRH, LC_TCMS)) == PSZ_PRESET_HITP_R1) and ok;
  ok = (pszppl_preset(pszppl_compose(LorenzoZigZag, FZG, CodecNull)) == PSZ_PRESET_LRZZZ_FZG) and ok;

  // the composer fills in what a codec needs, and resolves a pass 1 an LC pass 2 can read
  ok = (pszppl_compose(Lorenzo, HFR_PBKC, CodecNull).hist == HistNull) and ok;
  ok = (pszppl_compose(Lorenzo, HFR_V4, CodecNull).hist == HistGeneric) and ok;
  ok = (pszppl_compose(SplineY25, HFR_V4, LC_TCMS).codec1 == HFR_V4) and ok;
  ok = pszppl_supported(pszppl_compose(SplineY25, HFR_V4, LC_TCMS)) and ok;

  // and says nothing about validity, which is its own question
  ok = (not pszppl_supported(pszppl_compose(LorenzoZigZag, HFR_V4, CodecNull))) and ok;
  ok = (not pszppl_supported(pszppl_compose(Lorenzo, FZG, CodecNull))) and ok;
  ok = pszppl_supported(pszppl_compose(LorenzoZigZag, FZG, CodecNull)) and ok;

  // the compile-time walk and the runtime rule must agree
  ok = (ZigZagHFR::valid ==
        (bool)pszppl_supported(pszppl_compose(LorenzoZigZag, HFR_V4, CodecNull))) and ok;
  ok = (LrzFZG::valid ==
        (bool)pszppl_supported(pszppl_compose(Lorenzo, FZG, CodecNull))) and ok;
  ok = (ZigZagFZG::valid ==
        (bool)pszppl_supported(pszppl_compose(LorenzoZigZag, FZG, CodecNull))) and ok;
  ok = (HFRunderLC::valid ==
        (bool)pszppl_supported(pszppl_compose(SplineY25, HFR_V4, LC_TCMS))) and ok;

  // a pass-2 chain is an LC codec, but never a codec 1
  ok = psz::_2609::is_lc(LC_BITR) and psz::_2609::is_lc(LC_RTR) and ok;
  ok = psz::_2609::is_lc(LC_TCMS) and psz::_2609::is_lc(LC_DRH) and ok;
  ok = (not psz::_2609::is_lc_pass1(LC_RTR)) and psz::_2609::is_lc_pass1(LC_TCMS) and ok;
  ok = (not pszppl_supported(pszppl_compose(Lorenzo, LC_RTR, CodecNull))) and ok;
  ok = (not pszppl_supported(pszppl_compose(Lorenzo, LC_BITR, CodecNull))) and ok;

  // the edge table and the C support rule must agree over the whole space
  {
    psz_predictor const p1s[] = {Lorenzo, LorenzoZigZag, SplineY24, SplineY25};
    psz_codec const cs[] = {HF,       HF_r2,     HFR,       HFR_V2,   HFR_V3,  HFR_V4,  HFR_PBKC,
                            HFR_PBKGO, HFR_PBKF, LC_TCMS,  LC_DRH,   LC_BITR, LC_RTR,  FZG,
                            CodecNull};
    int checked = 0;
    for (auto p1 : p1s)
      for (auto c1 : cs)
        for (auto c2 : cs) {
          bool const table = psz::_2609::_p1_to_c1(p1, c1) and psz::_2609::_c1_to_c2(c1, c2);
          psz_ppl ppl{p1, HistGeneric, c1, c2};
          bool const rule = pszppl_supported(ppl) != 0;
          if (table != rule) {
            fprintf(stderr, "edge table and support rule differ: p1=%d c1=%d c2=%d\n", p1, c1, c2);
            ok = false;
          }
          checked++;
        }
    if (checked != 900) ok = false;
  }

  // only the HFR encoders pin the radius, and they pin it to the prebuilt books
  ok = (pszpreset_radius(PSZ_PRESET_HICR) == 512) and ok;
  ok = (pszpreset_radius(PSZ_PRESET_HITP) == 512) and ok;

  return ok ? 0 : 1;
}
