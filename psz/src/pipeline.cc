// 26-09-12

#include "pipeline.h"

#include "module.hh"

using psz::_2609::_c1_to_c2;
using psz::_2609::_p1_to_c1;
using psz::_2609::is_lc_pass2;
using psz::_2609::is_spline;
using psz::_2609::needs_book;

const char* psz_error_string(int e)
{
  switch (e) {
    case PSZ_SUCCESS: return "success";
    case PSZ_WARN_RADIUS_TOO_LARGE: return "radius too large";
    case PSZ_WARN_OUTLIER_TOO_MANY: return "more outliers than the outlier buffer holds";
    case PSZ_ABORT_UNSUPPORTED_TYPE: return "unsupported data type";
    case PSZ_ABORT_UNSUPPORTED_DIMENSION: return "unsupported dimensionality";
    case PSZ_ABORT_NOT_IMPLEMENTED: return "not implemented";
    case PSZ_ABORT_NO_SUCH_PREDICTOR: return "no such predictor";
    case PSZ_ABORT_NO_SUCH_CODEC: return "no such codec";
    case PSZ_ABORT_TOO_MANY_UNPREDICTABLE: return "too many unpredictables";
    case PSZ_ABORT_TOO_MANY_ENC_BREAK: return "too many encoding breaks";
    case PSZ_ABORT_COMPRESSED_TOO_LARGE: return "compressed size exceeds the output buffer";
    case PSZ_ABORT_UNSUPPORTED_PIPELINE: return "unsupported pipeline";
    default: return "unknown error";
  }
}

int pszpredictor_is_spline(psz_predictor p) { return is_spline(p); }

int pszcodec_is_pass2(psz_codec c) { return is_lc_pass2(c); }

int pszppl_supported(psz_ppl p)
{
  return _p1_to_c1(p.predictor, p.codec1) and _c1_to_c2(p.codec1, p.codec2);
}

psz_ppl pszppl_compose(psz_predictor p1, psz_codec c1, psz_codec optional_c2)
{
  psz_ppl ppl;
  ppl.predictor = p1;
  ppl.codec1 = c1;
  ppl.codec2 = optional_c2;

  if (optional_c2 == LC_TCMS) ppl.codec2 = (c1 == LC_TCMS or c1 == LC_DRH) ? LC_BITR : LC_RTR;

  ppl.hist = needs_book(ppl.codec1) ? HistGeneric : HistNull;

  return ppl;
}

psz_preset pszppl_preset(psz_ppl p)
{
  if (p.codec1 == FZG) return PSZ_PRESET_LRZZZ_FZG;
  if (pszpredictor_is_spline(p.predictor) and pszcodec_is_pass2(p.codec2)) {
    if (p.codec1 == LC_TCMS) return PSZ_PRESET_HITP;
    if (p.codec1 == LC_DRH) return PSZ_PRESET_HITP_R1;
    return PSZ_PRESET_HICR;
  }
  return pszcodec_is_pass2(p.codec2) ? PSZ_PRESET_P1_C1_C2 : PSZ_PRESET_P1_C1;
}

int pszpreset_is_generic(psz_preset p)
{ return p == PSZ_PRESET_P1_C1 or p == PSZ_PRESET_P1_C1_C2; }

psz_ppl pszpreset_pipeline(psz_preset p)
{
  psz_ppl ppl;

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

int pszppl_needs_eq4(psz_ppl p)
{
  if (p.codec1 == FZG) return 0;
  switch (p.codec1) {
    case HFR:
    case HFR_PBKC:
    case HFR_PBKGO:
    case HFR_V3:
    case HFR_V4: return 1;
    default: return 0;
  }
}

int pszpreset_needs_eq4(psz_preset p) { return pszppl_needs_eq4(pszpreset_pipeline(p)); }

int pszppl_radius(psz_ppl p) { return pszppl_needs_eq4(p) ? 128 : 512; }

int pszpreset_radius(psz_preset p) { return pszppl_radius(pszpreset_pipeline(p)); }
