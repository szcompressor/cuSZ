#ifndef PSZ_PIPELINE_H
#define PSZ_PIPELINE_H

#include "cusz/type.h"

#ifdef __cplusplus
extern "C" {
#endif

int pszpredictor_is_spline(psz_predictor p);

int pszcodec_is_pass2(psz_codec c);

int pszppl_supported(psz_ppl p);

psz_ppl pszppl_compose(psz_predictor p1, psz_codec c1, psz_codec optional_c2);

psz_preset pszppl_preset(psz_ppl p);

int pszpreset_is_generic(psz_preset p);

psz_ppl pszpreset_pipeline(psz_preset p);

int pszppl_needs_eq4(psz_ppl p);

int pszpreset_needs_eq4(psz_preset p);

int pszppl_radius(psz_ppl p);

int pszpreset_radius(psz_preset p);

#ifdef __cplusplus
}
#endif

#endif /* PSZ_PIPELINE_H */
