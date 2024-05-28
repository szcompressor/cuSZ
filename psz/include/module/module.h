#ifndef B140A45C_CB00_4A4F_B9BC_F84ABB108FD9
#define B140A45C_CB00_4A4F_B9BC_F84ABB108FD9

#include "cusz/array.h"

// modular functions as of 2401
// clang-format off
pszerror psz_predict_lorenzo(psz_carray in, psz_rc const rc, psz_carray out_errquant, psz_outlier out_outlier, f4*void* stream);

/**/pszerror psz_predict_spline(psz_carray in, psz_rc const rc, psz_carray out_errquant, psz_carray out_anchor, psz_outlier out_outlier, void* stream);
/**/pszerror psz_histogram(psz_carray in, psz_carray hist, void* stream);
/**/pszerror psz_encode_entropy(psz_carray in, psz_carray out_encoded, void* stream);
/**/pszerror psz_encode_dictionary(pszcompressor_stream cor, psz_carray in, psz_carray out_encoded, void* stream);
/**/pszerror psz_archive(pszcompressor_stream cor, psz_carray in_encoded, psz_outlier in_scattered_outlier, psz_header* out_header, psz_carray out_archive, void* stream);

/**/pszerror psz_unarchive(psz_header* in_header, psz_carray in_archive, psz_carray out_encoded, psz_outlier out_outlier, void* stream);
/**/pszerror psz_decode_entropy(psz_carray in_encoded, psz_carray out_decoded, void* stream);
/**/pszerror psz_decode_dictionary(psz_carray in_encoded, psz_carray out_decoded, void* stream);
pszerror psz_reverse_predict_lorenzo(psz_carray in_errquant, psz_outlier in_scattered_outlier, psz_rc const rc, psz_carray out_reconstruct, f4*,void* stream);
/**/pszerror psz_reverse_predict_spline(psz_carray in_errquant, psz_carray in_anchor, psz_outlier in_scattered_outlier, psz_carray out_reconstruct, void* stream);
// clang-format on

#endif /* B140A45C_CB00_4A4F_B9BC_F84ABB108FD9 */
