#ifndef B140A45C_CB00_4A4F_B9BC_F84ABB108FD9
#define B140A45C_CB00_4A4F_B9BC_F84ABB108FD9

#include "cusz/array.h"

// modular functions as of 2401
// clang-format off
pszerror psz_predict_lorenzo(pszarray in, pszrc2 const rc, pszarray out_errquant, pszoutlier out_outlier, void* stream);

/**/pszerror psz_predict_spline(pszarray in, pszrc2 const rc, pszarray out_errquant, pszarray out_anchor, pszoutlier out_outlier, void* stream);
/**/pszerror psz_histogram(pszarray in, pszarray hist, void* stream);
/**/pszerror psz_encode_entropy(pszarray in, pszarray out_encoded, void* stream);
/**/pszerror psz_encode_dictionary(pszcompressor_stream cor, pszarray in, pszarray out_encoded, void* stream);
/**/pszerror psz_archive(pszcompressor_stream cor, pszarray in_encoded, pszoutlier in_scattered_outlier, pszheader* out_header, pszarray out_archive, void* stream);

/**/pszerror psz_unarchive(pszheader* in_header, pszarray in_archive, pszarray out_encoded, pszoutlier out_outlier, void* stream);
/**/pszerror psz_decode_entropy(pszarray in_encoded, pszarray out_decoded, void* stream);
/**/pszerror psz_decode_dictionary(pszarray in_encoded, pszarray out_decoded, void* stream);
pszerror psz_reverse_predict_lorenzo(pszarray in_errquant, pszoutlier in_scattered_outlier, pszrc2 const rc, pszarray out_reconstruct, void* stream);
/**/pszerror psz_reverse_predict_spline(pszarray in_errquant, pszarray in_anchor, pszoutlier in_scattered_outlier, pszarray out_reconstruct, void* stream);
// clang-format on

#endif /* B140A45C_CB00_4A4F_B9BC_F84ABB108FD9 */
