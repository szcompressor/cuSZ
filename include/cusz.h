/**
 * @file cusz.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-04-29
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CUSZ_H
#define CUSZ_H

#include <stddef.h>

#include "cusz/record.h"
#include "cusz/type.h"
#include "header.h"

// #define cusz_create psz_create
// #define cusz_release psz_release
// #define cusz_compress psz_compress
// #define cusz_decompress psz_decompress

pszpredictor pszdefault_predictor();
pszquantizer pszdefault_quantizer();
pszhfrc pszdefault_hfcoder();
pszframe* pszdefault_framework();

pszcompressor* psz_create(pszframe* framework, psz_dtype const type);

pszerror psz_release(pszcompressor* comp);

pszerror psz_compress_init(
    pszcompressor* comp, pszlen const uncomp_len, pszctx* ctx);

pszerror psz_compress(
    pszcompressor* comp, void* uncompressed, pszlen const uncomp_len,
    ptr_pszout compressed, size_t* comp_bytes, pszheader* header, void* record,
    void* stream);

pszerror psz_decompress_init(pszcompressor* comp, pszheader* header);

pszerror psz_decompress(
    pszcompressor* comp, pszout compressed, size_t const comp_len,
    void* decompressed, pszlen const decomp_len, void* record, void* stream);


// modular functions as of 2401
// clang-format off
pszerror psz_predict_lorenzo(pszarray* in, pszarray* out_errquant, pszrc2 const rc, pszoutlier* out_outlier, void* stream);
pszerror psz_predict_spline(pszarray* in, pszarray* out_errquant, pszrc2 const rc, pszarray* out_anchor, pszoutlier* out_outlier, void* stream);
pszerror psz_histogram(pszarray* in, pszarray* hist, void* stream);
pszerror psz_encode_entropy(pszarray* in, pszarray* out_encoded, void* stream);
pszerror psz_encode_dictionary(pszcompressor_stream cor, pszarray* in, pszarray* out_encoded, void* stream);
pszerror psz_archive(pszcompressor_stream cor, pszarray* in_encoded, pszoutlier* in_outlier, pszheader* out_header, pszarray* out_archive, void* stream);

pszerror psz_unarchive(pszheader* in_header, pszarray* in_archive, pszarray* out_encoded, pszoutlier* out_outlier, void* stream);
pszerror psz_decode_entropy(pszarray* in_encoded, pszarray* out_decoded, void* stream);
pszerror psz_decode_dictionary(pszarray* in_encoded, pszarray* out_decoded, void* stream);
pszerror psz_reverse_predict_lorenzo(pszarray* in_errquant, pszoutlier* in_outlier, pszarray* out_reconstruct, void* stream);
pszerror psz_reverse_predict_spline(pszarray* in_errquant, pszarray* in_anchor, pszoutlier* in_outlier, pszarray* out_reconstruct, void* stream);
// clang-format on

#endif

#ifdef __cplusplus
}
#endif
