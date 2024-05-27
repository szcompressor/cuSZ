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

#endif

#ifdef __cplusplus
}
#endif
