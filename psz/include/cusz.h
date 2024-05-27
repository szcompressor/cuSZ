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

#include "context.h"
#include "cusz/record.h"
#include "cusz/type.h"
#include "header.h"

pszcompressor* psz_create(
    /* input */ psz_dtype const dtype,  //
    /* config */ psz_predtype const predictor, int const quantizer_radius,
    psz_codectype const codec,  //
    /* runtime */ double const eb, psz_mode const mode);

pszcompressor* psz_create_default(
    psz_dtype const dtype, double const eb, psz_mode const mode);

pszcompressor* psz_create_from_context(pszctx* const ctx);

pszerror psz_release(pszcompressor* comp);

pszerror psz_compress_init(pszcompressor* comp, psz_len3 const uncomp_len);

pszerror psz_compress(
    pszcompressor* comp, void* uncompressed, psz_len3 const uncomp_len,
    uint8_t** compressed, size_t* comp_bytes, pszheader* header, void* record,
    void* stream);

pszerror psz_decompress_init(pszcompressor* comp, pszheader* header);

pszerror psz_decompress(
    pszcompressor* comp, uint8_t* compressed, size_t const comp_len,
    void* decompressed, psz_len3 const decomp_len, void* record, void* stream);

#endif

#ifdef __cplusplus
}
#endif
