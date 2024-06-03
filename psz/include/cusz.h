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

#ifndef CUSZ_H
#define CUSZ_H

#ifdef __cplusplus
extern "C" {
#endif

#include "context.h"
#include "cusz/type.h"
#include "header.h"

#define psz_create capi_psz_create
#define psz_create_default capi_psz_create_default
#define psz_create_from_context capi_psz_create_from_context
#define psz_create_from_header capi_psz_create_from_header
#define psz_load_context_from_header capi_psz_load_context_from_header
#define psz_release capi_psz_release
#define psz_compress_init capi_psz_compress_init
#define psz_decompress_init capi_psz_decompress_init
#define psz_compress capi_psz_compress
#define psz_decompress capi_psz_decompress
#define psz_stage_time capi_psz_stage_time

psz_compressor* capi_psz_create(
    /* data */ psz_dtype const dtype, psz_len3 const uncomp_len,  //
    /* config */ psz_predtype const predictor, int const quantizer_radius,
    psz_codectype const codec);

psz_compressor* capi_psz_create_default(psz_dtype const dtype, psz_len3 const);

psz_compressor* capi_psz_create_from_context(pszctx* const, psz_len3 const);
psz_compressor* capi_psz_create_from_header(psz_header* const);

pszerror capi_psz_release(psz_compressor* comp);

pszerror capi_psz_compress(
    psz_compressor* comp, void* uncompressed, psz_len3 const uncomp_len,
    double const eb, psz_mode const mode, uint8_t** compressed,
    size_t* comp_bytes, psz_header* header, void* record, void* stream);

pszerror capi_psz_decompress(
    psz_compressor* comp, uint8_t* compressed, size_t const comp_len,
    void* decompressed, psz_len3 const decomp_len, void* record, void* stream);

float* capi_psz_stage_time(psz_compressor*);

// defined in context.cc
extern void capi_psz_version();
extern void capi_psz_versioninfo();

#ifdef __cplusplus
}
#endif

#endif