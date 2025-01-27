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
#define psz_release capi_psz_release
#define psz_compress capi_psz_compress
#define psz_decompress capi_psz_decompress
#define psz_clear_buffer capi_psz_clear_buffer
#define psz_make_timerecord capi_psz_make_timerecord
#define psz_review_comp_time_breakdown capi_psz_review_comp_time_breakdown
#define psz_review_comp_time_from_header capi_psz_review_comp_time_from_header
#define psz_review_decomp_time_from_header capi_psz_review_decomp_time_from_header
#define psz_review_compression capi_psz_review_compression
#define psz_review_decompression capi_psz_review_decompression
#define psz_review_evaluated_quality capi_psz_review_evaluated_quality

/**
 * @brief create a cuSZ compressor object with detailed specification; used in
 * setting up compression
 *
 * @param dtype input data type, select from `F4` (`F8` to be supported)
 * @param uncomp_len3 3D length of uncompressed data
 * @param predictor select from `lorenzo`, `spline`
 * @param quantizer_radius internal quantization range; 512 is recommended
 * @deprecated quantizer_radius This parameter is deprecated and will be
 * removed.
 * @param codec selecto from `Huffman` (more to be supported)
 * @return psz_compressor* an object of compressor wrapper for compression
 */
psz_compressor* capi_psz_create(
    /* data */ psz_dtype const dtype, psz_len3 const uncomp_len,  //
    /* config */ psz_predtype const predictor, int const quantizer_radius,
    psz_codectype const codec);

void* capi_psz_create__experimental(
    psz_dtype const dtype, psz_len3 const uncomp_len3, psz_predtype const predictor,
    int const quantizer_radius, psz_codectype const codec);

/**
 * @brief create a cuSZ compressor object with default specificaition; used in
 * setting up compression
 *
 * @param dtype input data type, select from `F4` (`F8` to be supported)
 * @param uncomp_len3 3D length of uncompressed data
 * @return psz_compressor* an object of compressor wrapper for compression
 */
psz_compressor* capi_psz_create_default(psz_dtype const dtype, psz_len3 const);

/**
 * @brief create a cuSZ compressor object with a pSZ context object; need to
 * configure context object first; used in setting up compression
 *
 * @return psz_compressor* an object of compressor wrapper for compression
 */
psz_compressor* capi_psz_create_from_context(pszctx* const, psz_len3 const);

/**
 * @brief create a cuSZ compressor object with a pSZ header; used in setting up
 * decompression
 *
 * @return psz_compressor* an object of compressor wrapper for decompression
 */
psz_compressor* capi_psz_create_from_header(psz_header* const);

void* capi_psz_create_from_header__experimental(psz_header* const);

/**
 * @brief release the compressor object
 *
 * @param comp compressor object
 * @return pszerror error status
 */
pszerror capi_psz_release(psz_compressor* comp);

pszerror capi_psz_release__experimental(void* comp);

/**
 * @brief compresses data using the specified runtime parameters.
 *
 * @param comp compressor object
 * @param d_in input: device pointer to the data array to be compressed
 * @param in_len3 input: the 3D length of the input data array
 * @param eb config: user-specified error bound (e.g., 1e-3); used with `mode`
 * (the next param)
 * @param mode config: select form `Abs` or `Rel`
 * @param d_compressed output: pointer to the device pointer to the internal
 * buffer holding the compressed data
 * @param comp_bytes output: number of bytes of `compressed`; changed by the
 * compressor internally
 * @param header output: pointer to the header as part of the internal buffer
 * @param record logging: breakdown time for each kernel; this is not an
 * end-to-end measurement.
 * @param stream the specified GPU stream
 * @return pszerror error status
 */
pszerror capi_psz_compress(
    psz_compressor* comp, void* d_in, psz_len3 const in_len3, double const eb, psz_mode const mode,
    uint8_t** d_compressed, size_t* comp_bytes, psz_header* header, void* record, void* stream);

pszerror capi_psz_compress__experimental(
    void* comp, void* d_in, psz_len3 const in_len3, double const eb, psz_mode const mode,
    uint8_t** d_compressed, size_t* comp_bytes, psz_header* header, void* record, void* stream);

/**
 * @brief decompress data from the archive
 *
 * @param comp compressor object
 * @param d_compressed input: device pointer to the data archive
 * @param comp_len input: the 1D length of the data archive
 * @param d_decompressed output: device pointer to the data buffer to hold the
 * decompressed data; prepared by
 * @param decomp_len the auxillary 3D length of the input data array
 * @param record logging: breakdown time for each kernel; this is not an
 * end-to-end measurement.
 * @param stream the specified GPU stream
 * @return pszerror error status
 */
pszerror capi_psz_decompress(
    psz_compressor* comp, uint8_t* d_compressed, size_t const comp_len, void* d_decompressed,
    psz_len3 const decomp_len, void* record, void* stream);

/**
 * @brief clear the internal buffer of the compressor object
 *
 * @param comp compressor object
 * @return pszerror error status
 */
pszerror capi_psz_clear_buffer(psz_compressor* comp);

// defined in context.cc
extern void capi_psz_version();
extern void capi_psz_versioninfo();

// review
void* capi_psz_make_timerecord();
void capi_psz_review_comp_time_breakdown(void* _r, psz_header* h);
void capi_psz_review_comp_time_from_header(psz_header* h);
void capi_psz_review_decomp_time_from_header(psz_header* h);
void capi_psz_review_compression(void* r, psz_header* h);
void capi_psz_review_decompression(void* r, size_t bytes);
void capi_psz_review_evaluated_quality(psz_runtime, psz_dtype, void*, void*, size_t, size_t, bool);

#ifdef __cplusplus
}
#endif

#endif