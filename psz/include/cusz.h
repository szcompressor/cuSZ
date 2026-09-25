// @file cusz.h
// @author Jiannan Tian
// @version 0.19

#ifndef CUSZ_H
#define CUSZ_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "cusz/header.h"
#include "cusz/type.h"

void psz_version();
void psz_versioninfo();

psz_ctx* psz_compress_init(psz_dtype, psz_len, void* stream);
psz_ctx* psz_compress_init_2stage(psz_dtype, psz_len, void* stream);
psz_ctx* psz_compress_init_3stage(psz_dtype, psz_len, void* stream);
psz_ctx* psz_decompress_init(psz_header*, void* stream);
int psz_free(psz_ctx* ctx);
psz_errno psz_last_error();

psz_data_summary psz_compress_extrema_float(psz_ctx*, float* d_in);
psz_data_summary psz_compress_extrema_double(psz_ctx*, double* d_in);

int psz_compress_process_float(psz_ctx*, psz_ppl, double eb, float* d_in);
int psz_compress_process_double(psz_ctx*, psz_ppl, double eb, double* d_in);

int psz_compress_archive(psz_ctx*, psz_header*, uint8_t** d_out, size_t* out_bytes);
int psz_compress_reset(psz_ctx*);

int psz_compress_analysis_float(psz_ctx*, double eb, float* d_in, u4* h_hist);
int psz_compress_analysis_double(psz_ctx*, double eb, double* d_in, u4* h_hist);

int psz_decompress_process_float(psz_ctx*, uint8_t* d_in, size_t const in_bytes, float* out);
int psz_decompress_process_double(psz_ctx*, uint8_t* d_in, size_t const in_bytes, double* out);

int psz_decompress_reset(psz_ctx*);

int psz_assess_quality_float(psz_stats*, float* d_reconst, float* d_origin, size_t len);
int psz_assess_quality_double(psz_stats*, double* d_reconst, double* d_origin, size_t len);

void psz_print_concise_quality(psz_header*, psz_stats*, size_t comp_bytes);

void psz_review_compression(psz_header*);
void psz_review_compression_verbose(psz_header*);

void psz_review_decompression(psz_header*);
void psz_review_decompression_verbose(psz_header*);

#ifdef __cplusplus
}
#endif

#endif
