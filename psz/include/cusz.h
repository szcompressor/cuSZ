//
// @file cusz.h
// @author Jiannan Tian
// @brief
// @version 0.3
// @date 2022-04-29
//
// (C) 2022 by Washington State University, Argonne National Laboratory
//
//

#ifndef CUSZ_H
#define CUSZ_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cusz/context.h"
#include "cusz/header.h"
#include "cusz/type.h"
#include "cusz_rev1.h"
#include "stat.h"

// defined in context.cc
extern void psz_version();
extern void psz_versioninfo();

/* CUSZ_LEGACY_MACRO_ALIASING: compatibility layer for deprecated OOD connector symbols */
#ifndef CUSZ_DISABLE_LEGACY_MACROS
#define CUSZ_LEGACY_MACRO_ALIASING 1
#define cusz_datatype psz_dtype
#define cusz_mode psz_mode
#define cusz_predictortype psz_predictor
#define cusz_codectype psz_codec
#define cusz_error_status psz_error_status
#define cusz_header psz_header
#define cusz_compressor psz_resource
#define cusz_config psz_rc2
#define cusz_len psz_len

#ifndef FP32
#define FP32 F4
#endif
#ifndef FP64
#define FP64 F8
#endif
#ifndef Lorenzo0
#define Lorenzo0 Lorenzo
#endif
#ifndef LorenzoI
#define LorenzoI Lorenzo
#endif
#ifndef LorenzoII
#define LorenzoII LorenzoZigZag
#endif
#ifndef Spline3
#define Spline3 Spline
#endif

typedef enum { Auto, Dense, Sparse } cusz_pipelinetype;
typedef enum { Tree, Canonical } cusz_huffman_booktype;
typedef enum { Host, Device, None } cusz_executiontype;
typedef enum { Coarse, Fine } cusz_huffman_codingtype;

typedef struct cusz_custom_predictor {
  psz_predictor type;
  bool anchor;
  bool nondestructive;
} cusz_custom_predictor;

typedef struct cusz_custom_quantization {
  int radius;
  bool delayed;
} cusz_custom_quantization;

typedef struct cusz_custom_codec {
  psz_codec type;
  bool variable_length;
  float presumed_density;
} cusz_custom_codec;

typedef struct cusz_custom_huffman_codec {
  cusz_huffman_booktype book_type;
  cusz_executiontype execution_type;
  cusz_huffman_codingtype coding_type;
  int booklen;
  int coarse_pardeg;
} cusz_custom_huffman_codec;

typedef struct cusz_custom_framework {
  psz_dtype datatype;
  cusz_pipelinetype pipeline;
  cusz_custom_predictor predictor;
  cusz_custom_quantization quantization;
  cusz_custom_codec codec;
  cusz_custom_huffman_codec huffman;
} cusz_custom_framework;

#endif

void psz_print_concise_quality(psz_header* h, psz_stats* s, size_t comp_bytes);
void* psz_make_timerecord();
void psz_review_comp_time_breakdown(void* _r, psz_header* h);
void psz_review_comp_time_from_header(psz_header* h);
void psz_review_comp_time_from_header_verbose(psz_header* h);
void psz_review_decomp_time_from_header(psz_header* h);
void psz_review_compression(void* r, psz_header* h);
void psz_review_decompression(void* r, size_t bytes);

#ifdef __cplusplus
}
#endif

#endif