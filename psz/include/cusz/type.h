/**
 * @file type.h
 * @author Jiannan Tian
 * @brief C-complient type definitions; no methods in this header.
 * @version 0.3
 * @date 2022-04-29
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#ifndef PSZ_TYPE_H
#define PSZ_TYPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "c_type.h"

typedef _portable_device psz_device;
typedef _portable_runtime psz_runtime;
typedef _portable_runtime psz_backend;
typedef _portable_toolkit psz_toolkit;

typedef _portable_stream_t psz_stream_t;
typedef _portable_mem_control psz_mem_control;
typedef _portable_dtype psz_dtype;
typedef _portable_len3 psz_len3;
typedef _portable_size3 psz_size3;
typedef _portable_data_summary psz_data_summary;

#define CUSZ_SUCCESS PSZ_SUCCESS
#define CUSZ_NOT_IMPLEMENTED PSZ_ABORT_NOT_IMPLEMENTED

typedef enum {
  PSZ_SUCCESS,
  PSZ_WARN_RADIUS_TOO_LARGE,
  PSZ_WARN_OUTLIER_TOO_MANY,
  PSZ_ABORT_UNSUPPORTED_TYPE,
  PSZ_ABORT_UNSUPPORTED_DIMENSION,
  PSZ_ABORT_NOT_IMPLEMENTED,
} psz_error_status;
typedef psz_error_status pszerror;

// aliasing
typedef uint8_t byte_t;
typedef size_t szt;

#define DEFAULT_PREDICTOR Lorenzo
#define DEFAULT_HISTOGRAM HistogramGeneric
#define DEFAULT_CODEC Huffman
#define NULL_HISTOGRAM NullHistogram
#define NULL_CODEC NullCodec

typedef enum { Abs, Rel, Verbatim } psz_mode;
typedef enum { Lorenzo, LorenzoZigZag, LorenzoProto, Spline } psz_predtype;

typedef enum {
  FP64toFP32,
  LogTransform,
  ShiftedLogTransform,
  Binning2x2,
  Binning2x1,
  Binning1x2,
} _future_psz_preprocestype;

typedef enum {
  Huffman,
  LC,
  // HuffmanRevisit,// status: not ready
  FZGPUCodec,  // status: not in use
  // RunLength,     // status: obsolete
  NullCodec,
} psz_codectype;

typedef enum {
  HistogramGeneric,
  HistogramSparse,
  NullHistogram,
} psz_histotype;

typedef enum {
  STAGE_PREDICT = 0,
  STAGE_HISTOGRM = 1,
  STAGE_BOOK = 3,
  STAGE_HUFFMAN = 4,
  STAGE_OUTLIER = 5,
  STAGE_END = 6
} psz_time_stage;

struct psz_context;
typedef struct psz_context psz_ctx;

struct psz_header;
typedef struct psz_header psz_header;

typedef struct psz_compressor {
  void* compressor;
  psz_ctx* ctx;
  psz_error_status last_error;
  void* mem;
  float stage_time[STAGE_END];
} psz_compressor;

// nested struct object (rather than ptr) results in Swig creating a `__get`,
// which can be breaking. Used `prefix_` instead.
typedef struct psz_statistics {
  psz_data_summary odata, xdata;
  f8 score_PSNR, score_MSE, score_NRMSE, score_coeff;
  f8 max_err_abs, max_err_rel, max_err_pwrrel;
  size_t max_err_idx;
  f8 autocor_lag_one, autocor_lag_two;
  f8 user_eb;
  size_t len;
} psz_statistics;

typedef struct psz_interp_params {
  double alpha, beta;

  bool use_md[6];
  bool use_natural[6];
  bool reverse[6];
  uint8_t auto_tuning;
} psz_interp_params;

typedef struct psz_interp_params INTERPOLATION_PARAMS;

// C-style "constructor"
static inline psz_interp_params make_default_params(void)
{
  psz_interp_params p = {
      .alpha = 1.75,
      .beta = 4.0,
      .use_md = {1, 1, 0, 0, 0, 0},
      .use_natural = {0, 0, 0, 0, 0, 0},
      .reverse = {0, 0, 0, 0, 0, 0},
      .auto_tuning = 3};
  return p;
}

#ifdef __cplusplus
}
#endif

#endif
