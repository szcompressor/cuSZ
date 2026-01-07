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
// psz_data_summary now defined in stat.h

// Currently, 3D is the highest supported dimention.
typedef psz_len3 psz_len;

#define CUSZ_SUCCESS PSZ_SUCCESS

typedef enum {
  PSZ_SUCCESS,
  PSZ_WARN_RADIUS_TOO_LARGE,
  PSZ_WARN_OUTLIER_TOO_MANY,
  PSZ_ABORT_UNSUPPORTED_TYPE,
  PSZ_ABORT_UNSUPPORTED_DIMENSION,
  PSZ_ABORT_NOT_IMPLEMENTED,
  PSZ_ABORT_NO_SUCH_PREDICTOR,
  PSZ_ABORT_NO_SUCH_CODEC,
  PSZ_ABORT_TOO_MANY_UNPREDICTABLE,
  PSZ_ABORT_TOO_MANY_ENC_BREAK,
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

// clang-format off
typedef enum { Abs, Rel } psz_mode;
typedef enum { Lorenzo, LorenzoZigZag, LorenzoProto, Spline } psz_predictor;
typedef enum { FP64toFP32, LogTransform, ShiftedLogTransform, Binning2x2, Binning2x1, Binning1x2 } _future_psz_preprocess;
typedef enum { Huffman, HuffmanRevisit, LC, FZCodec, RunLength, NullCodec } psz_codec;
typedef enum { HistogramGeneric, HistogramSparse, NullHistogram } psz_hist;
// clang-format on

typedef struct psz_pipeline {
  psz_predictor predictor;
  psz_hist hist;
  psz_codec codec1;
  psz_codec codec2;
} psz_pipeline;

typedef struct psz_runtime_config2 {
  psz_mode mode;
  double eb;
  uint16_t radius;
} psz_rc2;

struct psz_context;
typedef struct psz_context psz_ctx;

struct psz_header;
typedef struct psz_header psz_header;

typedef struct psz_compressor {
  void* compressor;
  psz_ctx* ctx;
  psz_error_status last_error;
  void* mem;
} psz_compressor;

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
