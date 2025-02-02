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

typedef enum {
  CUSZ_SUCCESS,
  CUSZ_GENERAL_GPU_FAILURE,
  CUSZ_FAIL_ONDISK_FILE_ERROR,
  CUSZ_FAIL_DATA_NOT_READY,
  PSZ_TYPE_UNSUPPORTED,
  PSZ_ERROR_GPU_GENERAL,
  PSZ_ERROR_OUTLIER_OVERFLOW,
  PSZ_ERROR_IO,
  // specify error when calling CUDA API
  CUSZ_FAIL_GPU_MALLOC,
  CUSZ_FAIL_GPU_MEMCPY,
  CUSZ_FAIL_GPU_ILLEGAL_ACCESS,
  // specify error related to our own memory manager
  CUSZ_FAIL_GPU_OUT_OF_MEMORY,
  // when compression is useless
  CUSZ_FAIL_INCOMPRESSIABLE,
  // TODO component related error
  CUSZ_FAIL_UNSUPPORTED_DATATYPE,
  CUSZ_FAIL_UNSUPPORTED_QUANTTYPE,
  CUSZ_FAIL_UNSUPPORTED_PRECISION,
  CUSZ_FAIL_UNSUPPORTED_PIPELINE,
  // not-implemented error
  CUSZ_NOT_IMPLEMENTED,
  // too many outliers
  CUSZ_OUTLIER_TOO_MANY,
  // specified wrong timer
  PSZ_WRONG_TIMER_SPECIFIED,
} psz_error_status;
typedef psz_error_status pszerror;

typedef uint8_t byte_t;
typedef size_t szt;

#define DEFAULT_PREDICTOR Lorenzo
#define DEFAULT_HISTOGRAM HistogramGeneric
#define DEFAULT_CODEC Huffman
#define NULL_HISTOGRAM NullHistogram
#define NULL_CODEC NullCodec

typedef enum { Abs, Rel } psz_mode;
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
  HuffmanRevisit,
  FZGPUCodec,
  RunLength,
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

#ifdef __cplusplus
}
#endif

#endif
