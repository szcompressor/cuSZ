// Author: Jiannan Tian
// C-complient type definitions; no methods in this header.

#ifndef PSZ_TYPE_H
#define PSZ_TYPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "c_type.h"

typedef _ptb_device psz_device;
typedef _ptb_runtime psz_runtime;
typedef _ptb_runtime psz_backend;
typedef _ptb_toolkit psz_toolkit;

typedef _ptb_stream_t psz_stream_t;
typedef _ptb_mem_control psz_mem_control;
typedef _ptb_dtype psz_dtype;
typedef _ptb_len3 psz_len3;

typedef struct psz_data_summary {
  double min, max, rng, std, avg;
} psz_data_summary;

typedef struct psz_stats {
  psz_data_summary odata, xdata;
  struct {
    double PSNR, MSE, NRMSE, coeff;
  } score;
  struct {
    double abs, rel, pwrrel;
    size_t idx;
  } max_err;
  struct {
    double lag_one, lag_two;
  } autocor;
  double user_eb;
  size_t len;
} psz_stats;

typedef psz_len3 psz_len;

#define CUSZ_SUCCESS PSZ_SUCCESS

typedef enum {
  PSZ_SUCCESS = 0,
  // PSZ_WARN_RADIUS_TOO_LARGE = 1,
  PSZ_WARN_OUTLIER_TOO_MANY = 2,
  PSZ_ABORT_UNSUPPORTED_TYPE = 3,
  PSZ_ABORT_UNSUPPORTED_DIMENSION = 4,
  PSZ_ABORT_NOT_IMPLEMENTED = 5,
  // PSZ_ABORT_NO_SUCH_PREDICTOR = 6,
  PSZ_ABORT_NO_SUCH_CODEC = 7,
  // PSZ_ABORT_TOO_MANY_UNPREDICTABLE = 8,
  // PSZ_ABORT_TOO_MANY_ENC_BREAK = 9,
  PSZ_ABORT_COMPRESSED_TOO_LARGE = 10,
  PSZ_ABORT_UNSUPPORTED_PIPELINE = 11,
} psz_errno;
typedef psz_errno pszerror;

const char* psz_error_string(int e);

// aliasing
typedef uint8_t byte_t;
typedef size_t szt;

#define DEFAULT_PREDICTOR Lorenzo
#define DEFAULT_HISTOGRAM HistGeneric
#define DEFAULT_CODEC HFR_PBKC
#define DEFAULT_CODEC_ALT HFR_V4
#define NULL_HIST HistNull
#define NULL_CODEC CodecNull

// clang-format off
typedef enum { Abs, Rel } psz_mode;
typedef enum { Lorenzo = 0, LorenzoZigZag = 1, SplineY25 = 2, SplineY24 = 3 } psz_predictor;

// HF_r2:      -c1 hf-rev2
// HFR V2:    -c1 hfr-v2            Tian et al. 2020, refined.
// HFR_V4:    -c1 hfr-v4            backporting HFR-PBKC under single-book mode.
// HFR-PBKC:  -c1 hfr-pbkc (default)
// HFR-PBKGO: -c1 hfr-pbkgo
typedef enum {
  CodecNull = 0,
  HF = 1, HF_r2 = 2,
  HFR = 3, HFR_V2 = 4, HFR_V3 = 5, HFR_V4 = 6,
  HFR_PBKC = 7, HFR_PBKGO = 8, HFR_PBKF = 9,
  LC_TCMS = 10, LC_DRH = 11, LC_BITR = 14, LC_RTR = 15,
  FZG = 12
} psz_codec;
typedef enum { HistGeneric, HistSp, HistNull } psz_hist;
// clang-format on

typedef struct psz_pipeline {
  psz_predictor predictor;
  // DEPRECATED: needs_book(codec1) and _compose set psz_hist value.
  // TODO drop once HistSp vs HistGeneric settles.
  psz_hist hist;
  psz_codec codec1;
  psz_codec codec2;
} psz_ppl;

typedef enum {
  // generic
  PSZ_PRESET_P1_C1,
  PSZ_PRESET_P1_C1_C2,

  // fixed, historical
  PSZ_PRESET_LRZZZ_FZG,
  PSZ_PRESET_HICR,     // spline, HF_r2, LC-(1)
  PSZ_PRESET_HITP,     // spline, LC_TCMS, LC_BITR
  PSZ_PRESET_HITP_R1,  // spline, LC_DRH, LC_BITR
} psz_preset;

struct psz_context;
typedef struct psz_context psz_ctx;

struct psz_header;
typedef struct psz_header psz_header;

typedef struct psz_interp_params {
  double alpha, beta;

  bool use_md[6];
  bool use_natural[6];
  bool reverse[6];
  uint8_t auto_tuning;
} psz_interp_params;

typedef struct psz_interp_params INTERP_PARAMS;

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
