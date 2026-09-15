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
  PSZ_ABORT_COMPRESSED_TOO_LARGE,
  PSZ_ABORT_UNSUPPORTED_PIPELINE,
} psz_error_status;
typedef psz_error_status pszerror;

const char* psz_error_string(int e);

// aliasing
typedef uint8_t byte_t;
typedef size_t szt;

#define DEFAULT_PREDICTOR Lorenzo
#define DEFAULT_HISTOGRAM HistGeneric
#define DEFAULT_CODEC HFR_V4
#define DEFAULT_CODEC_ALT HFR_PBKC
#define NULL_HIST HistNull
#define NULL_CODEC CodecNull

// clang-format off
typedef enum { Abs, Rel } psz_mode;
typedef enum { Lorenzo = 0, LorenzoZigZag = 1, SplineY25 = 2, SplineY24 = 3 } psz_predictor;

// HF_r2:      -c1 hf-rev2
// HFR V2:    -c1 hfr-v2            Tian et al. 2020, refined.
// HFR_V4:    -c1 hfr-v4 (default). backporting HFR-PBKC under single-book mode. 
// HFR-PBKC:  -c1 hfr-pbkc
// HFR-PBKGO: -c1 hfr-pbkgo
typedef enum {
  HF = 0, HF_r2 = 1,
  HFR = 2, HFR_V2 = 3, HFR_V3 = 4, HFR_V4 = 5,
  HFR_PBKC = 6, HFR_PBKGO = 7, HFR_PBKF = 8,
  LC_TCMS = 9, LC_DRH = 10, LC_BITR = 13, LC_RTR = 14,
  FZG = 11,
  CodecNull = 99
} psz_codec;
typedef enum { HistGeneric, HistSp, HistNull } psz_hist;
// clang-format on

typedef struct psz_pipeline {
  psz_predictor predictor;
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

typedef struct psz_runtime_config2 {
  psz_mode mode;
  double eb;
  uint16_t radius;
} psz_rc2;

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
