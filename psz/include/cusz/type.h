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
} psz_error_status;
typedef psz_error_status pszerror;

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
typedef enum { Lorenzo, LorenzoZigZag, Spline } psz_predictor;
// HFr2: HF (ph1+ph2) + LAGO concat (replaces legacy ph3 host-scan + ph4 copy) + AoS bheader_backport[].
// HFR: Tian et al. 2020, refined.
// HFR-PBKC: --codec1 hfr-pbkc. HFR-PBKGO: --codec1 hfr-pbkgo.
// HFR_V3: DEPRECATED (kept for bin_hf + reading old archives); superseded by HFR_V4. --codec1 hfr-v3.
// HFR_V4: default. v3 book pick, but on the HFR-PBKC kernel in single-book mode; --codec1 hfr-v4.
typedef enum { HF, HFr2, HFR, HFR_PBKC, HFR_PBKGO, HFR_PBKF, LC, FZG, RLE, HFR_V3, HFR_V4, CodecNull } psz_codec;
typedef enum { HistGeneric, HistSp, HistNull } psz_hist;
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
