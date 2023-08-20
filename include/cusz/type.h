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

#include <stddef.h>
#include <stdint.h>

enum cusz_execution_policy { CPU, CUDA, HIP, ONEAPI, HIP_NV, ONEAPI_NV };
typedef enum cusz_execution_policy cusz_execution_policy;
typedef enum cusz_execution_policy pszexepolicy;
typedef enum cusz_execution_policy pszpolicy;

//////// state enumeration

typedef enum cusz_error_status {  //
  CUSZ_SUCCESS = 0x00,
  CUSZ_FAIL_ONDISK_FILE_ERROR = 0x01,
  CUSZ_FAIL_DATA_NOT_READY = 0x02,
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
  CUSZ_NOT_IMPLEMENTED = 0x0100,
} cusz_error_status;
typedef cusz_error_status psz_error_status;
typedef cusz_error_status pszerror;

typedef enum cusz_datatype  //
{ __F0 = 0,
  F4 = 4,
  F8 = 8,
  __U0 = 10,
  U1 = 11,
  U2 = 12,
  U4 = 14,
  U8 = 18,
  __I0 = 20,
  I1 = 21,
  I2 = 22,
  I4 = 24,
  I8 = 28,
  ULL = 31 } cusz_datatype;
typedef cusz_datatype cusz_dtype;
typedef cusz_datatype psz_dtype;
typedef cusz_datatype pszdtype;

// aliasing
typedef uint8_t u1;
typedef uint16_t u2;
typedef uint32_t u4;
typedef uint64_t u8;
typedef unsigned long long ull;
typedef int8_t i1;
typedef int16_t i2;
typedef int32_t i4;
typedef int64_t i8;
typedef float f4;
typedef double f8;
typedef size_t szt;

typedef enum cusz_executiontype  //
{ Device = 0,
  Host = 1,
  None = 2 } cusz_executiontype;
typedef cusz_executiontype psz_space;
typedef cusz_executiontype pszspace;

typedef enum cusz_mode  //
{ Abs = 0,
  Rel = 1 } cusz_mode;
typedef cusz_mode pszmode;

typedef enum cusz_predictortype  //
{ Lorenzo0 = 0,
  Lorenzo = 1,
  LorenzoI = 2,
  Spline = 3 } cusz_predictortype;
typedef cusz_predictortype pszpredictor_type;

typedef enum cusz_preprocessingtype  //
{ FP64toFP32 = 0,
  LogTransform,
  ShiftedLogTransform,
  Binning2x2,
  Binning2x1,
  Binning1x2,
} cusz_preprocessingtype;
typedef cusz_preprocessingtype psz_preprocess;
typedef cusz_preprocessingtype pszpreprocess;

typedef enum cusz_codectype  //
{ Huffman = 0,
  RunLength,
  // NvcompCascade,
  // NvcompLz4,
  // NvcompSnappy,
} cusz_codectype;
typedef cusz_codectype pszcodec;

typedef enum cusz_huffman_booktype  //
{ Canonical = 1,
  Sword = 2,
  Mword = 3 } cusz_huffman_booktype;
typedef cusz_huffman_booktype pszhfbk;

typedef enum cusz_huffman_codingtype  //
{ Coarse = 0,
  Fine = 1 } cusz_huffman_codingtype;
typedef cusz_huffman_codingtype pszhfpar;

//////// configuration template
typedef struct pszlen {
  // clang-format off
    union { size_t x0, x; };
    union { size_t x1, y; };
    union { size_t x2, z; };
    union { size_t x3, w; };
  // clang-format on
} pszlen;

typedef struct cusz_predictor {
  pszpredictor_type type;
} cusz_predictor;
typedef cusz_predictor pszpredictor;

typedef struct cusz_quantizer {
  int radius;
} cusz_quantizer;
typedef cusz_quantizer pszquantizer;

typedef struct cusz_hfcoder {
  pszhfbk book;
  pszhfpar style;

  int booklen;
  int coarse_pardeg;
} cusz_hfcoder;
// typedef cusz_hfcoder pszhfrc;
typedef cusz_hfcoder pszhfrc;

////// wrap-up

typedef struct cusz_framework {
  pszpredictor predictor;
  pszquantizer quantizer;
  pszhfrc hfcoder;
  f4 max_outlier_percent;
} cusz_custom_framcusz_frameworkework;
typedef cusz_framework pszframework;
typedef cusz_framework pszframe;

struct cusz_context;
typedef struct cusz_context pszctx;

struct cusz_header;
typedef struct cusz_header pszheader;

typedef struct cusz_compressor {
  void* compressor;
  pszctx* ctx;
  pszheader* header;
  pszframe* framework;
  pszdtype type;
} cusz_compressor;
typedef cusz_compressor pszcompressor;

typedef struct cusz_runtime_config {
  f8 eb;
  pszmode mode;
} cusz_runtime_config;
typedef cusz_runtime_config pszruntimeconfig;
typedef cusz_runtime_config pszrc;

typedef struct Res {
  f8 min, max, rng, std;
} pszscanres;
typedef pszscanres Res;

typedef struct cusz_stats {
  // clang-format off
    pszscanres odata, xdata;
    struct { f8 PSNR, MSE, NRMSE, coeff; } score;
    struct { f8 abs, rel, pwrrel; size_t idx; } max_err;
    struct { f8 lag_one, lag_two; } autocor;
    f8 user_eb;
    size_t len;
  // clang-format on
} cusz_stats;
typedef cusz_stats pszsummary;

typedef u1* pszout;
// used for bridging some compressor internal buffer
typedef pszout* ptr_pszout;

#ifdef __cplusplus
}
#endif

#endif
