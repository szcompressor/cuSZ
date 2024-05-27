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

typedef enum psz_backend { SEQ, CUDA, HIP, ONEAPI, THRUST } pszpolicy;
typedef enum psz_backend psz_platform;
typedef enum psz_device { CPU, NVGPU, AMDGPU, INTELGPU } pszdevice;

typedef void* uninit_stream_t;

//////// state enumeration

typedef enum psz_error_status {  //
  CUSZ_SUCCESS,
  //
  CUSZ_GENERAL_GPU_FAILURE,
  //
  CUSZ_FAIL_ONDISK_FILE_ERROR,
  CUSZ_FAIL_DATA_NOT_READY,
  //
  PSZ_ERROR_GPU_GENERAL,
  //
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
} psz_error_status;
typedef psz_error_status pszerror;

typedef enum psz_dtype  //
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
  ULL = 31 } psz_dtype;

typedef psz_dtype pszdtype;

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

typedef uint8_t byte_t;
typedef size_t szt;

typedef enum psz_space  //
{ Device = 0,
  Host = 1,
  None = 2 } psz_space;

typedef enum psz_mode  //
{ Abs = 0,
  Rel = 1 } psz_mode;

typedef enum psz_predtype  //
{ Lorenzo = 0,
  Spline = 1 } psz_predtype;

typedef enum psz_preptype  //
{ FP64toFP32 = 0,
  LogTransform,
  ShiftedLogTransform,
  Binning2x2,
  Binning2x1,
  Binning1x2,
} psz_prep_type;

typedef enum psz_codectype  //
{ Huffman = 0,
  HuffmanRevisit,
  RunLength,
} psz_codectype;

typedef enum psz_hfbktype  //
{ Canonical = 1,
  Sword = 2,
  Mword = 3 } psz_hfbktype;

typedef enum psz_hfpartype  //
{ Coarse = 0,
  Fine = 1 } psz_hfpartype;

//////// configuration template
typedef struct pszlen {
  // clang-format off
    union { size_t x0, x; };
    union { size_t x1, y; };
    union { size_t x2, z; };
    union { size_t x3, w; };
  // clang-format on
} pszlen;

typedef struct pszpredictor {
  psz_predtype type;
} pszpredictor;

typedef struct psz_quantizer {
  int radius;
} psz_quantizer;
typedef psz_quantizer pszquantizer;

typedef struct psz_hfruntimeconfig {
  psz_hfbktype book;
  psz_hfpartype style;

  int booklen;
  int coarse_pardeg;
} psz_hfruntimeconfig;
typedef psz_hfruntimeconfig pszhfrc;

////// wrap-up

// typedef struct psz_framework {
//   pszpredictor predictor;
//   pszquantizer quantizer;
//   pszhfrc hfcoder;
//   f4 max_outlier_percent;
// } psz_framework;
// typedef psz_framework pszframe;

struct psz_context;
typedef struct psz_context pszctx;

struct psz_header;
typedef struct psz_header pszheader;

typedef struct psz_compressor {
  void* compressor;
  pszctx* ctx;
  pszheader* header;
  // pszframe* framework;  // deprecated
  psz_dtype type;
} psz_compressor;
typedef psz_compressor pszcompressor;

typedef struct psz_runtimeconfig {
  f8 eb;
  psz_mode mode;
  psz_predtype pred_type;
  bool est_cr;
} psz_runtimeconfig;
typedef psz_runtimeconfig pszrc;

typedef struct Res {
  f8 min, max, rng, std;
} pszscanres;
typedef pszscanres Res;

typedef struct psz_summary {
  // clang-format off
    pszscanres odata, xdata;
    struct { f8 PSNR, MSE, NRMSE, coeff; } score;
    struct { f8 abs, rel, pwrrel; size_t idx; } max_err;
    struct { f8 lag_one, lag_two; } autocor;
    f8 user_eb;
    size_t len;
  // clang-format on
} psz_summary;
typedef psz_summary pszsummary;

// typedef u1* pszout;
// // used for bridging some compressor internal buffer
// typedef pszout* ptr_pszout;

typedef struct psz_api_array {
  void* const buf;
  pszlen const len3;
  psz_dtype dtype;
} pszarray;

typedef pszarray psz_data_input;
typedef pszarray psz_input;
typedef pszarray psz_in;
typedef pszarray* pszarray_mutable;

typedef struct psz_rettype_archive {
  u1* compressed;
  size_t* comp_bytes;
  pszheader* header;
} psz_archive;

typedef psz_archive psz_data_output;
typedef psz_archive psz_output;
typedef psz_archive psz_out;

typedef struct psz_api_compact {
  void* const val;
  uint32_t* idx;
  uint32_t* num;
  uint32_t reserved_len;
  pszdtype const dtype;
} sz_api_compact;

typedef sz_api_compact pszcompact;
typedef pszcompact pszoutlier;
typedef pszoutlier* pszoutlier_mutable;



// 2401
typedef struct __pszimpl_binding_compressor_stream {
  pszcompressor* compressor;
  void* stream;
} __pszimpl_compressor_stream_binding;

typedef __pszimpl_compressor_stream_binding pszbinding_compresssor_stream;
typedef pszbinding_compresssor_stream pszcompressor_stream;

typedef struct __pszimpl_rc {
  double eb;
  int radius;
} __pszimpl_rc;
typedef __pszimpl_rc pszrc2;

// forward
struct pszprof;

typedef enum psz_timing_mode {
  CPU_BARRIER_AND_TIMING,
  CPU_BARRIER,
  GPU_AUTOMONY
} psz_timing_mode;

#ifdef __cplusplus
}
#endif

#endif
