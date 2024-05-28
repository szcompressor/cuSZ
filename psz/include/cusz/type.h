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

typedef enum { SEQ, CUDA, HIP, ONEAPI, THRUST } psz_backend;
typedef enum { CPU, NVGPU, AMDGPU, INTELGPU } psz_device;
typedef enum { Device = 0, Host = 1, IamLost = 2 } psz_space;
typedef psz_backend psz_policy;

typedef void* psz_stream_t;

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

typedef enum psz_mode  //
{ Abs = 0,
  Rel = 1 } psz_mode;

typedef enum psz_predtype  //
{ Lorenzo = 0,
  Spline = 1 } psz_predtype;

typedef enum psz_preprocestype  //
{ FP64toFP32 = 0,
  LogTransform,
  ShiftedLogTransform,
  Binning2x2,
  Binning2x1,
  Binning1x2,
} psz_preprocestype;

typedef enum psz_codectype  //
{ Huffman = 0,
  HuffmanRevisit,
  RunLength,
} psz_codectype;

typedef struct psz_len3 {
  size_t x, y, z;
} psz_len3;

struct psz_context;
typedef struct psz_context psz_ctx;

struct psz_header;
typedef struct psz_header psz_header;

typedef struct psz_compressor {
  void* compressor;
  psz_ctx* ctx;
  psz_header* header;
  psz_dtype type;
} psz_compressor;

typedef struct psz_basic_data_description {
  f8 min, max, rng, std;
} psz_basic_data_description;
typedef psz_basic_data_description psz_data_desc;

// nested struct object (rather than ptr) results in Swig creating a `__get`,
// which can be breaking. Used `prefix_` instead.
typedef struct psz_statistic_summary {
  psz_data_desc odata, xdata;
  f8 score_PSNR, score_MSE, score_NRMSE, score_coeff;
  f8 max_err_abs, max_err_rel, max_err_pwrrel;
  size_t max_err_idx;
  f8 autocor_lag_one, autocor_lag_two;
  f8 user_eb;
  size_t len;
} psz_summary;

typedef struct psz_capi_array {
  void* const buf;
  psz_len3 const len3;
  psz_dtype dtype;
} psz_carray;

typedef psz_carray psz_data_input;
typedef psz_carray psz_input;
typedef psz_carray psz_in;
typedef psz_carray* pszarray_mutable;

typedef struct psz_rettype_archive {
  u1* compressed;
  size_t* comp_bytes;
  psz_header* header;
} psz_archive;

typedef psz_archive psz_data_output;
typedef psz_archive psz_output;
typedef psz_archive psz_out;

/**
 * @brief This is an archive description of compaction array rather than
 * runtime one, which deals with host-device residency status.
 *
 */
typedef struct psz_capi_compact {
  void* const val;
  uint32_t* idx;
  uint32_t* num;
  uint32_t reserved_len;
  psz_dtype const dtype;
} psz_capi_compact;

typedef psz_capi_compact psz_capi_outlier;
typedef psz_capi_compact psz_compact;
typedef psz_capi_compact psz_outlier;
typedef psz_outlier* psz_outlier_mutable;

typedef struct psz_runtime_config {
  double eb;
  int radius;
} psz_runtime_config;
typedef psz_runtime_config psz_rc;

// forward
struct psz_profiling;

typedef enum psz_timing_mode {
  CPU_BARRIER_AND_TIMING,
  CPU_BARRIER,
  GPU_AUTOMONY
} psz_timing_mode;

#ifdef __cplusplus
}
#endif

#endif
