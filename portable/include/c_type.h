#ifndef _PORTABLE_TYPE_H
#define _PORTABLE_TYPE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// enum for device type
typedef enum { CPU, NVGPU, AMDGPU, INTELGPU } _portable_device;
// enum for runtime type
typedef enum { SEQ, SIMD, OPENMP, CUDA, ROCM, SYCL, THRUST_DPL } _portable_runtime;
// enum for toolkit type
typedef enum { VENDOR_NATIVE, KOKKOS, ONEAPI, HIP } _portable_toolkit;

// clang-format off
// enum for memory control
typedef enum _portable_mem_control {
  Malloc, MallocHost, MallocManaged, MallocShared,
  Free, FreeHost, FreeManaged, FreeShared,
  ClearHost, ClearDevice,
  H2D, H2H, D2H, D2D,
  Async_H2D, Async_H2H, Async_D2H, Async_D2D,
  ToFile, FromFile,
  ExtremaScan,
  DBG,
} _portable_mem_control;
// clang-format on

// error status instead of int as ret-type
typedef enum {
  _SUCCESS,
  _FAIL_GENERAL,
  _FAIL_UNSUPPORTED_DTYPE,
  _NOT_IMPLIMENTED
} _portable_error_status;

// symbol of dtypes
typedef enum { F4, F8, U1, U2, U4, U8, I1, I2, I4, I8, ULL } _portable_dtype;

// type aliasing
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

typedef void* _portable_stream_t;

//  mirror CUDA dim3: using u4 dtype and x-y-z order
typedef struct _portable_len3 {
  size_t x, y, z;
} _portable_len3;
typedef _portable_len3 _portable_dim3;

// mirror typical math order: z-y-x
typedef struct _portable_size3 {
  size_t z, y, x;
} _portable_size3;

// struct for basic data summary
typedef struct _portable_data_summary {
  f8 min, max, rng, std, avg;
} _portable_data_summary;

#ifdef __cplusplus
}
#endif

#endif /* _PORTABLE_TYPE_H */
