#ifndef FA7FCE65_CF3E_4561_986C_64F5638E398D
#define FA7FCE65_CF3E_4561_986C_64F5638E398D

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// typedef enum phf_execution_policy { SEQ, CUDA, HIP, ONEAPI } phfpolicy;
// typedef phf_execution_policy phf_platform;
// typedef enum phf_device { CPU, NVGPU, AMDGPU, INTELGPU } phfdevice;

typedef void* uninit_stream_t;

//////// state enumeration

typedef enum phf_error_status {  //
  PHF_SUCCESS = 0x00,
  PHF_FAIL_GPU_MALLOC,
  PHF_FAIL_GPU_MEMCPY,
  PHF_FAIL_GPU_ILLEGAL_ACCESS,
  PHF_FAIL_GPU_OUT_OF_MEMORY,
  PHF_NOT_IMPLEMENTED
} phf_error_status;
typedef phf_error_status phferr;

typedef struct phf_dtype  //
{
  static const int F4 = 4;
  static const int F8 = 8;
  static const int U1 = 11;
  static const int U2 = 12;
  static const int U4 = 14;
  static const int U8 = 18;
  static const int I1 = 21;
  static const int I2 = 22;
  static const int I4 = 24;
  static const int I8 = 28;
  static const int ULL = 31;
} phf_dtype;

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

#ifdef __cplusplus
}
#endif

#endif /* FA7FCE65_CF3E_4561_986C_64F5638E398D */
