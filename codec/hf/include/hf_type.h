#ifndef FA7FCE65_CF3E_4561_986C_64F5638E398D
#define FA7FCE65_CF3E_4561_986C_64F5638E398D

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "c_type.h"

typedef void* phf_stream_t;

//////// state enumeration

typedef enum {  //
  PHF_SUCCESS = 0x00,
  PHF_WRONG_DTYPE,
  PHF_FAIL_GPU_MALLOC,
  PHF_FAIL_GPU_MEMCPY,
  PHF_FAIL_GPU_ILLEGAL_ACCESS,
  PHF_FAIL_GPU_OUT_OF_MEMORY,
  PHF_NOT_IMPLEMENTED
} phf_error_status;
typedef phf_error_status phferr;

typedef enum { HF_U1, HF_U2, HF_U4, HF_U8, HF_ULL, HF_INVALID } phf_dtype;

#define PHFHEADER_FORCED_ALIGN 128
#define PHFHEADER_HEADER 0
#define PHFHEADER_REVBK 1
#define PHFHEADER_PAR_NBIT 2
#define PHFHEADER_PAR_ENTRY 3
#define PHFHEADER_BITSTREAM 4
#define PHFHEADER_END 5

typedef uint32_t PHF_METADATA;
typedef uint8_t PHF_BIN;
typedef uint8_t PHF_BYTE;

typedef struct {
  int bklen : 16;
  int sublen, pardeg;
  size_t original_len;
  size_t total_nbit, total_ncell;  // TODO change to uint32_t
  uint32_t entry[PHFHEADER_END + 1];
} phf_header;

#define capi_phf_encoded_bytes phf_encoded_bytes
uint32_t capi_phf_encoded_bytes(phf_header* h);

typedef struct {
  void* codec;
  phf_header* header;
  phf_dtype data_type;
} phf_codec;

#ifdef __cplusplus
}
#endif

#endif /* FA7FCE65_CF3E_4561_986C_64F5638E398D */
