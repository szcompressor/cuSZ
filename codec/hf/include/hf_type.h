#ifndef FA7FCE65_CF3E_4561_986C_64F5638E398D
#define FA7FCE65_CF3E_4561_986C_64F5638E398D

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

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

#define PHFHEADER_HEADER 0
#define PHFHEADER_REVBK 1
// coarse-grained
#define PHFHEADER_PAR_NBIT 2
#define PHFHEADER_PAR_ENTRY 3
#define PHFHEADER_BITSTREAM 4
#define PHFHEADER_END 5
// HFR: fine-grained
#define PHFHEADER_HFR_DN_BITCOUNT 2
#define PHFHEADER_HFR_DN_START_LOC 3
#define PHFHEADER_HFR_DN_BITSTREAM 4
#define PHFHEADER_HFR_SP_VAL 5
#define PHFHEADER_HFR_SP_IDX 6
#define PHFHEADER_HFR_END 7

typedef uint32_t PHF_METADATA;
typedef uint8_t PHF_BIN;
typedef uint8_t PHF_BYTE;

typedef union {
  // placeholding size
  uint8_t __[128];

  struct {
    int bklen;
    int sublen, pardeg;
    size_t original_len, total_nbit, total_ncell;
    uint32_t entry[PHFHEADER_END + 1];
  };

  struct {
    int HFR_bklen;
    int HFR_pardeg;
    size_t HFR_original_len;
    size_t HFR_total_ncell;  // the final of dn_loc_inc
    int HFR_sp_num;
    uint32_t HFR_entry[PHFHEADER_HFR_END + 1];
    bool HFR_in_use;
  };
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
