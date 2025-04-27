// 24-05-30 by Jiannan Tian

#ifndef HF_H
#define HF_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "c_type.h"

typedef void* phf_stream_t;

//////// state enumeration

#define PHF_SUCCESS 0
#define PHF_WRONG_DTYPE 1
#define PHF_FAIL_GPU_MALLOC 2
#define PHF_FAIL_GPU_MEMCPY 3
#define PHF_FAIL_GPU_ILLEGAL_ACCESS 4
#define PHF_FAIL_GPU_OUT_OF_MEMORY 5
#define PHF_NOT_IMPLEMENTED 99

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

#define phf_version capi_phf_version
#define phf_versioninfo capi_phf_versioninfo
#define phf_create capi_phf_create
#define phf_release capi_phf_release
#define phf_buildbook capi_phf_buildbook
#define phf_encode capi_phf_encode
#define phf_decode capi_phf_decode
#define phf_reverse_book_bytes capi_phf_reverse_book_bytes
#define phf_allocate_reverse_book_decode capi_phf_allocate_reverse_book

// phf helper, used by compressor; not exposed in py-binding
size_t capi_phf_coarse_tune_sublen(size_t);
void capi_phf_coarse_tune(size_t len, int* sublen, int* pardeg);

// management
void capi_phf_version();
void capi_phf_versioninfo();

// codec
phf_codec* capi_phf_create(size_t const inlen, phf_dtype const t, int const bklen);
int capi_phf_release(phf_codec*);
// TODO hist_len is not necessary; alternatively, it can force check size.
int capi_phf_buildbook(phf_codec* codec, uint32_t* d_hist, phf_stream_t);
int capi_phf_encode(
    phf_codec* codec, void* in, size_t const inlen, uint8_t** encoded, size_t* enc_bytes,
    phf_stream_t);
int capi_phf_decode(phf_codec* codec, uint8_t* encoded, void* decoded, phf_stream_t);

// helpers
size_t capi_phf_reverse_book_bytes(u2 bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES);
uint8_t* capi_phf_allocate_reverse_book(u2 bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES);

void pszanalysis_hf_buildtree(
    uint32_t* freq, int const bklen, double* entropy, double* cr, int const symbol_byte);

#ifdef __cplusplus
}
#endif
#endif