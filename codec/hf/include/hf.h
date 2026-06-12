// 24-05-30 by Jiannan Tian

#ifndef HF_H
#define HF_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "c_type.h"

typedef void* phf_stream_t;

//////// state enumeration

typedef enum {
  PHF_SUCCESS = 0,                  // should be used
  PHF_WRONG_DTYPE = 1,              // unused
  PHF_FAIL_GPU_MALLOC = 2,          // unused
  PHF_FAIL_GPU_MEMCPY = 3,          // unused
  PHF_FAIL_GPU_ILLEGAL_ACCESS = 4,  // unused
  PHF_FAIL_GPU_OUT_OF_MEMORY = 5,   // unused
  PHF_NOT_IMPLEMENTED = 99,         // used
} phf_status;

typedef enum { HF_U1, HF_U2, HF_U4, HF_U8, HF_ULL, HF_INVALID } phf_dtype;

#define PHFHEADER_FORCED_ALIGN 128  // byte alignment of the header section, not a section id

typedef enum {
  PHFHEADER_HEADER = 0,
  PHFHEADER_RVBK,
  PHFHEADER_PAR_NBIT,
  PHFHEADER_PAR_ENTRY,
  PHFHEADER_BITSTREAM,
  PHFHEADER_PBK_HEADERS,
  PHFHEADER_HF_REV2_HEADER,
  PHFHEADER_END,
} phf_header_section;

typedef u4 PHF_METADATA;
typedef u1 PHF_BIN;
typedef u1 PHF_BYTE;

typedef struct {
  u1 log_bklen;  // bklen = 1<<log_bklen
  u1 g_encid;    // HFR-v3 global bkid (real member: offsetof'd by the async header patch)
  u4 total_ncell;
  u4 ori_len;
  int sublen, pardeg;  // pardeg = #blocks; can exceed u2 for large inputs
  u4 entry[PHFHEADER_END + 1];
} phf_header;

u4 phf_encoded_bytes(phf_header* h);
void phf_print_header(const phf_header* h, const char* dtype_str);

typedef struct {
  void* codec;
  phf_header* header;
  phf_dtype data_type;
} phf_codec;

// phf helper, used by compressor; not exposed in py-binding
size_t phf_coarse_tune_sublen(size_t);
void phf_coarse_tune(size_t len, int* sublen, int* pardeg);

// management
void phf_version();
void phf_versioninfo();

// codec
phf_codec* phf_create(size_t const inlen, phf_dtype const t, int const bklen);
int phf_release(phf_codec*);
// TODO hist_len is not necessary; alternatively, it can force check size.
int phf_buildbook(phf_codec* codec, u4* d_hist, phf_stream_t);
int phf_encode(
    phf_codec* codec, void* in, size_t const inlen, u1** encoded, size_t* enc_bytes, phf_stream_t);
int phf_encode_HFR(
    phf_codec* codec, void* in, bool use_HFR, size_t const inlen, u1** encoded, size_t* enc_bytes,
    phf_stream_t);
int phf_decode(phf_codec* codec, u1* encoded, void* decoded, phf_stream_t);

// helpers
size_t phf_reverse_book_bytes(u2 bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES);
u1* phf_allocate_reverse_book(u2 bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES);

void pszanalysis_hf_buildtree(
    u4* freq, int const bklen, double* entropy, double* cr, int const symbol_byte);

#ifdef __cplusplus
}
#endif
#endif