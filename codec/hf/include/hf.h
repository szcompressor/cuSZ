// 24-05-30 by Jiannan Tian

#ifndef HF_H
#define HF_H
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "hf_type.h"

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
phferr capi_phf_release(phf_codec*);
// TODO hist_len is not necessary; alternatively, it can force check size.
phferr capi_phf_buildbook(phf_codec* codec, uint32_t* d_hist, phf_stream_t);
phferr capi_phf_encode(
    phf_codec* codec, void* in, size_t const inlen, uint8_t** encoded, size_t* enc_bytes,
    phf_stream_t);
phferr capi_phf_decode(phf_codec* codec, uint8_t* encoded, void* decoded, phf_stream_t);

// helpers
size_t capi_phf_reverse_book_bytes(u2 bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES);
uint8_t* capi_phf_allocate_reverse_book(u2 bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES);

#ifdef __cplusplus
}
#endif
#endif