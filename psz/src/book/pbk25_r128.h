#ifndef PBK25_R128_H
#define PBK25_R128_H

// clang-format off
unsigned char PBK25_R128_BOOK_h[] = {
#include "pbk25_r128_book.inc"
};

#ifdef __cplusplus
constexpr
#else
static const
#endif
unsigned int PBK25_R128_BOOK_h_len = 25600;

unsigned char PBK25_R128_RVBK_h[] = {
#include "pbk25_r128_rvbk.inc"
};

#ifdef __cplusplus
constexpr
#else
static const
#endif
unsigned int PBK25_R128_RVBK_h_len = 12800;
// Per-book layout: first[32]·u4 + entry[32]·u4 + keys[256]·u1 = 512B.
// Both T=u1 and T=u2 callers pin E_storage = uint8_t in single_thread_inflate.
// clang-format on

#endif  // PBK25_R128_H
