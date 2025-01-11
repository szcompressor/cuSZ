#ifndef PHF_HFBUF_H
#define PHF_HFBUF_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#include "c_type.h"

size_t phf_reverse_book_bytes(u2 bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES);
uint8_t* phf_allocate_reverse_book(u2 bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES);

#ifdef __cplusplus
}
#endif

#endif /* PHF_HFBUF_H */
