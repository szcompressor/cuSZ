#ifndef D0A3CB01_B969_44CE_915B_4920812B8B5E
#define D0A3CB01_B969_44CE_915B_4920812B8B5E
#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#include "cusz/type.h"

void* psz_make_timerecord();
void psz_review_timerecord(void* _r, psz_header* h);
void psz_review_cr(psz_header* h);
void psz_review_compression(void* r, psz_header* h);
void psz_review_decompression(void* r, size_t bytes);

#ifdef __cplusplus
}
#endif
#endif /* D0A3CB01_B969_44CE_915B_4920812B8B5E */
