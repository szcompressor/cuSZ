// 24-05-31

#include "cusz/header.h"

#include <stdio.h>

#include "cusz/type.h"

psz_len3 pszheader_len(psz_header* h) { return h->len; }

size_t pszheader_len_linear(psz_header* h) { return h->len.x * h->len.y * h->len.z; }

size_t pszheader_segments(psz_header* h)
{
  printf("sizeof(psz_header): %lu\n", sizeof(psz_header));
  printf("sizeof(psz_interp_params): %lu\n", sizeof(psz_interp_params));
  printf("sizeof(psz_rc2): %lu\n", sizeof(psz_rc2));
  printf("sizeof(psz_pipeline): %lu\n", sizeof(psz_pipeline));
  return sizeof(psz_header);
}

size_t pszheader_filesize(psz_header* h)
{
  int END = sizeof(h->entry) / sizeof(h->entry[0]);
  return h->entry[END - 1];
}

size_t pszheader_uncompressed_len(psz_header* h) { return pszheader_len_linear(h); }

size_t pszheader_compressed_bytes(psz_header* h) { return pszheader_filesize(h); }