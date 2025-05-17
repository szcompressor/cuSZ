// 24-05-31

#include "cusz/header.h"

#include "cusz/type.h"

psz_len3 pszheader_len3(psz_header* h)
{
  psz_len3 l3;
  l3.x = h->x, l3.y = h->y, l3.y = h->z;
  return l3;
}

size_t pszheader_linear_len(psz_header* h) { return h->x * h->y * h->z; }

size_t pszheader_filesize(psz_header* h)
{
  int END = sizeof(h->entry) / sizeof(h->entry[0]);
  return h->entry[END - 1];
}

size_t pszheader_uncompressed_len(psz_header* h) { return pszheader_linear_len(h); }

size_t pszheader_compressed_bytes(psz_header* h) { return pszheader_filesize(h); }