#include "hfbuf.h"

size_t phf_reverse_book_bytes(uint16_t bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES)
{
  static const int CELL_BITWIDTH = BK_UNIT_BYTES * 8;
  return BK_UNIT_BYTES * (2 * CELL_BITWIDTH) + SYM_BYTES * bklen;
}

uint8_t* phf_allocate_reverse_book(uint16_t bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES)
{
  return new uint8_t[phf_reverse_book_bytes(bklen, BK_UNIT_BYTES, SYM_BYTES)];
}