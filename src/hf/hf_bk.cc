#include "hf/hf_bk.hh"

#include "cusz/type.h"
#include "hf/hf_buildtree_impl1.hh"
#include "hf/hf_buildtree_impl2.hh"
#include "hf/hf_canon.hh"

template <typename E, typename H>
void hf_build_and_canonize_book_serial(
    uint32_t* freq, int const bklen, H* book, uint8_t* revbook,
    int const revbook_bytes, float* time)
{
  constexpr auto TYPE_BITS = sizeof(E) * 8;
  auto bk_bytes = sizeof(H) * bklen;
  auto space_bytes = hf_space<E, H>::space_bytes(bklen);
  auto revbook_ofst = hf_space<E, H>::revbook_offset(bklen);
  auto space = new u1[space_bytes];
  memset(space, 0, space_bytes);

  memset(book, 0xff, bk_bytes);

  hf_buildtree_impl1<H>(freq, bklen, book);
  // hf_buildtree_impl2<H>(freq, bklen, book);

  memcpy(space, book, bk_bytes);  // copy in
  canonize<E, H>(space, bklen);

  memcpy(book, space + bk_bytes, bk_bytes);              // copy out
  memcpy(revbook, space + revbook_ofst, revbook_bytes);  // copy out

  delete[] space;
}

#define SPECIALIZE(E, H)                                           \
  template <>                                                      \
  void psz::hf_buildbook<CPU, E, H>(                               \
      uint32_t * freq, int const bklen, H* book, uint8_t* revbook, \
      int const revbook_bytes, float* time, void* stream)          \
  {                                                                \
    hf_build_and_canonize_book_serial<E, H>(                       \
        freq, bklen, book, revbook, revbook_bytes, time);          \
  }

SPECIALIZE(u1, u4)
SPECIALIZE(u2, u4)
SPECIALIZE(u4, u4)
SPECIALIZE(u1, u8)
SPECIALIZE(u2, u8)
SPECIALIZE(u4, u8)

#undef SPECIALIZE
