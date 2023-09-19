#include "busyheader.hh"
#include "cusz/type.h"
#include "hf/hfbk.hh"
#include "hf/hfbk_impl.hh"
#include "hf/hfcanon.hh"
#include "utils/timer.hh"

template <typename E, typename H>
void hf_build_and_canonize_book_serial(
    uint32_t* freq, int const bklen, H* book, uint8_t* revbook,
    int const revbook_bytes, float* time)
{
  constexpr auto TYPE_BITS = sizeof(H) * 8;
  auto bk_bytes = sizeof(H) * bklen;
  auto space_bytes = hf_space<E, H>::space_bytes(bklen);
  auto revbook_ofst = hf_space<E, H>::revbook_offset(bklen);
  auto space = new hf_canon_reference<E, H>(bklen);
  *time = 0;

  // mask the codebook to 0xff
  memset(book, 0xff, bk_bytes);

  // part 1
  {
    f4 t;
    hf_buildtree_impl1<H>(freq, bklen, book, &t);
    // hf_buildtree_impl2<H>(freq, bklen, book, &t);
    // cout << t << endl;
    *time += t;
  }

  space->icb() = book;  // external

  {  // part 2
    auto a = hires::now();

    space->canonize();

    auto b = hires::now();
    auto t2 = static_cast<duration_t>(b - a).count() * 1000;
    // cout << t2 << endl;
    *time += t2;
  }

  // copy to output1
  memcpy(book, space->ocb(), bk_bytes);

  // copy to output2
  auto offset = 0;
  memcpy(revbook, space->first(), sizeof(int) * TYPE_BITS);
  offset += sizeof(int) * TYPE_BITS;
  memcpy(revbook + offset, space->entry(), sizeof(int) * TYPE_BITS);
  offset += sizeof(int) * TYPE_BITS;
  memcpy(revbook + offset, space->keys(), sizeof(E) * bklen);

  // memcpy(space, book, bk_bytes);  // copy in
  // canonize<E, H>(space, bklen);
  // memcpy(book, space + bk_bytes, bk_bytes);              // copy out
  // memcpy(revbook, space + revbook_ofst, revbook_bytes);  // copy out
  delete space;
}

#define SPECIALIZE(E, H)                                           \
  template <>                                                      \
  void psz::hf_buildbook<SEQ, E, H>(                               \
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
SPECIALIZE(u1, ull)
SPECIALIZE(u2, ull)
SPECIALIZE(u4, ull)

#undef SPECIALIZE
