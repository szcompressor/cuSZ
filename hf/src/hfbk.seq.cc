#include "hfbk.hh"

#include <bitset>
#include <cstdint>

#include "busyheader.hh"
#include "cusz/type.h"
#include "hfbk_impl.hh"
#include "hfcanon.hh"
#include "hfword.hh"
#include "utils/timer.hh"

template <typename E, typename H>
void hf_build_and_canonize_book_serial(
    uint32_t* freq, int const bklen, H* book, uint8_t* revbook,
    int const revbook_bytes, float* time)
{
  using PW4 = PackedWordByWidth<4>;
  using PW8 = PackedWordByWidth<8>;

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

  // print
  // for (auto i = 0; i < bklen; i++) {
  //   auto pw4 = reinterpret_cast<PW4*>(book + i);
  //   cout << "old-" << i << "\t";
  //   cout << bitset<PW4::FIELDWIDTH_bits>(pw4->bits) << "\t";
  //   cout << pw4->bits << "\t";
  //   cout << bitset<PW4::FIELDWIDTH_word>(pw4->word) << "\n";
  // }

  space->input_bk() = book;  // external

  {  // part 2
    auto a = hires::now();

    space->canonize();

    auto b = hires::now();
    auto t2 = static_cast<duration_t>(b - a).count() * 1000;
    // cout << t2 << endl;
    *time += t2;
  }

  // copy to output1
  memcpy(book, space->output_bk(), bk_bytes);

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

template <typename E, typename H = uint32_t>
void hf_build_and_canonize_book_serial_v2(
    uint32_t* freq, int const bklen, uint32_t* bk4, uint8_t* revbook,
    int const revbook_bytes, float* time)
{
  using PW4 = PackedWordByWidth<4>;
  using PW8 = PackedWordByWidth<8>;

  constexpr auto TYPE_BITS = sizeof(H) * 8;
  auto bk_bytes = sizeof(H) * bklen;
  auto space_bytes = hf_space<E, H>::space_bytes(bklen);
  auto revbook_ofst = hf_space<E, H>::revbook_offset(bklen);
  auto space = new hf_canon_reference<E, H>(bklen);
  *time = 0;

  // mask the codebook to 0xff
  memset(bk4, 0xff, bk_bytes);

  // internal buffer
  auto bk8 = new uint64_t[bklen];
  memset(bk8, 0xff, sizeof(uint64_t) * bklen);

  // part 1
  {
    f4 t;
    hf_buildtree_impl1<uint64_t>(freq, bklen, bk8, &t);
    // hf_buildtree_impl2<uint64_t>(freq, bklen, bk8, &t);
    *time += t;
  }

  // resolve the issue of being longer than 32 bits
  for (auto i = 0; i < bklen; i++) {
    auto pw8 = reinterpret_cast<PW8*>(bk8 + i);
    auto pw4 = reinterpret_cast<PW4*>(bk4 + i);

    if (*(bk8 + i) == ~((uint64_t)0x0)) {
      //   // not meaningful
    }
    else {
      if (pw8->bits > pw4->FIELDWIDTH_word) {
        pw4->bits = pw4->OUTLIER_CUTOFF;
        pw4->word = 0;  // not meaningful
        cout << i << "\tlarger than FIELDWIDTH_word" << endl;
      }
      else {
        pw4->bits = pw8->bits;
        pw4->word = pw8->word;
      }
    }
  }
  // for (auto i = 0; i < bklen; i++) {
  //   auto pw4 = reinterpret_cast<PW4*>(bk4 + i);
  //   cout << "new-" << i << "\t";
  //   cout << bitset<PW4::FIELDWIDTH_bits>(pw4->bits) << "\t";
  //   cout << pw4->bits << "\t";
  //   cout << bitset<PW4::FIELDWIDTH_word>(pw4->word) << "\n";
  // }

  space->input_bk() = bk4;  // external

  {  // part 2
    auto a = hires::now();

    space->canonize();

    auto b = hires::now();
    auto t2 = static_cast<duration_t>(b - a).count() * 1000;
    // cout << t2 << endl;
    *time += t2;
  }

  // copy to output1
  memcpy(bk4, space->output_bk(), bk_bytes);

  // copy to output2
  auto offset = 0;
  memcpy(revbook, space->first(), sizeof(int) * TYPE_BITS);
  offset += sizeof(int) * TYPE_BITS;
  memcpy(revbook + offset, space->entry(), sizeof(int) * TYPE_BITS);
  offset += sizeof(int) * TYPE_BITS;
  memcpy(revbook + offset, space->keys(), sizeof(E) * bklen);

  delete space;
}

#define SPECIALIZE(E, H)                                           \
  template <>                                                      \
  void psz::hf_buildbook<SEQ, E, H>(                               \
      uint32_t * freq, int const bklen, H* book, uint8_t* revbook, \
      int const revbook_bytes, float* time, void* stream)          \
  {                                                                \
    hf_build_and_canonize_book_serial_v2<E, H>(                    \
        freq, bklen, book, revbook, revbook_bytes, time);          \
  }

SPECIALIZE(u1, u4)
SPECIALIZE(u2, u4)
SPECIALIZE(u4, u4)
// SPECIALIZE(u1, u8)
// SPECIALIZE(u2, u8)
// SPECIALIZE(u4, u8)
// SPECIALIZE(u1, ull)
// SPECIALIZE(u2, ull)
// SPECIALIZE(u4, ull)

#undef SPECIALIZE
