#include <bitset>
#include <cstdint>

#include "busyheader.hh"
#include "cusz/type.h"
#include "cxx_hfbk.h"
#include "hf.h"
#include "hfbk_impl.hh"
#include "hfcanon.hh"
#include "hfword.hh"
#include "utils/timer.hh"

template <typename E, typename H>
void phf_CPU_build_canonized_codebook_v1(
    uint32_t* freq, int const bklen, H* book, uint8_t* revbook, int const revbook_bytes,
    float* milliseconds)
{
  using PW4 = HuffmanWord<4>;
  using PW8 = HuffmanWord<8>;

  constexpr auto TYPE_BITS = sizeof(H) * 8;
  auto bk_bytes = sizeof(H) * bklen;
  auto space_bytes = hf_space<E, H>::space_bytes(bklen);
  auto revbook_ofst = hf_space<E, H>::revbook_offset(bklen);
  auto space = new hf_canon_reference<E, H>(bklen);
  if (milliseconds) *milliseconds = 0;

  // mask the codebook to 0xff
  memset(book, 0xff, bk_bytes);

  // part 1
  {
    auto a = hires::now();

    phf_CPU_build_codebook_v1<H>(freq, bklen, book);
    // phf_CPU_build_codebook_v2<H>(freq, bklen, book);

    auto z = hires::now();
    if (milliseconds) *milliseconds += static_cast<duration_t>(z - a).count() * 1000;
  }

  // print
  // for (auto i = 0; i < bklen; i++) {
  //   auto pw4 = reinterpret_cast<PW4*>(book + i);
  //   cout << "old-" << i << "\t";
  //   cout << bitset<PW4::FIELD_BITCOUNT>(pw4->bitcount) << "\t";
  //   cout << pw4->bitcount << "\t";
  //   cout << bitset<PW4::FIELD_CODE>(pw4->prefix_code) << "\n";
  // }

  space->input_bk() = book;  // external

  {  // part 2
    auto a = hires::now();

    space->canonize();

    auto b = hires::now();
    auto t2 = static_cast<duration_t>(b - a).count() * 1000;
    // cout << t2 << endl;
    if (milliseconds) *milliseconds += t2;
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

template <typename E, typename H>
void phf_CPU_build_canonized_codebook_v2(
    uint32_t* freq, int const bklen, uint32_t* bk4, uint8_t* revbook, int const revbook_bytes,
    float* milliseconds)
{
  using PW4 = HuffmanWord<4>;
  using PW8 = HuffmanWord<8>;

  constexpr auto TYPE_BITS = sizeof(H) * 8;
  auto bk_bytes = sizeof(H) * bklen;
  auto space_bytes = hf_space<E, H>::space_bytes(bklen);
  auto revbook_ofst = hf_space<E, H>::revbook_offset(bklen);
  auto space = new hf_canon_reference<E, H>(bklen);
  if (milliseconds) *milliseconds = 0;

  // mask the codebook to 0xff
  memset(bk4, 0xff, bk_bytes);

  // internal buffer
  auto bk8 = new uint64_t[bklen];
  memset(bk8, 0xff, sizeof(uint64_t) * bklen);

  // part 1
  {
    auto a = hires::now();

    phf_CPU_build_codebook_v1<uint64_t>(freq, bklen, bk8);
    // phf_CPU_build_codebook_v2<uint64_t>(freq, bklen, bk8);

    auto z = hires::now();
    auto t1 = static_cast<duration_t>(z - a).count() * 1000;
    // cout << t1 << endl;
    if (milliseconds) *milliseconds += t1;
  }

  // resolve the issue of being longer than 32 bits
  for (auto i = 0; i < bklen; i++) {
    auto pw8 = reinterpret_cast<PW8*>(bk8 + i);
    auto pw4 = reinterpret_cast<PW4*>(bk4 + i);

    if (*(bk8 + i) == ~((uint64_t)0x0)) {
      //   // not meaningful
    }
    else {
      if (pw8->bitcount > pw4->FIELD_CODE) {
        pw4->bitcount = pw4->OUTLIER_CUTOFF;
        pw4->prefix_code = 0;  // not meaningful
        cout << i << "\tlarger than FIELD_CODE" << endl;
      }
      else {
        pw4->bitcount = pw8->bitcount;
        pw4->prefix_code = pw8->prefix_code;
      }
    }
  }
  // for (auto i = 0; i < bklen; i++) {
  //   auto pw4 = reinterpret_cast<PW4*>(bk4 + i);
  //   cout << "new-" << i << "\t";
  //   cout << bitset<PW4::FIELD_BITCOUNT>(pw4->bitcount) << "\t";
  //   cout << pw4->bitcount << "\t";
  //   cout << bitset<PW4::FIELD_CODE>(pw4->prefix_code) << "\n";
  // }

  space->input_bk() = bk4;  // external

  {  // part 2
    auto a = hires::now();

    space->canonize();

    auto z = hires::now();
    auto t2 = static_cast<duration_t>(z - a).count() * 1000;
    // cout << t2 << endl;
    if (milliseconds) *milliseconds += t2;
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

#define INSTANTIATE_PHF_CPU_BUILD_CANONICAL(E, H)                                           \
  template void phf_CPU_build_canonized_codebook_v2<E, H>(                                  \
      uint32_t * freq, int const bklen, H* book, uint8_t* revbook, int const revbook_bytes, \
      float* milliseconds);

INSTANTIATE_PHF_CPU_BUILD_CANONICAL(u1, u4)
INSTANTIATE_PHF_CPU_BUILD_CANONICAL(u2, u4)
INSTANTIATE_PHF_CPU_BUILD_CANONICAL(u4, u4)
// INSTANTIATE_PHF_CPU_BUILD_CANONICAL(u1, u8)
// INSTANTIATE_PHF_CPU_BUILD_CANONICAL(u2, u8)
// INSTANTIATE_PHF_CPU_BUILD_CANONICAL(u4, u8)
// INSTANTIATE_PHF_CPU_BUILD_CANONICAL(u1, ull)
// INSTANTIATE_PHF_CPU_BUILD_CANONICAL(u2, ull)
// INSTANTIATE_PHF_CPU_BUILD_CANONICAL(u4, ull)

#undef INSTANTIATE_PHF_CPU_BUILD_CANONICAL

size_t phf_reverse_book_bytes(uint16_t bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES)
{
  static const int CELL_BITWIDTH = BK_UNIT_BYTES * 8;
  return BK_UNIT_BYTES * (2 * CELL_BITWIDTH) + SYM_BYTES * bklen;
}

uint8_t* phf_allocate_reverse_book(uint16_t bklen, size_t BK_UNIT_BYTES, size_t SYM_BYTES)
{
  return new uint8_t[phf_reverse_book_bytes(bklen, BK_UNIT_BYTES, SYM_BYTES)];
}