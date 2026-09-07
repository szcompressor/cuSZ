

#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "c_type.h"
#include "hf.h"
#include "hf_impl.hh"

using std::cerr;
using std::cout;
using std::endl;

template <typename E, typename H>
void phf_CPU_build_canonized_codebook_v2(
    u4* freq, int const bklen, u4* bk4, uint8_t* revbook, int const revbook_bytes,
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
  auto bk8 = new u8[bklen];

  // Halve-and-retry the histogram until every code fits PW4::FIELD_CODE.
  auto local_freq = new u4[bklen];
  memcpy(local_freq, freq, sizeof(u4) * bklen);
  for (;;) {
    memset(bk8, 0xff, sizeof(u8) * bklen);
    phf_CPU_build_codebook_v1<u8>(local_freq, bklen, bk8);
    int max_bitcount = 0;
    for (auto i = 0; i < bklen; i++) {
      if (bk8[i] == ~(u8)0x0) continue;
      auto bitcount = reinterpret_cast<PW8*>(bk8 + i)->bitcount;
      if ((int)bitcount > max_bitcount) max_bitcount = (int)bitcount;
    }
    if (max_bitcount <= PW4::FIELD_CODE) break;
    for (auto i = 0; i < bklen; i++)
      if (local_freq[i] > 0) local_freq[i] = local_freq[i] > 1 ? local_freq[i] >> 1 : 1;
  }
  delete[] local_freq;

  // narrow to PW4; the rescale loop above guarantees every code now fits
  for (auto i = 0; i < bklen; i++) {
    auto pw8 = reinterpret_cast<PW8*>(bk8 + i);
    auto pw4 = reinterpret_cast<PW4*>(bk4 + i);

    if (*(bk8 + i) == ~((u8)0x0)) {
      // not meaningful
    }
    else {
      if (pw8->bitcount > pw4->FIELD_CODE) {
        cerr << "phf_CPU_build_canonized_codebook_v2: rescale invariant broken at " << i << endl;
        abort();
      }
      pw4->bitcount = pw8->bitcount;
      pw4->prefix_code = pw8->prefix_code;
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
    space->canonize();
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

#define INSTANTIATE_PHF_CPU_BUILD_CANONICAL(E, H)                                     \
  template void phf_CPU_build_canonized_codebook_v2<E, H>(                            \
      u4 * freq, int const bklen, H* book, uint8_t* revbook, int const revbook_bytes, \
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
{ return new uint8_t[phf_reverse_book_bytes(bklen, BK_UNIT_BYTES, SYM_BYTES)]; }