/**
 * @file hf_canon.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-07-29
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "hf/hfserial_canon.hh"

#include "busyheader.hh"
#include "cusz/type.h"
#include "hf/hf_word.hh"

template <typename E, typename H>
int hf_canon_space<E, H>::canonize()
{
  using Space = hf_canon_space<E, H>;
  using PW = PackedWordByWidth<sizeof(H)>;

  constexpr auto FILL = ~((H)0x0);

  // states
  int max_l = 0;

  for (auto c = icb(); c < icb() + booklen; c++) {
    auto pw = (PW*)c;
    int l = pw->bits;

    if (*c != FILL) {
      max_l = l > max_l ? l : max_l;
      numl(l) += 1;
    }
  }

  for (int i = 1; i < Space::TYPE_BITS; i++) entry(i) = numl(i - 1);
  for (int i = 1; i < Space::TYPE_BITS; i++) entry(i) += entry(i - 1);
  for (auto i = 0; i < Space::TYPE_BITS; i++) iterby(i) = entry(i);

  //////// first code

  first(max_l) = 0;
  for (int l = max_l - 1; l >= 1; l--) {
    first(l) = static_cast<int>((first(l + 1) + numl(l + 1)) / 2.0 + 0.5);
  }
  first(0) = 0xff;  // no off-by-one error

  /* debug */ for (auto l = 1; l <= max_l; l++)
    printf("l: %3d\tnuml: %3d\tfirst: %3d\n", l, numl(l), first(l));

  for (auto i = 0; i < booklen; i++) canon(i) = ~((H)0x0);
  for (auto i = 0; i < booklen; i++) ocb(i) = ~((H)0x0);

  // Reverse Codebook Generation
  for (auto i = 0; i < booklen; i++) {
    auto c = icb(i);
    uint8_t l = reinterpret_cast<PW*>(&c)->bits;

    if (c != FILL) {
      canon(iterby(l)) = static_cast<H>(first(l) + iterby(l) - entry(l));
      keys(iterby(l)) = i;

      // printf(
      //     "i: %4d, l: %4d\t"
      //     "iterby(l) - entry(l): %4d\t"
      //     "canon(iterby(l)): ",
      //     i, l, iterby(l) - entry(l));
      // cout << bitset<32>(canon(iterby(l))) << endl;

      reinterpret_cast<PW*>(&(canon(iterby(l))))->bits = l;
      iterby(l)++;
    }
  }

  for (auto i = 0; i < booklen; i++)
    if (canon(i) != FILL) ocb(keys(i)) = canon(i);

  return 0;
}

// #define INIT(E, H) template int hf_canonize_serial<E, H>(void*);
#define INIT(E, H) template class hf_canon_space<E, H>;

INIT(uint8_t, uint32_t)
INIT(uint16_t, uint32_t)
INIT(uint32_t, uint32_t)
INIT(uint8_t, uint64_t)
INIT(uint16_t, uint64_t)
INIT(uint32_t, uint64_t)

#undef INIT
