/**
 * @file hfcanon.seq.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-07-29
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */


#include "busyheader.hh"
#include "cusz/type.h"
#include "hf/hfcanon.hh"
#include "hf/hfword.hh"

template <typename E, typename H>
int canonize(u1* bin, uint32_t const bklen)
{
  constexpr auto TYPE_BITS = sizeof(E) * 8;

  // layout: (1) in-cb, (2) out-cb, (3) canon, (4) numl, (5) iterby
  //         (6) first, (7) entry, (8) keys
  // (6) to (8) make up "revbook"

  auto seg1 = sizeof(H) * (3 * bklen);
  auto seg2 = sizeof(u4) * (4 * TYPE_BITS);

  H* icb = (H*)bin;
  H* ocb = icb + bklen;
  H* canon = icb + bklen * 2;
  auto numl = (u4*)(bin + seg1);
  auto iterby = numl + TYPE_BITS;
  auto first = numl + TYPE_BITS * 2;
  auto entry = numl + TYPE_BITS * 3;
  auto keys = (E*)(bin + seg1 + seg2);

  using PW = PackedWordByWidth<sizeof(H)>;

  constexpr auto FILL = ~((H)0x0);

  // states
  int max_l = 0;

  for (auto c = icb; c < icb + bklen; c++) {
    auto pw = (PW*)c;
    int l = pw->bits;

    if (*c != FILL) {
      max_l = l > max_l ? l : max_l;
      numl[l] += 1;
    }
  }

  for (int i = 1; i < TYPE_BITS; i++) entry[i] = numl[i - 1];
  for (int i = 1; i < TYPE_BITS; i++) entry[i] += entry[i - 1];
  for (auto i = 0; i < TYPE_BITS; i++) iterby[i] = entry[i];

  //////// first code

  first[max_l] = 0;
  for (int l = max_l - 1; l >= 1; l--) {
    first[l] = static_cast<int>((first[l + 1] + numl[l + 1]) / 2.0 + 0.5);
  }
  first[0] = 0xff;  // no off-by-one error

  // /* debug */ for (auto l = 1; l <= max_l; l++)
  //   printf("l: %3d\tnuml: %3d\tfirst: %3d\n", l, numl[l], first[l]);

  for (auto i = 0; i < bklen; i++) canon[i] = FILL;
  for (auto i = 0; i < bklen; i++) ocb[i] = FILL;

  // Reverse Codebook Generation
  for (auto i = 0; i < bklen; i++) {
    auto c = icb[i];
    uint8_t l = reinterpret_cast<PW*>(&c)->bits;

    if (c != FILL) {
      canon[iterby[l]] = static_cast<H>(first[l] + iterby[l] - entry[l]);
      keys[iterby[l]] = i;

      reinterpret_cast<PW*>(&(canon[iterby[l]]))->bits = l;
      iterby[l]++;
    }
  }

  for (auto i = 0; i < bklen; i++)
    if (canon[i] != FILL) ocb[keys[i]] = canon[i];

  return 0;
}

#define INIT(E, H) template int canonize<E, H>(u1*, uint32_t const);

INIT(u1, u4)
INIT(u2, u4)
INIT(u4, u4)
INIT(u1, u8)
INIT(u2, u8)
INIT(u4, u8)

#undef INIT

///////////////////////////////////////////////////////////////////////////////

template <typename E, typename H>
int hf_canon_reference<E, H>::canonize()
{
  using Space = hf_canon_reference<E, H>;
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

  // /* debug */ for (auto l = 1; l <= max_l; l++)
  //   printf("l: %3d\tnuml: %3d\tfirst: %3d\n", l, numl(l), first(l));

  for (auto i = 0; i < booklen; i++) canon(i) = ~((H)0x0);
  for (auto i = 0; i < booklen; i++) ocb(i) = ~((H)0x0);

  // Reverse Codebook Generation
  for (auto i = 0; i < booklen; i++) {
    auto c = icb(i);
    uint8_t l = reinterpret_cast<PW*>(&c)->bits;

    if (c != FILL) {
      canon(iterby(l)) = static_cast<H>(first(l) + iterby(l) - entry(l));
      keys(iterby(l)) = i;

      reinterpret_cast<PW*>(&(canon(iterby(l))))->bits = l;
      iterby(l)++;
    }
  }

  for (auto i = 0; i < booklen; i++)
    if (canon(i) != FILL) ocb(keys(i)) = canon(i);

  return 0;
}

// #define INIT(E, H) template int hf_canonize_serial<E, H>(void*);
#define INIT(E, H) template class hf_canon_reference<E, H>;

INIT(u1, u4)
INIT(u2, u4)
INIT(u4, u4)
INIT(u1, u8)
INIT(u2, u8)
INIT(u4, u8)
INIT(u1, ull)
INIT(u2, ull)
INIT(u4, ull)


#undef INIT
