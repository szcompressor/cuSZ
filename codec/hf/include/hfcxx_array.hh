#ifndef BD7D510D_8AA0_4DA6_93F2_6F2CC27BF1A5
#define BD7D510D_8AA0_4DA6_93F2_6F2CC27BF1A5

#include <cstddef>
#include <cstdint>

#include "hfcxx_array.hh"
#include "mem/array_cxx.h"

using namespace portable;

#define hfarray_cxx array1
#define hfcxx_array array1
#define hfcxx_compact compact_array1

template <typename Hf>
struct hfcxx_book {
  hfarray_cxx<Hf> bk;
  Hf const alt_prefix_code;  // even if u8 can use short u4 internal
  u4 const alt_bitcount;
};

template <typename Hf>
struct hfcxx_dense {
  Hf* const out;
  u4* bits;
  size_t n_part;
};

struct hfpar_description {
  const size_t sublen;
  const size_t pardeg;
};

#endif /* BD7D510D_8AA0_4DA6_93F2_6F2CC27BF1A5 */
