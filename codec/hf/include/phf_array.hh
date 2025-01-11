#ifndef BD7D510D_8AA0_4DA6_93F2_6F2CC27BF1A5
#define BD7D510D_8AA0_4DA6_93F2_6F2CC27BF1A5

#include <cstddef>
#include <cstdint>

#include "mem/cxx_array.h"

namespace phf {

template <typename T>
using array = _portable::array1<T>;

template <typename T>
using sparse = _portable::compact_array1<T>;

template <typename Hf>
struct book {
  Hf* bk;
  u2 bklen;
  Hf const alt_prefix_code;  // even if u8 can use short u4 internal
  u4 const alt_bitcount;
};

template <typename Hf>
struct dense {
  Hf* const out;
  u4* bits;
  size_t n_part;
};

struct par_config {
  const size_t sublen;
  const size_t pardeg;
};

}  // namespace phf

#endif /* BD7D510D_8AA0_4DA6_93F2_6F2CC27BF1A5 */
