#ifndef BD7D510D_8AA0_4DA6_93F2_6F2CC27BF1A5
#define BD7D510D_8AA0_4DA6_93F2_6F2CC27BF1A5

#include <cstddef>

template <typename T>
struct hfarray_cxx {
  T* const buf;
  size_t len;
};

struct hfpar_description {
  const size_t sublen;
  const size_t pardeg;
};

#endif /* BD7D510D_8AA0_4DA6_93F2_6F2CC27BF1A5 */
