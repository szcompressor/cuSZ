#ifndef A851557F_29B7_4865_AC4A_B5B59930E5F6
#define A851557F_29B7_4865_AC4A_B5B59930E5F6

#include "cusz/type.h"
#include "typing.hh"

namespace portable {

// dense array, 3d
template <typename T>
struct array3 {
  T* const buf;
  psz_len3 len3;
};

// dense array, 1d
template <typename T>
struct array1 {
  T* const buf;
  size_t len;
};

// sparse array, 1d
template <typename T>
struct compact_array1 {
  T* const val;
  uint32_t* idx;
  uint32_t* num;
  uint32_t* host_num;
  size_t reserved_len;
};

template <typename T>
struct pszpredict_2output {
  array3<u4> dense;
  compact_array1<T> sparse;
};

}  // namespace portable

#endif /* A851557F_29B7_4865_AC4A_B5B59930E5F6 */
