#ifndef A851557F_29B7_4865_AC4A_B5B59930E5F6
#define A851557F_29B7_4865_AC4A_B5B59930E5F6

#include "array.h"
#include "type.h"
#include "typing.hh"

template <typename T>
struct pszarray_cxx {
  T* const buf;
  pszlen len3;

  // using type = T;
  // using psztype = typename PszType<T>::type;
};

template <typename T>
struct pszcompact_cxx {
  T* const val;
  uint32_t* idx;
  uint32_t* num;
  size_t reserved_len;

  // using type = T;
  // using psztype = typename PszType<T>::type;
};

template <typename T>
struct pszpredict_2output {
  pszarray_cxx<u4> dense;
  pszcompact_cxx<T> sparse;
};

#endif /* A851557F_29B7_4865_AC4A_B5B59930E5F6 */
