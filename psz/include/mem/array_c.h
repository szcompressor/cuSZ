#ifndef B3BD768D_9F34_4E8C_8809_A79D449180B4
#define B3BD768D_9F34_4E8C_8809_A79D449180B4

#include "cusz/type.h"

// 2401 update
typedef struct __pszimpl_array {
  void* const buf;
  pszlen len3;
  pszdtype const dtype;
} __pszimpl_array;

typedef __pszimpl_array pszarray;
typedef pszarray* pszarray_mutable;

typedef struct __pszimpl_compact {
  void* const val;
  uint32_t* idx;
  uint32_t* num;
  uint32_t reserved_len;
  pszdtype const dtype;
} __pszimpl_compact;

typedef __pszimpl_compact pszcompact;
typedef pszcompact pszoutlier;
typedef pszoutlier* pszoutlier_mutable;

#endif /* B3BD768D_9F34_4E8C_8809_A79D449180B4 */
