#ifndef _PORTABLE_CXX_TYPE_HH
#define _PORTABLE_CXX_TYPE_HH

#include "c_type.h"

namespace _ptb {

// Result of parsing a dimension string (x,y,z order).
// ndim is the number of components found in the string (1–3).
// Projects should alias this for their own API:
//   using psz_xyz_t = _ptb::xyz_t;
struct xyz_t {
  _ptb_len3 len;
  int       ndim;
};

// Compare two structs with x/y/z fields by value.
template <typename TRIO>
inline bool val_eq(TRIO a, TRIO b)
{
  return (a.x == b.x) and (a.y == b.y) and (a.z == b.z);
}

}  // namespace _ptb

#endif  // _PORTABLE_CXX_TYPE_HH
