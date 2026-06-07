#ifndef _PORTABLE_MEM_VIEW_H
#define _PORTABLE_MEM_VIEW_H

#include "../c_type.h"

namespace _ptb {

template <typename T, typename _len3>
struct view {
  T*    data;
  _len3 extent;
  _len3 leap;
};

}  // namespace _ptb

namespace _ptb::host {

template <typename T>
using view = ::_ptb::view<T, _ptb_len3>;

}  // namespace _ptb::host

#endif /* _PORTABLE_MEM_VIEW_H */
