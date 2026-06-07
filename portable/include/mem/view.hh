#ifndef _PORTABLE_MEM_VIEW_H
#define _PORTABLE_MEM_VIEW_H

#include "../c_type.h"

namespace _ptb {

// field-matching, assuming the same xyz-order
template <class type1_len3, class type2_len3>
type2_len3 _len2len(type1_len3 len)
{
  using E = decltype(type2_len3::x);
  return type2_len3{static_cast<E>(len.x), static_cast<E>(len.y), static_cast<E>(len.z)};
}

// clang-format off
template <         class _len3> struct box  { _len3 begin, end; };
template <class T, class _len3> struct view { T* ptr; _len3 extent, leap; };
// clang-format on

// x-fastest (LayoutLeft) strides for a parent of size `extent`.
template <class _len3>
_len3 leap_of(_len3 extent)
{
  return _len3{1, extent.x, extent.x * extent.y};
}

// Whole-domain box [0, extent).
template <class _len3>
box<_len3> whole(_len3 extent)
{
  return box<_len3>{_len3{0, 0, 0}, extent};
}

// Bind box `b` onto `base` in a parent of size `parent`: offset ptr + sub-extent +
// parent strides (an unmanaged Kokkos-style subview).
template <class T, class _len3>
view<T, _len3> bind(T* base, _len3 parent, box<_len3> b)
{
  auto  leap = leap_of(parent);
  auto  off  = b.begin.x * leap.x + b.begin.y * leap.y + b.begin.z * leap.z;
  _len3 extent{b.end.x - b.begin.x, b.end.y - b.begin.y, b.end.z - b.begin.z};
  return view<T, _len3>{base + off, extent, leap};
}

// Whole-domain view over `base` (the root subview): packed leap, no offset.
template <class T, class _len3>
view<T, _len3> make_view(T* base, _len3 extent)
{
  return bind(base, extent, whole(extent));
}

// Read-only whole-domain view: take a mutable base, hand back view<const T>.
template <class T, class _len3>
view<const T, _len3> make_const_view(T* base, _len3 extent)
{
  return bind(static_cast<const T*>(base), extent, whole(extent));
}

}  // namespace _ptb

namespace _ptb::host {

using box = box<_ptb_len3>;

template <class T>
using view = ::_ptb::view<T, _ptb_len3>;

}  // namespace _ptb::host

#endif /* _PORTABLE_MEM_VIEW_H */
