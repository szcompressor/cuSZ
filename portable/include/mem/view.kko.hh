#ifndef _PORTABLE_MEM_VIEW_CUH
#define _PORTABLE_MEM_VIEW_CUH

#if defined(_PORTABLE_USE_KOKKOS)

namespace _ptb::kokkos {

template <typename T>
using view = Kokkos::View<T*, Kokkos::LayoutLeft>;

template <typename T>
using view1 = Kokkos::View<T*, Kokkos::LayoutLeft>;

template <typename T>
using view2 = Kokkos::View<T**, Kokkos::LayoutLeft>;

template <typename T>
using view3 = Kokkos::View<T***, Kokkos::LayoutLeft>;

}  // namespace _ptb::kokkos

#endif

#endif /* _PORTABLE_MEM_VIEW_CUH */
