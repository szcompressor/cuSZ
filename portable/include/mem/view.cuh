#ifndef _PORTABLE_MEM_VIEW_CUH
#define _PORTABLE_MEM_VIEW_CUH

#include "view.hh"

#if defined(_PORTABLE_USE_CUDA)

#include "cuda_runtime.h"

namespace _ptb::cuda {

using box = ::_ptb::box<dim3>;

template <typename T>
using view = ::_ptb::view<T, dim3>;

}  // namespace _ptb::cuda

#endif

#endif /* _PORTABLE_MEM_VIEW_CUH */
