#ifndef PSZ_KERNEL_SPVN_HH
#define PSZ_KERNEL_SPVN_HH

#include "cusz/type.h"
#include "mem/sp_interface.h"

namespace psz::module {

template <typename T, typename M>
struct CPU_scatter {
  [[deprecated("To be replaced by kernel_v2.")]] static int kernel(
      T* val, M* idx, int nnz, T* out);

  using ValIdx = _portable::compact_cell<T, M>;
  static int kernel_v2(ValIdx* val_idx, int nnz, T* out);
};

template <typename T, typename M>
struct GPU_scatter {
  [[deprecated("To be replaced by kernel_v2.")]] static int kernel(
      T* val, M* idx, int nnz, T* out, f4* milliseconds, void* stream);

  using ValIdx = _portable::compact_cell<T, M>;
  static int kernel_v2(ValIdx* val_idx, int nnz, T* out, void* stream);
};

}  // namespace psz::module

#endif /* PSZ_KERNEL_SPVN_HH */
