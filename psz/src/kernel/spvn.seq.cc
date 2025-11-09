#include "kernel/spvn.hh"

#include "c_type.h"

namespace psz {

template <typename T, typename M>
int psz::module::CPU_scatter<T, M>::kernel(T* val, M* idx, int const nnz, T* out)
{
  for (auto tid = 0; tid < nnz; tid++) {
    int dst_idx = idx[tid];
    out[dst_idx] = val[tid];
  }

  return PSZ_SUCCESS;
}

template <typename T, typename M>
int psz::module::CPU_scatter<T, M>::kernel_v2(
    typename CPU_scatter<T, M>::ValIdx* val_idx, int const nnz, T* out)
{
  for (auto tid = 0; tid < nnz; tid++) {
    auto [val, idx] = val_idx[tid];
    out[idx] = val;
  }

  return PSZ_SUCCESS;
}

}  // namespace psz

template struct psz::module::CPU_scatter<f4, u4>;
template struct psz::module::CPU_scatter<f8, u4>;