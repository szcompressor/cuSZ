#ifndef FC6B1F0D_E138_4DF5_8AC9_896900D1BE21
#define FC6B1F0D_E138_4DF5_8AC9_896900D1BE21

#include "cusz/type.h"

namespace psz {
namespace cu_hip {

template <typename T, typename Criterion, typename M = u4>
__global__ void spvn_gather(
    T* in, szt const in_len, int const radius, T* cval, M* cidx, int* cn,
    Criterion criteria)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < in_len) {
    auto d = in[tid];
    auto quantizable = criteria(d, radius);

    if (not quantizable) {
      auto cur_idx = atomicAdd(cn, 1);
      cidx[cur_idx] = tid;
      cval[cur_idx] = d;
      in[tid] = 0;
    }
  }
}

template <typename T, typename M = u4>
__global__ void spvn_scatter(T* val, M* idx, int const nnz, T* out)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < nnz) {
    int dst_idx = idx[tid];
    out[dst_idx] = val[tid];
  }
}

}  // namespace cu_hip
}  // namespace psz

#endif /* FC6B1F0D_E138_4DF5_8AC9_896900D1BE21 */
