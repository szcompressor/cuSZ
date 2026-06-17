#ifndef PSZ_KERNEL_IMPL_SPVN_CUHIP_INL
#define PSZ_KERNEL_IMPL_SPVN_CUHIP_INL

#include "cusz/type.h"
#include "kernel.hh"
#include "mem/sp_interface.h"
#include "utils/err.hh"

namespace psz {

template <typename T, typename Criterion, typename M = u4>
[[deprecated("Superseded by the AoS version.")]] __global__ void KCU_spvn_gather(
    T* in, szt const in_len, int const radius, T* cval, M* cidx, int* cn, Criterion criteria)
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
[[deprecated("Superseded by kernel_v3_fuse.")]] __global__ void KCU_spvn_scatter(
    T* val, M* idx, int const nnz, T* out)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < nnz) {
    auto dst_idx = idx[tid];
    out[dst_idx] = val[tid];
  }
}

// Fuse the compact (val,gid) outliers into the output buffer that already holds the decoded eq.
template <typename T, typename M = u4, typename ValIdx = _ptb::compact_cell<T, M>>
__global__ void KCU_spvn_fuse(ValIdx* val_idx, int const nnz, T* out)
{
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < nnz) {
    auto [val, idx] = val_idx[tid];
    out[idx] += val;
  }
}

}  // namespace psz

template <typename T, typename M>
int psz::module::GPU_scatter<T, M>::kernel(T* val, M* idx, int const nnz, T* out, void* stream)
{
  auto grid_dim = (nnz - 1) / 128 + 1;
  psz::KCU_spvn_scatter<T, M>
      <<<grid_dim, 128, 0, (cudaStream_t)stream>>>(val, idx, nnz, out);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  return PSZ_SUCCESS;
}

template <typename T, typename M>
int psz::module::GPU_scatter<T, M>::kernel_v3_fuse(
    typename GPU_scatter<T, M>::ValIdx* val_idx, int const nnz, T* out, void* stream)
{
  auto grid_dim = (nnz - 1) / 128 + 1;
  psz::KCU_spvn_fuse<T, M>
      <<<grid_dim, 128, 0, (cudaStream_t)stream>>>(val_idx, nnz, out);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));

  return PSZ_SUCCESS;
}

#endif /* PSZ_KERNEL_IMPL_SPVN_CUHIP_INL */
