/**
 * @file spv.hip
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <hip/hip_runtime.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/tuple.h>

#include "kernel/spv.hh"
#include "utils/timer.hh"

namespace psz {

template <typename T, typename M>
void spv_gather_thrust(
    T* in, szt const in_len, T* d_val, M* d_idx, int* nnz, f4* milliseconds,
    hipStream_t stream)
{
  using thrust::placeholders::_1;

  thrust::hip::par.on(stream);
  thrust::counting_iterator<uint32_t> zero(0);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  // find out the indices
  *nnz = thrust::copy_if(
             thrust::device, zero, zero + in_len, in, d_idx, _1 != 0) -
         d_idx;

  // fetch corresponding values
  thrust::copy(
      thrust::device, thrust::make_permutation_iterator(in, d_idx),
      thrust::make_permutation_iterator(in + *nnz, d_idx + *nnz), d_val);

  STOP_GPUEVENT_RECORDING(stream);
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;
}

template <typename T, typename M>
void spv_scatter_thrust(
    T* d_val, M* d_idx, int const nnz, T* decoded, f4* milliseconds,
    hipStream_t stream)
{
  thrust::hip::par.on(stream);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  thrust::scatter(thrust::device, d_val, d_val + nnz, d_idx, decoded);

  STOP_GPUEVENT_RECORDING(stream);
  TIME_ELAPSED_GPUEVENT(milliseconds);
  DESTROY_GPUEVENT_PAIR;
}

}  // namespace psz

#define SPECIALIZE_SPV(T, M)                                               \
  template <>                                                              \
  void psz::spv_gather<HIP, T, M>(                                         \
      T * in, szt const in_len, T* d_val, M* d_idx, int* nnz,              \
      f4* milliseconds, void* stream)                                      \
  {                                                                        \
    psz::spv_gather<T, M>(                                                 \
        in, in_len, d_val, d_idx, nnz, milliseconds, (hipStream_t)stream); \
  }                                                                        \
  template <>                                                              \
  void psz::spv_scatter<HIP, T, M>(                                        \
      T * d_val, M * d_idx, int const nnz, T* decoded, f4* milliseconds,   \
      void* stream)                                                        \
  {                                                                        \
    psz::spv_scatter_cu(                                                   \
        d_val, d_idx, nnz, decoded, milliseconds, (hipStream_t)stream);    \
  }

SPECIALIZE_SPV(f4, u4)
SPECIALIZE_SPV(f8, u4)

#undef SPECIALIZE_SPV