/**
 * @file spv_gpu.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-22
 * (update) 2022-10-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef F88E11A6_6B61_4C6F_8B2E_30EEAAB4D204
#define F88E11A6_6B61_4C6F_8B2E_30EEAAB4D204

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/tuple.h>

#include "utils/timer.h"

namespace psz {
namespace detail {

template <typename T, typename M>
void spv_gather(
    T*           in,
    size_t const in_len,
    T*           d_val,
    uint32_t*    d_idx,
    int*         nnz,
    float*       milliseconds,
    cudaStream_t stream)
{
    using thrust::placeholders::_1;

    thrust::cuda::par.on(stream);
    thrust::counting_iterator<uint32_t> zero(0);

    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream);

    // find out the indices
    *nnz = thrust::copy_if(thrust::device, zero, zero + in_len, in, d_idx, _1 != 0) - d_idx;

    // fetch corresponding values
    thrust::copy(
        thrust::device, thrust::make_permutation_iterator(in, d_idx),
        thrust::make_permutation_iterator(in + *nnz, d_idx + *nnz), d_val);

    STOP_CUDAEVENT_RECORDING(stream);
    TIME_ELAPSED_CUDAEVENT(milliseconds);
    DESTROY_CUDAEVENT_PAIR;
}

template <typename T, typename M>
void spv_scatter(T* d_val, uint32_t* d_idx, int const nnz, T* decoded, float* milliseconds, cudaStream_t stream)
{
    thrust::cuda::par.on(stream);

    CREATE_CUDAEVENT_PAIR;
    START_CUDAEVENT_RECORDING(stream);

    thrust::scatter(thrust::device, d_val, d_val + nnz, d_idx, decoded);

    STOP_CUDAEVENT_RECORDING(stream);
    TIME_ELAPSED_CUDAEVENT(milliseconds);
    DESTROY_CUDAEVENT_PAIR;
}

}  // namespace detail
}  // namespace psz

#endif /* F88E11A6_6B61_4C6F_8B2E_30EEAAB4D204 */
