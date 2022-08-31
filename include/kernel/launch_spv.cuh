/**
 * @file launch_sparse_vec_method.cuh
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef F4C1E2EB_2BF7_46DE_8A7D_BA4D6130A87E
#define F4C1E2EB_2BF7_46DE_8A7D_BA4D6130A87E

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "../utils/timer.hh"

template <typename T, typename M>
void launch_spv_gather(
    T*           in,
    size_t const in_len,
    T*           d_val,
    uint32_t*    d_idx,
    int&         nnz,
    float&       milliseconds,
    hipStream_t stream)
{
    using thrust::placeholders::_1;

    thrust::hip::par.on(stream);
    thrust::counting_iterator<int> zero(0);

    //cuda_timer_t t;
    //t.timer_start(stream);

    // find out the indices
    nnz = thrust::copy_if(thrust::device, zero, zero + in_len, in, d_idx, _1 != 0) - d_idx;

    // fetch corresponding values
    thrust::copy(
        thrust::device, thrust::make_permutation_iterator(in, d_idx),
        thrust::make_permutation_iterator(in + nnz, d_idx + nnz), d_val);

    //t.timer_end(stream);
    //milliseconds = t.get_time_elapsed();
}

template <typename T, typename M>
void launch_spv_scatter(T* d_val, uint32_t* d_idx, int const nnz, T* decoded, float& milliseconds, hipStream_t stream)
{
    thrust::hip::par.on(stream);
    //cuda_timer_t t;
    //t.timer_start(stream);
    thrust::scatter(thrust::device, d_val, d_val + nnz, d_idx, decoded);
    //t.timer_end(stream);
    //milliseconds = t.get_time_elapsed();
}

#endif /* F4C1E2EB_2BF7_46DE_8A7D_BA4D6130A87E */
