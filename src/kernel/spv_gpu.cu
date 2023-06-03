/**
 * @file spv_gpu.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/tuple.h>
#include "kernel/spv_gpu.hh"
#include "utils/timer.h"

namespace psz {

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

}  // namespace psz

#define SPV(Tliteral, Mliteral, T, M)                                                                                \
    template void psz::spv_gather<T, M>(                                                                             \
        T * in, size_t const in_len, T* d_val, uint32_t* d_idx, int* nnz, float* milliseconds, cudaStream_t stream); \
                                                                                                                     \
    template void psz::spv_scatter<T, M>(                                                                            \
        T * d_val, uint32_t * d_idx, int const nnz, T* decoded, float* milliseconds, cudaStream_t stream);

SPV(ui8, ui32, uint8_t, uint32_t)
SPV(ui16, ui32, uint16_t, uint32_t)
SPV(ui32, ui32, uint32_t, uint32_t)
SPV(ui64, ui32, uint64_t, uint32_t)
SPV(fp32, ui32, float, uint32_t)
SPV(fp64, ui32, double, uint32_t)

#undef SPV
