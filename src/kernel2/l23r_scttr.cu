/**
 * @file l23r_scttr.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-05
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <stdint.h>
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>

#include "utils/cuda_err.cuh"
#include "utils/timer.h"

template <typename T>
void psz_adhoc_scttr(
    T* val, uint32_t* idx, int const n, T* out, float* milliseconds,
    cudaStream_t stream)
{
  thrust::cuda::par.on(stream);

  CREATE_CUDAEVENT_PAIR;
  START_CUDAEVENT_RECORDING(stream);

  thrust::scatter(thrust::device, val, val + n, idx, out);

  STOP_CUDAEVENT_RECORDING(stream);
  TIME_ELAPSED_CUDAEVENT(milliseconds);
  DESTROY_CUDAEVENT_PAIR;
}

template void psz_adhoc_scttr<float>(
    float* val, uint32_t* idx, const int n, float* out, float* milliseconds,
    cudaStream_t stream);

template void psz_adhoc_scttr<double>(
    double* val, uint32_t* idx, const int n, double* out, float* milliseconds,
    cudaStream_t stream);
