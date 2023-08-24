/**
 * @file extrema_cu.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-19
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>

#include "cusz/type.h"
#include "detail/extrema_g.inl"
#include "stat/compare_cu.hh"
#include "utils/err.hh"

template <typename T>
void psz::cuda_extrema(T* in, size_t len, T res[4])
{
  static const int MINVAL = 0;
  static const int MAXVAL = 1;
  //   static const int AVGVAL = 2;  // TODO
  static const int RNG = 3;

  // TODO use external stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto chunk = 32768;
  auto nworker = 128;
  auto R = chunk / nworker;

  T h_min, h_max, failsafe;
  T *d_minel, *d_maxel;

  CHECK_GPU(cudaMalloc(&d_minel, sizeof(T)));
  CHECK_GPU(cudaMalloc(&d_maxel, sizeof(T)));

  // failsafe init
  CHECK_GPU(cudaMemcpy(&failsafe, in, sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_GPU(cudaMemcpy(d_minel, in, sizeof(T), cudaMemcpyDeviceToDevice));
  CHECK_GPU(cudaMemcpy(d_maxel, in, sizeof(T), cudaMemcpyDeviceToDevice));

  // launch
  psz::extrema_cu<T><<<div(len, chunk), nworker, sizeof(T) * 2, stream>>>(
      in, len, d_minel, d_maxel, failsafe, R);

  cudaStreamSynchronize(stream);

  // collect results
  CHECK_GPU(cudaMemcpy(&h_min, d_minel, sizeof(T), cudaMemcpyDeviceToHost));
  CHECK_GPU(cudaMemcpy(&h_max, d_maxel, sizeof(T), cudaMemcpyDeviceToHost));

  res[MINVAL] = h_min;
  res[MAXVAL] = h_max;
  res[RNG] = h_max - h_min;

  CHECK_GPU(cudaFree(d_minel));
  CHECK_GPU(cudaFree(d_maxel));

  cudaStreamDestroy(stream);
}

template void psz::cuda_extrema<f4>(f4* d_ptr, szt len, f4 res[4]);
template void psz::cuda_extrema<f8>(f8* d_ptr, szt len, f8 res[4]);