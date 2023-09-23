/**
 * @file timer.hh
 * @author Jiannan Tian
 * @brief High-resolution timer wrapper from <chrono> and util functions for
 * timing both CPU and CUDA function
 * @version 0.2
 * @date 2021-01-05
 * (created) 2019-08-26 (rev) 2021-12-23 (rev) 22-10-31
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#ifndef UTILS_TIMER_HH
#define UTILS_TIMER_HH

#include <chrono>
#include <utility>

using hires = std::chrono::high_resolution_clock;
using duration_t = std::chrono::duration<double>;
using hires_clock_t = std::chrono::time_point<hires>;

struct psztime;
typedef struct psztime psztime;
typedef struct psztime psz_cputimer;

#include "timer/timer.noarch.hh"

#if defined(PSZ_USE_CUDA)

#include "timer/timer.cu.hh"

#elif defined(PSZ_USE_HIP)

#include "timer/timer.hip.hh"

#elif defined(PSZ_USE_1API)

#include "timer/timer.dp.hh"

#endif

#if defined(__FUTURE)

typedef struct Timer {
  hires_clock_t start, end;

  void timer_start() { start = hires::now(); }
  void timer_end() { end = hires::now(); }
  double get_time_elapsed()
  {
    return static_cast<duration_t>(end - start).count();
  }

} host_timer_t;

#ifdef __CUDACC__

/**
 * @brief CUDA event based timer. Synopsis:
 * cuda_timer_t t;
 * t.timer_start();
 * kernel<<<grid_dim, block_dim, nbytes, stream>>>(...);
 * t.timer_end();
 * cudaStreamSynchronize(stream);
 * auto ms = t.get_time_elapsed();
 *
 */
typedef struct CUDATimer {
  cudaEvent_t start, stop;
  float milliseconds;

  // stream not involved
  void timer_start()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
  }

  void timer_end()
  {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  }

  // stream involved
  void timer_start(cudaStream_t stream)
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);  // set event as not occurred
  }

  void timer_end(cudaStream_t stream)
  {
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);  // block host until `stream` meets `stop`
  }

  // get time
  float get_time_elapsed()
  {
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
  }

} cuda_timer_t;

#endif

// TODO handle return; testing
/**
 * @brief A timer wrapper for arbitrary function (no handling return for now);
 * Adapted from https://stackoverflow.com/a/33900479/8740097 (CC BY-SA 3.0)
 *
 * @tparam F auto function type
 * @tparam Args variadic function argument type
 * @param func non-return function to be timed
 * @param args variadic function arguments
 * @return double time in seconds
 */
template <typename F, typename... Args>
double TimeThisRoutine(F func, Args&&... args)
{
  auto t0 = hires::now();
  func(std::forward<Args>(args)...);
  return static_cast<duration_t>(hires::now() - t0).count();
}

#ifdef __CUDACC__
typedef struct CUDAKernelConfig {
  dim3 dim_grid;
  dim3 dim_block;
  size_t shmem_nbyte{0};
  cudaStream_t stream;

} kernelcfg;

// TODO use cudaEvent
/**
 * @brief A timer wrapper for arbitrary CUDA function
 *
 * @tparam F auto function type
 * @tparam Args variadic function argument type
 * @param func CUDA kernel function to be time
 * @param cfg CUDA kernel config
 * @param args variadic function arguments
 * @return double time in seconds
 */
template <typename F, typename... Args>
float TimeThisCUDARoutine(F func, kernelcfg cfg, Args&&... args)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  func<<<cfg.dim_grid, cfg.dim_block, cfg.shmem_nbyte, cfg.stream>>>(  //
      args...
      // std::forward<Args>(args)... // also works
  );
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaStreamSynchronize(cfg.stream);

  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);

  return milliseconds;
}

#endif
#endif

#endif  // UTILS_TIMER_HH
