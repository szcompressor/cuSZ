/**
 * @file timer.hh
 * @author Jiannan Tian
 * @brief High-resolution timer wrapper from <chrono> and util functions for timing both CPU and CUDA function
 * @version 0.2
 * @date 2021-01-05
 * Created on 2019-08-26
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#ifndef UTILS_TIMER_HH
#define UTILS_TIMER_HH

#include <chrono>
#include <utility>

using hires         = std::chrono::high_resolution_clock;
using duration_t    = std::chrono::duration<double>;
using hires_clock_t = std::chrono::time_point<hires>;

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
    float       milliseconds;

    void timer_start()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    float timer_end_get_elapsed_time()
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }

    void timer_end()
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }

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
double TimeThisFunction(F func, Args&&... args)
{
    auto t0 = hires::now();
    func(std::forward<Args>(args)...);
    return static_cast<duration_t>(hires::now() - t0).count();
}

#ifdef __CUDACC__
typedef struct CUDAKernelConfig {
    dim3         dim_grid;
    dim3         dim_block;
    size_t       num_shmem_bytes{0};
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
double TimeThisCUDAFunction(F func, kernelcfg cfg, Args&&... args)
{
    auto t0 = hires::now();
    func<<<cfg.dim_grid, cfg.dim_block, cfg.num_shmem_bytes, cfg.stream>>>(  //
        args...
        // std::forward<Args>(args)... // also works
    );
    cudaDeviceSynchronize();
    return static_cast<duration_t>(hires::now() - t0).count();
}

#endif

#endif  // UTILS_TIMER_HH
