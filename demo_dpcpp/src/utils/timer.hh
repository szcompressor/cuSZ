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

#endif  // UTILS_TIMER_HH
