#ifndef UTILS_TIMER_HH
#define UTILS_TIMER_HH

/**
 * @file timer.hh
 * @author Jiannan Tian
 * @brief High-resolution timer wrapper from <chrono>
 * @version 0.2
 * @date 2021-01-05
 * Created on 2019-08-26
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <chrono>
#include <utility>

using hires         = std::chrono::high_resolution_clock;
using duration_t    = std::chrono::duration<double>;
using hires_clock_t = std::chrono::time_point<hires>;

#endif  // UTILS_TIMER_HH
