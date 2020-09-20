

#ifndef TIMER_HH
#define TIMER_HH

/**
 * @file timer.hh
 * @author Jiannan Tian
 * @brief High-resolution timer wrapper from <chrono>
 * @version 0.1
 * @date 2020-09-20
 * Created on 2019-08-26
 *
 * @copyright Copyright (c) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <chrono>
#include <iostream>

using std::cerr;
using std::cout;
using std::endl;

using hires = std::chrono::high_resolution_clock;
typedef std::chrono::duration<double>                               duration_t;
typedef std::chrono::time_point<std::chrono::high_resolution_clock> hires_clock_t;

#endif  // TIMER_HH
