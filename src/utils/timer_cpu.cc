/**
 * @file timer_cpu.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-31
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "utils/timer.h"

#include <chrono>
#include <utility>

using hires         = std::chrono::high_resolution_clock;
using duration_t    = std::chrono::duration<double>;
using hires_clock_t = std::chrono::time_point<hires>;

struct asz_timer {
    hires_clock_t start, stop;
};

// cpu timer specific
asz_timer* asz_cputimer_create() { return new asz_timer; }
void       asz_cputimer_destroy(asz_timer* t) { delete t; }
void       asz_cputimer_start(asz_timer* t) { t->start = hires::now(); }
void       asz_cputimer_end(asz_timer* t) { t->stop = hires::now(); }
double     asz_cputime_elapsed(asz_timer* t) { return static_cast<duration_t>((t->stop) - (t->start)).count(); }
