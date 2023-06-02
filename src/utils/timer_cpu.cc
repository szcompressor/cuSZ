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

struct psz_timer {
  hires_clock_t start, stop;
};

// cpu timer specific
psz_timer* psz_cputimer_create() { return new psz_timer; }
void       psz_cputimer_destroy(psz_timer* t) { delete t; }
void       psz_cputimer_start(psz_timer* t) { t->start = hires::now(); }
void       psz_cputimer_end(psz_timer* t) { t->stop = hires::now(); }
double     psz_cputime_elapsed(psz_timer* t)
{
  return static_cast<duration_t>((t->stop) - (t->start)).count();
}
