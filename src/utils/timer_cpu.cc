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

// dipatcher

asz_timer* asz_timer_create(asz_policy const p, void* stream)
{
    asz_timer* t = nullptr;
    if (p == asz_policy::CPU) {  //
        t         = asz_cputimer_create();
        t->policy = p;
    }
    else if (p == asz_policy::CUDA) {
        if (stream)
            t = asz_cudastreamtimer_create(stream);
        else
            t = asz_cudatimer_create();

        t->policy = p;
    }
    else
        return nullptr;

    return t;
}

void asz_timer_destroy(asz_timer* t)
{
    if (t->policy == asz_policy::CPU)
        asz_cputimer_destroy(t);
    else if (t->policy == asz_policy::CUDA) {
        if (t->stream)
            asz_cudastreamtimer_destroy(t);
        else
            asz_cudatimer_destroy(t);
    }
}

void asz_timer_start(asz_timer* t)
{
    if (t->policy == asz_policy::CPU)
        asz_cputimer_start(t);
    else if (t->policy == asz_policy::CUDA) {
        if (t->stream)
            asz_cudastreamtimer_start(t);
        else
            asz_cudatimer_start(t);
    }
}

void asz_timer_end(asz_timer* t)
{
    if (t->policy == asz_policy::CPU)
        asz_cputimer_end(t);
    else if (t->policy == asz_policy::CUDA) {
        if (t->stream)
            asz_cudastreamtimer_end(t);
        else
            asz_cudatimer_end(t);
    }
}

double asz_time_elapsed(asz_timer* t)
{
    if (t->policy == asz_policy::CPU)
        return asz_cputime_elapsed(t);
    else if (t->policy == asz_policy::CUDA) {
        if (t->stream)
            return asz_cudastreamtime_elapsed(t);
        else
            return asz_cudatime_elapsed(t);
    }
    return 0;
}

// cpu timer specific

asz_timer* asz_cputimer_create()
{
    return new asz_timer{
        .policy = asz_policy::CPU,  //
        .start  = new hires_clock_t,
        .end    = new hires_clock_t};
}

void asz_cputimer_destroy(asz_timer* t)
{
    delete static_cast<hires_clock_t*>(t->start);
    delete static_cast<hires_clock_t*>(t->end);
    delete t;
}

void asz_cputimer_start(asz_timer* t)
{  //
    *static_cast<hires_clock_t*>(t->start) = hires::now();
}

void asz_cputimer_end(asz_timer* t)
{  //
    *static_cast<hires_clock_t*>(t->end) = hires::now();
}

double asz_cputime_elapsed(asz_timer* t)
{
    return static_cast<duration_t>(
               *static_cast<hires_clock_t*>(t->end) -  //
               *static_cast<hires_clock_t*>(t->start))
        .count();
}
