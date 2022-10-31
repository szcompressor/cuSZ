/**
 * @file timer_gpu.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-31
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>
#include "utils/timer.h"

// cuda timer specific

asz_timer* asz_cudatimer_create()
{
    auto t = new asz_timer{
        .policy = asz_policy::CUDA,  //
        .start  = new cudaEvent_t,
        .end    = new cudaEvent_t};

    cudaEventCreate(static_cast<cudaEvent_t*>(t->start));
    cudaEventCreate(static_cast<cudaEvent_t*>(t->end));

    return t;
}

void asz_cudatimer_destroy(asz_timer* t)
{
    delete static_cast<cudaEvent_t*>(t->start);
    delete static_cast<cudaEvent_t*>(t->end);
    delete t;
}

void asz_cudatimer_start(asz_timer* t)
{  //
    cudaEventRecord(*static_cast<cudaEvent_t*>(t->start));
}

void asz_cudatimer_end(asz_timer* t)
{  //
    cudaEventRecord(*static_cast<cudaEvent_t*>(t->end));
    cudaEventSynchronize(*static_cast<cudaEvent_t*>(t->end));
}

double asz_cudatime_elapsed(asz_timer* t)
{
    float second;
    cudaEventElapsedTime(&second, *static_cast<cudaEvent_t*>(t->start), *static_cast<cudaEvent_t*>(t->end));
    return second / 1000;
}

// cuda streamtimer specific

asz_timer* asz_cudastreamtimer_create(void* stream)
{
    auto t = new asz_timer{
        .policy = asz_policy::CUDA,  //
        .start  = new cudaEvent_t,
        .end    = new cudaEvent_t,
        .stream = stream};

    cudaEventCreate(static_cast<cudaEvent_t*>(t->start));
    cudaEventCreate(static_cast<cudaEvent_t*>(t->end));

    return t;
}

void asz_cudastreamtimer_destroy(asz_timer* t)
{
    delete static_cast<cudaEvent_t*>(t->start);
    delete static_cast<cudaEvent_t*>(t->end);
    delete t;
}

void asz_cudastreamtimer_start(asz_timer* t)
{  //
    cudaEventRecord(*static_cast<cudaEvent_t*>(t->start), static_cast<cudaStream_t>(t->stream));
}

void asz_cudastreamtimer_end(asz_timer* t)
{  //
    cudaEventRecord(*static_cast<cudaEvent_t*>(t->end), static_cast<cudaStream_t>(t->stream));
    cudaEventSynchronize(*static_cast<cudaEvent_t*>(t->end));
}

double asz_cudastreamtime_elapsed(asz_timer* t)
{
    float second;
    cudaEventElapsedTime(&second, *static_cast<cudaEvent_t*>(t->start), *static_cast<cudaEvent_t*>(t->end));
    return second / 1000;
}
