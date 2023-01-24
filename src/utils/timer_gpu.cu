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
#include <iostream>
#include "utils/timer.h"

typedef struct asz_cudatimer {
    cudaEvent_t  a, b;
    float        milliseconds;
    cudaStream_t stream;

    asz_cudatimer() { create(); }
    asz_cudatimer(cudaStream_t stream)
    {
        create();
        this->stream = stream;
    }

    void create()
    {
        cudaEventCreate(&a);
        cudaEventCreate(&b);
    }

    void destroy()
    {
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }

    // stream not involved
    void start() { cudaEventRecord(a); }

    void stop()
    {
        cudaEventRecord(b);
        cudaEventSynchronize(b);
    }

    // stream involved
    void stream_start()
    {
        cudaEventRecord(a, stream);  // set event as not occurred
    }

    void stream_stop()
    {
        cudaEventRecord(b, stream);
        cudaEventSynchronize(b);  // block host until `stream` meets `stop`
    }

    // get time
    float time_elapsed()
    {
        cudaEventElapsedTime(&milliseconds, a, b);
        std::cout << "milliseconds: " << milliseconds << std::endl;
        return milliseconds;
    }
} asz_cudatimer;

// cuda timer specific
asz_cudatimer* asz_cudatimer_create() { return new asz_cudatimer{}; }
void           asz_cudatimer_destroy(asz_cudatimer* t) { t->destroy(); }
void           asz_cudatimer_start(asz_cudatimer* t) { t->start(); }
void           asz_cudatimer_end(asz_cudatimer* t) { t->stop(); }
double         asz_cudatime_elapsed(asz_cudatimer* t) { return t->time_elapsed() / 1000; }

// cuda streamtimer specific
asz_cudatimer* asz_cudastreamtimer_create(void* stream) { return new asz_cudatimer((cudaStream_t)stream); }
void           asz_cudastreamtimer_destroy(asz_cudatimer* t) { t->destroy(); }
void           asz_cudastreamtimer_start(asz_cudatimer* t) { t->stream_start(); }
void           asz_cudastreamtimer_end(asz_cudatimer* t) { t->stream_stop(); }
double         asz_cudastreamtime_elapsed(asz_cudatimer* t) { return t->time_elapsed() / 1000; }
