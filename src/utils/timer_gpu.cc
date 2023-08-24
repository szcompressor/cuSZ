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
#include "utils/timer.hh"

typedef struct psz_cudatimer {
  cudaEvent_t  a, b;
  float        milliseconds;
  cudaStream_t stream;

  psz_cudatimer() { create(); }
  psz_cudatimer(cudaStream_t stream)
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
} psz_cudatimer;

// cuda timer specific
psz_cudatimer* psz_cudatimer_create() { return new psz_cudatimer{}; }
void           psz_cudatimer_destroy(psz_cudatimer* t) { t->destroy(); }
void           psz_cudatimer_start(psz_cudatimer* t) { t->start(); }
void           psz_cudatimer_end(psz_cudatimer* t) { t->stop(); }
double         psz_cudatime_elapsed(psz_cudatimer* t)
{
  return t->time_elapsed() / 1000;
}

// cuda streamtimer specific
psz_cudatimer* psz_cudastreamtimer_create(void* stream)
{
  return new psz_cudatimer((cudaStream_t)stream);
}
void   psz_cudastreamtimer_destroy(psz_cudatimer* t) { t->destroy(); }
void   psz_cudastreamtimer_start(psz_cudatimer* t) { t->stream_start(); }
void   psz_cudastreamtimer_end(psz_cudatimer* t) { t->stream_stop(); }
double psz_cudastreamtime_elapsed(psz_cudatimer* t)
{
  return t->time_elapsed() / 1000;
}
