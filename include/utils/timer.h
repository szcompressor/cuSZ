/**
 * @file timer.h
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-10-31
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef B36B7228_E9EC_4E61_A1DC_19A4352C4EB3
#define B36B7228_E9EC_4E61_A1DC_19A4352C4EB3

#ifdef __cplusplus
extern "C" {
#endif

#include "../cusz/type.h"

struct asz_timer;
typedef struct asz_timer asz_timer;
typedef struct asz_timer asz_cputimer;

struct asz_cudatimer;
typedef struct asz_cudatimer asz_cudatimer;

// top-level/dispatcher
// asz_timer* asz_timer_create(asz_policy const p, void* stream);
// void       asz_timer_destroy(asz_timer* t);
// void       asz_timer_start(asz_timer* t);
// void       asz_timer_end(asz_timer* t);
// double     asz_time_elapsed(asz_timer* t);

asz_timer* asz_cputimer_create();
void       asz_cputimer_destroy(asz_timer* t);
void       asz_cputimer_start(asz_timer* t);
void       asz_cputimer_end(asz_timer* t);
double     asz_cputime_elapsed(asz_timer* t);

// 22-11-01 adding wrapper incurs unexpeted overhead in timing
asz_cudatimer* asz_cudatimer_create();
void           asz_cudatimer_destroy(asz_cudatimer* t);
void           asz_cudatimer_start(asz_cudatimer* t);
void           asz_cudatimer_end(asz_cudatimer* t);
double         asz_cudatime_elapsed(asz_cudatimer* t);

asz_cudatimer* asz_cudastreamtimer_create(void* stream);
void           asz_cudastreamtimer_destroy(asz_cudatimer* t);
void           asz_cudastreamtimer_start(asz_cudatimer* t);
void           asz_cudastreamtimer_end(asz_cudatimer* t);
double         asz_cudastreamtime_elapsed(asz_cudatimer* t);

// 22-11-01 CUDA timing snippet instead
#define CREATE_CUDAEVENT_PAIR \
    cudaEvent_t a, b;         \
    cudaEventCreate(&a);      \
    cudaEventCreate(&b);

#define DESTROY_CUDAEVENT_PAIR \
    cudaEventDestroy(a);       \
    cudaEventDestroy(b);

#define START_CUDAEVENT_RECORDING(STREAM) cudaEventRecord(a, STREAM);
#define STOP_CUDAEVENT_RECORDING(STREAM) \
    cudaEventRecord(b, STREAM);          \
    cudaEventSynchronize(b);

#define TIME_ELAPSED_CUDAEVENT(PTR_MILLISEC) cudaEventElapsedTime(PTR_MILLISEC, a, b);

// 22-11-01 HIP timing snippet instead
#define CREATE_HIPEVENT_PAIR \
    hipEvent_t a, b;         \
    hipEventCreate(&a);      \
    hipEventCreate(&b);

#define DESTROY_HIPEVENT_PAIR \
    hipEventDestroy(a);       \
    hipEventDestroy(b);

#define START_HIPEVENT_RECORDING(STREAM) hipEventRecord(a, STREAM);
#define STOP_HIPEVENT_RECORDING(STREAM) \
    hipEventRecord(b, STREAM);          \
    hipEventSynchronize(b);

#define TIME_ELAPSED_HIPEVENT(PTR_MILLISEC) hipEventElapsedTime(PTR_MILLISEC, a, b);

#ifdef __cplusplus
}
#endif

#endif /* B36B7228_E9EC_4E61_A1DC_19A4352C4EB3 */
