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

struct asz_timer {
    asz_policy policy;

    void* start;
    void* end;

    void* stream;
};

typedef struct asz_timer asz_timer;

// top-level/dispatcher
asz_timer* asz_timer_create(asz_policy const p, void* stream);
void       asz_timer_destroy(asz_timer* t);
void       asz_timer_start(asz_timer* t);
void       asz_timer_end(asz_timer* t);
double     asz_time_elapsed(asz_timer* t);

asz_timer* asz_cputimer_create();
void       asz_cputimer_destroy(asz_timer* t);
void       asz_cputimer_start(asz_timer* t);
void       asz_cputimer_end(asz_timer* t);
double     asz_cputime_elapsed(asz_timer* t);

asz_timer* asz_cudatimer_create();
void       asz_cudatimer_destroy(asz_timer* t);
void       asz_cudatimer_start(asz_timer* t);
void       asz_cudatimer_end(asz_timer* t);
double     asz_cudatime_elapsed(asz_timer* t);

asz_timer* asz_cudastreamtimer_create(void* stream);
void       asz_cudastreamtimer_destroy(asz_timer* t);
void       asz_cudastreamtimer_start(asz_timer* t);
void       asz_cudastreamtimer_end(asz_timer* t);
double     asz_cudastreamtime_elapsed(asz_timer* t);

#ifdef __cplusplus
}
#endif

#endif /* B36B7228_E9EC_4E61_A1DC_19A4352C4EB3 */
