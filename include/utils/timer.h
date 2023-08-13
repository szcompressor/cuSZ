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

struct psztime;
typedef struct psztime psztime;
typedef struct psztime psz_cputimer;

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
