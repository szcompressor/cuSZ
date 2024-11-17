#ifndef UTILS_TIMER_HH
#define UTILS_TIMER_HH

// Jiannan Tian
// (created) 2019-08-26
// (update) 2021-01-05, 2021-12-23, 2022-10-31, 2024-12-22

#include <chrono>
#include <utility>

using hires = std::chrono::high_resolution_clock;
using duration_t = std::chrono::duration<double>;
using hires_clock_t = std::chrono::time_point<hires>;

#define CREATE_CPU_TIMER                                    \
  std::chrono::time_point<std::chrono::steady_clock> a_ct1; \
  std::chrono::time_point<std::chrono::steady_clock> b_ct1;
#define START_CPU_TIMER a_ct1 = std::chrono::steady_clock::now();
#define STOP_CPU_TIMER b_ct1 = std::chrono::steady_clock::now();
#define TIME_ELAPSED_CPU_TIMER(PTR_MILLISEC) \
  ms = std::chrono::duration<float, std::milli>(b_ct1 - a_ct1).count();

#if defined(PSZ_USE_CUDA)

#define CREATE_GPUEVENT_PAIR \
  cudaEvent_t a, b;          \
  cudaEventCreate(&a);       \
  cudaEventCreate(&b);
#define DESTROY_GPUEVENT_PAIR \
  cudaEventDestroy(a);        \
  cudaEventDestroy(b);
#define START_GPUEVENT_RECORDING(STREAM) \
  cudaEventRecord(a, (cudaStream_t)STREAM);
#define STOP_GPUEVENT_RECORDING(STREAM)     \
  cudaEventRecord(b, (cudaStream_t)STREAM); \
  cudaEventSynchronize(b);
#define TIME_ELAPSED_GPUEVENT(PTR_MILLISEC) \
  cudaEventElapsedTime(PTR_MILLISEC, a, b);

#elif defined(PSZ_USE_HIP)

#define CREATE_GPUEVENT_PAIR \
  hipEvent_t a, b;           \
  hipEventCreate(&a);        \
  hipEventCreate(&b);
#define DESTROY_GPUEVENT_PAIR \
  hipEventDestroy(a);         \
  hipEventDestroy(b);
#define START_GPUEVENT_RECORDING(STREAM) \
  hipEventRecord(a, (hipStream_t)STREAM);
#define STOP_GPUEVENT_RECORDING(STREAM)   \
  hipEventRecord(b, (hipStream_t)STREAM); \
  hipEventSynchronize(b);
#define TIME_ELAPSED_GPUEVENT(PTR_MILLISEC) \
  hipEventElapsedTime(PTR_MILLISEC, a, b);

#elif defined(PSZ_USE_1API)

#define SYCL_TIME_DELTA(EVENT, MILLISEC)                                    \
  auto sycl_time_delta = [](sycl::event& e) {                               \
    cl_ulong start_time =                                                   \
        e.get_profiling_info<sycl::info::event_profiling::command_start>(); \
    cl_ulong end_time =                                                     \
        e.get_profiling_info<sycl::info::event_profiling::command_end>();   \
    return ((float)(end_time - start_time)) / 1e6;                          \
  };                                                                        \
  MILLISEC = sycl_time_delta(EVENT);

#define CREATE_GPUEVENT_PAIR      \
  auto start = new sycl::event(); \
  auto end = new sycl::event();

#define DESTROY_GPUEVENT_PAIR \
  dpct::destroy_event(start); \
  dpct::destroy_event(end);

#define START_GPUEVENT_RECORDING(STREAM) \
  *start = ((sycl::queue*)STREAM)->ext_oneapi_submit_barrier();

#define STOP_GPUEVENT_RECORDING(STREAM)                       \
  *end = ((sycl::queue*)STREAM)->ext_oneapi_submit_barrier(); \
  end->wait_and_throw();

#define TIME_ELAPSED_GPUEVENT(PTR_MILLISEC)                                  \
  *PTR_MILLISEC =                                                            \
      (end->get_profiling_info<sycl::info::event_profiling::command_end>() - \
       start->get_profiling_info<                                            \
           sycl::info::event_profiling::command_start>()) /                  \
      1e3f;

#endif

#endif /* UTILS_TIMER_HH */
