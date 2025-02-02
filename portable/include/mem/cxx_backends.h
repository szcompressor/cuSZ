#ifndef _PORTABLE_MEM_CXX_BACKENDS_H
#define _PORTABLE_MEM_CXX_BACKENDS_H

#define ALIGN_128(len) (((len) + 127) & ~127)
#define ALIGN_256(len) (((len) + 255) & ~255)
#define ALIGN_512(len) (((len) + 511) & ~511)
#define ALIGN_1Ki(len) (((len) + 1023) & ~1023)
#define ALIGN_2Ki(len) (((len) + 2047) & ~2047)
#define ALIGN_4Ki(len) (((len) + 4095) & ~4095)
#define ALIGN_8Ki(len) (((len) + 8191) & ~8191)

#include <cstring>
#include <stdexcept>

#if defined(_PORTABLE_USE_CUDA)
#include <cuda_runtime.h>
#elif defined(_PORTABLE_USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(_PORTABLE_USE_1API)
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#endif

#include "c_type.h"
#include "cxx_mem_ops.h"
#include "cxx_smart_ptr.h"

#define MAKE_STDLEN3(X, Y, Z) \
  std::array<size_t, 3> { X, Y, Z }

#define XYZ_TO_DIM3(LEN3) dim3(X(LEN3), Y(LEN3), Z(LEN3))
#define STDLEN3_TO_DIM3(LEN3) dim3(LEN3[0], LEN3[1], LEN3[2])
#define STDLEN3_TO_STRIDE3(LEN3) dim3(1, LEN3[0], LEN3[0] * LEN3[1])

#if defined(_PORTABLE_USE_CUDA)

#define GPULEN3 dim3
#define MAKE_GPULEN3(X, Y, Z) dim3(X, Y, Z)
#define GPU_BACKEND_SPECIFIC_STREAM cudaStream_t
#define GPU_EVENT cudaEvent_t
#define GPU_EVENT_CREATE(e) cudaEventCreate(e);

#elif defined(_PORTABLE_USE_HIP)

#define GPULEN3 dim3
#define MAKE_GPULEN3(X, Y, Z) dim3(X, Y, Z)
#define GPU_BACKEND_SPECIFIC_STREAM hipStream_t
#define GPU_EVENT hipEvent_t
#define GPU_EVENT_CREATE(e) hipEventCreate(e);

#elif defined(_PORTABLE_USE_1API)

#define GPULEN3 sycl::range<3>
#define MAKE_GPULEN3(X, Y, Z) sycl::range<3>(Z, Y, X)
#define GPU_BACKEND_SPECIFIC_STREAM sycl::queue*
#define GPU_EVENT sycl::event*
#define GPU_EVENT_CREATE(e) new sycl::event();

#endif

#if defined(_PORTABLE_USE_CUDA)

#define event_create_pair(...)                   \
  ([]() -> std::pair<cudaEvent_t, cudaEvent_t> { \
    cudaEvent_t a, b;                            \
    cudaEventCreate(&a);                         \
    cudaEventCreate(&b);                         \
    return {a, b};                               \
  })(__VA_ARGS__);
#define event_destroy_pair(a, b) \
  cudaEventDestroy(a);           \
  cudaEventDestroy(b);
#define event_recording_start(E1, STREAM) cudaEventRecord(E1, (cudaStream_t)STREAM);
#define event_recording_stop(E2, STREAM)     \
  cudaEventRecord(E2, (cudaStream_t)STREAM); \
  cudaEventSynchronize(E2);
#define event_time_elapsed(start, end, p_millisec) cudaEventElapsedTime(p_millisec, start, end);

#elif defined(_PORTABLE_USE_HIP)

#define event_create_pair(...)                 \
  ([]() -> std::pair<hipEvent_t, hipEvent_t> { \
    hipEvent_t a, b;                           \
    hipEventCreate(&a);                        \
    hipEventCreate(&b);                        \
    return {a, b};                             \
  })(__VA_ARGS__);
#define event_destroy_pair(a, b) \
  hipEventDestroy(a);            \
  hipEventDestroy(b);
#define event_recording_start(E1, STREAM) hipEventRecord(E1, (hipStream_t)STREAM);
#define event_recording_stop(E2, STREAM)   \
  hipEventRecord(E2, (hipStream_t)STREAM); \
  hipEventSynchronize(E2);
#define event_time_elapsed(start, end, p_millisec) hipEventElapsedTime(p_millisec, start, end);

#elif defined(_PORTABLE_USE_1API)

#define event_create_pair(...)                     \
  ([]() -> std::pair<sycl::event*, sycl::event*> { \
    sycl::event *a, *b;                            \
    a = new sycl::event();                         \
    b = new sycl::event();                         \
    return {a, b};                                 \
  })(__VA_ARGS__);
#define event_destroy_pair(a, b) \
  dpct::destroy_event(a);        \
  dpct::destroy_event(b);
#define event_recording_start(E1, STREAM) \
  *E1 = ((sycl::queue*)STREAM)->ext_oneapi_submit_barrier();
#define event_recording_stop(E2, STREAM)                     \
  *E2 = ((sycl::queue*)STREAM)->ext_oneapi_submit_barrier(); \
  E2->wait_and_throw();
#define event_time_elapsed_TODO(start, end, p_millisec)                                     \
  *p_millisec = (end->get_profiling_info<sycl::info::event_profiling::command_end>() -      \
                 start->get_profiling_info<sycl::info::event_profiling::command_start>()) / \
                1.0e6;

#endif

#if defined(_PORTABLE_USE_CUDA)

#define create_stream(...)     \
  ([]() -> cudaStream_t {      \
    cudaStream_t stream;       \
    cudaStreamCreate(&stream); \
    return stream;             \
  })(__VA_ARGS__);
#define destroy_stream(stream) ([](void* s) { cudaStreamDestroy((cudaStream_t)s); })(stream);

#elif defined(_PORTABLE_USE_HIP)

#define create_stream(...)    \
  ([]() -> hipStream_t {      \
    hipStream_t stream;       \
    hipStreamCreate(&stream); \
    return stream;            \
  })(__VA_ARGS__);
#define destroy_stream(stream) ([](void* s) { hipStreamDestroy((hipStream_t)s); })(stream);

#elif defined(_PORTABLE_USE_1API)

#define create_stream(...)                                         \
  ([]() -> dpct::queue_ptr {                                       \
    dpct::queue_ptr q = dpct::get_current_device().create_queue(); \
    return q;                                                      \
  })(__VA_ARGS__);
#define destroy_stream(stream) ([](void* q) { ((dpct::queue_ptr)q)->reset(); })(stream);

#endif

#if defined(_PORTABLE_USE_CUDA)

#define sync_by_stream(stream) cudaStreamSynchronize((cudaStream_t)stream);
#define sync_device cudaDeviceSynchronize();

#elif defined(_PORTABLE_USE_HIP)

#define sync_by_stream(stream) hipStreamSynchronize((hipStream_t)stream);
#define sync_device hipDeviceSynchronize();

#elif defined(_PORTABLE_USE_1API)

#define sync_by_stream(stream) ((dpct::queue_ptr)stream)->wait();
// TODO there is no device wide sync?
#define sync_device

#endif

#endif /* _PORTABLE_MEM_CXX_BACKENDS_H */
