#ifndef _PORTABLE_MEM_CXX_BACKENDS_H
#define _PORTABLE_MEM_CXX_BACKENDS_H

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

#define MAKE_STD_LEN3(X, Y, Z) \
  std::array<size_t, 3> { X, Y, Z }

#if defined(_PORTABLE_USE_CUDA)

#define GPU_LEN3 dim3
#define MAKE_GPU_LEN3(X, Y, Z) dim3(X, Y, Z)

#elif defined(_PORTABLE_USE_HIP)

#define GPU_LEN3 dim3
#define MAKE_GPU_LEN3(X, Y, Z) dim3(X, Y, Z)

#elif defined(_PORTABLE_USE_1API)

#define GPU_LEN3 sycl::range<3>
#define MAKE_GPU_LEN3(X, Y, Z) sycl::range<3>(Z, Y, X)

#endif

#if defined(_PORTABLE_USE_CUDA)
#define create_stream(...)     \
  ([]() -> cudaStream_t {      \
    cudaStream_t stream;       \
    cudaStreamCreate(&stream); \
    return stream;             \
  })(__VA_ARGS__);
#elif defined(_PORTABLE_USE_HIP)
#define create_stream(...)    \
  ([]() -> hipStream_t {      \
    hipStream_t stream;       \
    hipStreamCreate(&stream); \
    return stream;            \
  })(__VA_ARGS__);
#elif defined(_PORTABLE_USE_1API)
#define create_stream(...)                                         \
  ([]() -> dpct::queue_ptr {                                       \
    dpct::queue_ptr q = dpct::get_current_device().create_queue(); \
    return q;                                                      \
  })(__VA_ARGS__);
#endif

#if defined(_PORTABLE_USE_CUDA)
#define destroy_stream(stream) ([](void* s) { cudaStreamDestroy((cudaStream_t)s); })(stream);
#elif defined(_PORTABLE_USE_HIP)
#define destroy_stream(stream) ([](void* s) { hipStreamDestroy((cudaStream_t)s); })(stream);
#elif defined(_PORTABLE_USE_1API)
#define destroy_stream(stream) ([](void* q) { ((dpct::queue_ptr)q)->reset(); })(stream);
#endif

#if defined(_PORTABLE_USE_CUDA)
#define sync_by_stream(stream) cudaStreamSynchronize((cudaStream_t)stream);
#elif defined(_PORTABLE_USE_HIP)
#define sync_by_stream(stream) hipStreamSynchronize((hipStream_t)stream);
#elif defined(_PORTABLE_USE_1API)
#define sync_by_stream(stream) ((dpct::queue_ptr)stream)->wait();
#endif

#if defined(_PORTABLE_USE_CUDA)
#define sync_device cudaDeviceSynchronize();
#elif defined(_PORTABLE_USE_HIP)
#define sync_device hipDeviceSynchronize();
#elif defined(_PORTABLE_USE_1API)
#define sync_device
// TODO there is no device wide sync?
#endif

#endif /* _PORTABLE_MEM_CXX_BACKENDS_H */
