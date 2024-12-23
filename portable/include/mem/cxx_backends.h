#ifndef _PORTABLE_MEM_CXX_BACKENDS_H
#define _PORTABLE_MEM_CXX_BACKENDS_H

#include <cstring>
#include <stdexcept>

#if defined(PSZ_USE_CUDA)
#include <cuda_runtime.h>
#elif defined(PSZ_USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(PSZ_USE_1API)
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#endif

#include "c_type.h"
#include "cxx_smart_ptr.h"

#if defined(PSZ_USE_CUDA)

#define GPU_LEN3 dim3
#define MAKE_GPU_LEN3(X, Y, Z) dim3(X, Y, Z)

#elif defined(PSZ_USE_HIP)

#define GPU_LEN3 dim3
#define MAKE_GPU_LEN3(X, Y, Z) dim3(X, Y, Z)

#elif defined(PSZ_USE_1API)

#define GPU_LEN3 sycl::range<3>
#define MAKE_GPU_LEN3(X, Y, Z) sycl::range<3>(Z, Y, X)

#endif

template <_portable_mem_control>
struct _memcpy_direcion;

template <>
struct _memcpy_direcion<H2D> {
#if defined(PSZ_USE_CUDA)
  static const cudaMemcpyKind direction = cudaMemcpyHostToHost;
#elif defined(PSZ_USE_HIP)
  static const hipMemcpyKind direction = hipMemcpyHostToHost;
#elif defined(PSZ_USE_1API)
// no need
#endif
};

template <>
struct _memcpy_direcion<D2H> {
#if defined(PSZ_USE_CUDA)
  static const cudaMemcpyKind direction = cudaMemcpyDeviceToHost;
#elif defined(PSZ_USE_HIP)
  static const hipMemcpyKind direction = hipMemcpyDeviceToHost;
#elif defined(PSZ_USE_1API)
// no need
#endif
};

template <>
struct _memcpy_direcion<D2D> {
#if defined(PSZ_USE_CUDA)
  static const cudaMemcpyKind direction = cudaMemcpyDeviceToDevice;
#elif defined(PSZ_USE_HIP)
  static const hipMemcpyKind direction = hipMemcpyDeviceToDevice;
#elif defined(PSZ_USE_1API)
// no need
#endif
};

template <typename T>
T* malloc_device(size_t const len, void* stream = nullptr)
{
  T* __a;
#if defined(PSZ_USE_CUDA)
  cudaMalloc(&__a, len * sizeof(T));
  cudaMemset(__a, 0, len * sizeof(T));
#elif defined(PSZ_USE_HIP)
  hipMalloc(&__a, len * sizeof(T));
  hipMemset(__a, 0, len * sizeof(T));
#elif defined(PSZ_USE_1API)
  if (not stream)
    throw std::runtime_error(
        "[psz::error] SYCL backend does not allow stream to be null.");
  __a = sycl::malloc_device<T>(len, *((sycl::queue*)stream));
  ((sycl::queue*)stream)->wait();
#endif
  return __a;
}

template <typename T>
T* malloc_host(size_t const len, void* stream = nullptr)
{
  T* __a;
#if defined(PSZ_USE_CUDA)
  cudaMallocHost(&__a, len * sizeof(T));
  memset(__a, 0, len * sizeof(T));
#elif defined(PSZ_USE_HIP)
  hipHostMalloc(&__a, len * sizeof(T));
  memset(__a, 0, len * sizeof(T));
#elif defined(PSZ_USE_1API)
  if (not stream)
    throw std::runtime_error(
        "[psz::error] SYCL backend does not allow stream to be null.");
  __a = sycl::malloc_host<T>(len, *((sycl::queue*)stream));
  ((sycl::queue*)stream)->wait();
#endif
  return __a;
}

#define malloc_shared malloc_unified

template <typename T>
T* malloc_unified(size_t const len, void* stream = nullptr)
{
  T* __a;
#if defined(PSZ_USE_CUDA)
  cudaMallocManaged(&__a, len * sizeof(T));
#elif defined(PSZ_USE_HIP)
  hipMallocManaged(&__a, len * sizeof(T));
#elif defined(PSZ_USE_1API)
  if (not stream)
    throw std::runtime_error(
        "[psz::error] SYCL backend does not allow stream to be null.");
  __a = sycl::malloc_unified<T>(len, *((sycl::queue*)stream));
  ((sycl::queue*)stream)->wait();
#endif
  return __a;
}

template <typename T>
void free_device(T* __a, void* stream = nullptr)
{
#if defined(PSZ_USE_CUDA)
  cudaFree(__a);
#elif defined(PSZ_USE_HIP)
  hipFree(__a);
#elif defined(PSZ_USE_1API)
  if (not stream)
    throw std::runtime_error(
        "[psz::error] SYCL backend does not allow stream to be null.");
  sycl::free(__a, *((sycl::queue*)stream));
#endif
}

template <typename T>
void free_host(T* __a, void* stream = nullptr)
{
#if defined(PSZ_USE_CUDA)
  cudaFreeHost(__a);
#elif defined(PSZ_USE_HIP)
  hipHostFree(__a);
#elif defined(PSZ_USE_1API)
  if (not stream)
    throw std::runtime_error(
        "[psz::error] SYCL backend does not allow stream to be null.");
  sycl::free(__a, *((sycl::queue*)stream));
#endif
}

#define free_shared free_unified

template <typename T>
void free_unified(T* __a, void* stream = nullptr)
{
  free_device(__a, stream);
}

template <typename T>
std::unique_ptr<T, GPU_deleter_device> make_unique_device(
    size_t const len, void* stream = nullptr)
{
  T* ptr = malloc_device<T>(len);
  return std::unique_ptr<T, GPU_deleter_device>(
      ptr, GPU_deleter_device(stream));
}

template <typename T>
std::unique_ptr<T, GPU_deleter_host> make_unique_host(
    size_t const len, void* stream = nullptr)
{
  T* ptr = malloc_host<T>(len);
  return std::unique_ptr<T, GPU_deleter_host>(ptr, GPU_deleter_host(stream));
}

template <typename T>
std::unique_ptr<T, GPU_deleter_unified> make_unique_unified(
    size_t const len, void* stream = nullptr)
{
  T* ptr = malloc_unified<T>(len);
  return std::unique_ptr<T, GPU_deleter_unified>(
      ptr, GPU_deleter_unified(stream));
}

template <typename T>
std::shared_ptr<T> make_shared_device(size_t const len, void* stream = nullptr)
{
  T* ptr = malloc_device<T>(len);
  return std::shared_ptr<T>(ptr, GPU_deleter_device(stream));
}

template <typename T>
std::shared_ptr<T> make_shared_host(size_t const len, void* stream = nullptr)
{
  T* ptr = malloc_host<T>(len);
  return std::shared_ptr<T>(ptr, GPU_deleter_host(stream));
}

template <typename T>
std::shared_ptr<T> make_shared_unified(
    size_t const len, void* stream = nullptr)
{
  T* ptr = malloc_unified<T>(len);
  return std::shared_ptr<T>(ptr, GPU_deleter_unified(stream));
}

template <_portable_mem_control DIR, typename T>
void memcpy_allkinds(T* dst, T* src, size_t const len, void* stream = nullptr)
{
  static_assert(
      std::is_trivially_copyable_v<T>, "T must be a trivially copyable type.");

  constexpr auto direction = _memcpy_direcion<DIR>::direction;

#if defined(PSZ_USE_CUDA)
  cudaMemcpy(dst, src, sizeof(T) * len, direction);
#elif defined(PSZ_USE_HIP)
  hipMemcpy(dst, src, sizeof(T) * len, direction);
#elif defined(PSZ_USE_1API)
  if (not stream) {
    cerr << "[psz::warning] null queue is not allowed; "
            "fall back to default queue."
         << endl;
    dpct::device_ext& dev = dpct::get_current_device();
    sycl::queue& q = dev.default_queue();
    q.memcpy(dst, src, sizeof(T) * len).wait();
  }
#endif
}

template <_portable_mem_control DIR, typename T>
void memcpy_allkinds_async(
    T* dst, T* src, size_t const len, void* stream = nullptr)
{
  static_assert(
      std::is_trivially_copyable_v<T>, "T must be a trivially copyable type.");

  constexpr auto direction = _memcpy_direcion<DIR>::direction;

#if defined(PSZ_USE_CUDA)
  cudaMemcpyAsync(dst, src, sizeof(T) * len, direction, (cudaStream_t)stream);
#elif defined(PSZ_USE_HIP)
  hipMemcpyAsync(dst, src, sizeof(T) * len, direction, (hipStream_t)stream);
#elif defined(PSZ_USE_1API)
  ((sycl::queue*)stream)->memcpy(dst, src, sizeof(T) * len);
#endif
}

template <typename T>
void memset_device(T* __a, size_t const len, int value = 0)
{
  static_assert(
      std::is_trivially_copyable_v<T>, "T must be a trivially copyable type.");

#if defined(PSZ_USE_CUDA)
  cudaMemset(__a, value, sizeof(T) * len);
#elif defined(PSZ_USE_HIP)
  hipMemset(__a, value, sizeof(T) * len);
#elif defined(PSZ_USE_1API)
  memset(__a, value, sizeof(T) * len);
#endif
}

template <typename T>
void memset_host(T* __a, size_t const len, int value = 0)
{
  static_assert(
      std::is_trivially_copyable_v<T>, "T must be a trivially copyable type.");

  memset(__a, value, sizeof(T) * len);
}

#if defined(PSZ_USE_CUDA)
#define create_stream(...)     \
  ([]() -> cudaStream_t {      \
    cudaStream_t stream;       \
    cudaStreamCreate(&stream); \
    return stream;             \
  })(__VA_ARGS__);
#elif defined(PSZ_USE_HIP)
#define create_stream(...)    \
  ([]() -> hipStream_t {      \
    hipStream_t stream;       \
    hipStreamCreate(&stream); \
    return stream;            \
  })(__VA_ARGS__);
#elif defined(PSZ_USE_1API)
#define create_stream(...)                                         \
  ([]() -> dpct::queue_ptr {                                       \
    dpct::queue_ptr q = dpct::get_current_device().create_queue(); \
    return q;                                                      \
  })(__VA_ARGS__);
#endif

#if defined(PSZ_USE_CUDA)
#define destroy_stream(stream) \
  ([](void* s) { cudaStreamDestroy((cudaStream_t)s); })(stream);
#elif defined(PSZ_USE_HIP)
#define destroy_stream(stream) \
  ([](void* s) { hipStreamDestroy((cudaStream_t)s); })(stream);
#elif defined(PSZ_USE_1API)
#define destroy_stream(stream) \
  ([](void* q) { ((dpct::queue_ptr)q)->reset(); })(stream);
#endif

#if defined(PSZ_USE_CUDA)
#define sync_by_stream(stream) cudaStreamSynchronize((cudaStream_t)stream);
#elif defined(PSZ_USE_HIP)
#define sync_by_stream(stream) hipStreamSynchronize((hipStream_t)stream);
#elif defined(PSZ_USE_1API)
#define sync_by_stream(stream) ((dpct::queue_ptr)stream)->wait();
#endif

#if defined(PSZ_USE_CUDA)
#define sync_device cudaDeviceSynchronize();
#elif defined(PSZ_USE_HIP)
#define sync_device hipDeviceSynchronize();
#elif defined(PSZ_USE_1API)
#define sync_device
// TODO there is no device wide sync?
#endif

#endif /* _PORTABLE_MEM_CXX_BACKENDS_H */
