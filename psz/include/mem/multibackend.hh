#ifndef F8DE640C_EFD2_444C_992C_946B18F625F2
#define F8DE640C_EFD2_444C_992C_946B18F625F2

#include <cstring>
#include <stdexcept>

#include "mem/definition.hh"

#if defined(PSZ_USE_CUDA)

#include <cuda_runtime.h>
#define GPU_LEN3 dim3
#define MAKE_GPU_LEN3(X, Y, Z) dim3(X, Y, Z)

#elif defined(PSZ_USE_HIP)

#include <hip/hip_runtime.h>
#define GPU_LEN3 dim3
#define MAKE_GPU_LEN3(X, Y, Z) dim3(X, Y, Z)

#elif defined(PSZ_USE_1API)

#include <sycl/sycl.hpp>
#define GPU_LEN3 sycl::range<3>
#define MAKE_GPU_LEN3(X, Y, Z) sycl::range<3>(Z, Y, X)

#endif

template <pszmem_control>
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

template <typename T>
T* malloc_shared(size_t const len, void* stream = nullptr)
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
  __a = sycl::malloc_shared<T>(len, *((sycl::queue*)stream));
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

template <typename T>
void free_shared(T* __a, void* stream = nullptr)
{
  free_device(__a, stream);
}

template <typename T, pszmem_control DIR>
void memcpy_allkinds(T* dst, T* src, size_t const len, void* stream = nullptr)
{
#if defined(PSZ_USE_CUDA)
  cudaMemcpy(dst, src, sizeof(T) * len, _memcpy_direcion<DIR>::direction);
#elif defined(PSZ_USE_HIP)
  hipMemcpy(dst, src, sizeof(T) * len, _memcpy_direcion<DIR>::direction);
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

template <typename T, pszmem_control DIR>
void memcpy_allkinds_async(
    T* dst, T* src, size_t const len, void* stream = nullptr)
{
  if (not stream)
    throw std::runtime_error(
        "[psz::error] null stream/queue is not allowed because async-form "
        "memcpy is specified.");

#if defined(PSZ_USE_CUDA)
  cudaMemcpyAsync(
      dst, src, sizeof(T) * len, _memcpy_direcion<DIR>::direction,
      (cudaStream_t)stream);
#elif defined(PSZ_USE_HIP)
  hipMemcpyAsync(
      dst, src, sizeof(T) * len, _memcpy_direcion<DIR>::direction,
      (hipStream_t)stream);

#elif defined(PSZ_USE_1API)
  ((sycl::queue*)stream)->memcpy(dst, src, sizeof(T) * len).wait();
#endif
}

template <typename T>
void memset_device(T* __a, size_t const len, int value = 0)
{
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
  memset(__a, value, sizeof(T) * len);
}

#if defined(PSZ_USE_CUDA)
#define sync_by_stream(stream) cudaStreamSynchronize((cudaStream_t)stream);
#elif defined(PSZ_USE_HIP)
#define sync_by_stream(stream) hipStreamSynchronize((hipStream_t)stream);
#elif defined(PSZ_USE_1API)
#define sync_by_stream(stream) ((dpct::queue_ptr*)stream)->wait();
#endif

#if defined(PSZ_USE_CUDA)
#define sync_device cudaDeviceSynchronize();
#elif defined(PSZ_USE_HIP)
#define sync_device hipDeviceSynchronize();
#elif defined(PSZ_USE_1API)
#define sync_device
// TODO there is no device wide sync?
#endif

#endif /* F8DE640C_EFD2_444C_992C_946B18F625F2 */
