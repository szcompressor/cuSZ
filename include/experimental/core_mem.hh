#ifndef F8DE640C_EFD2_444C_992C_946B18F625F2
#define F8DE640C_EFD2_444C_992C_946B18F625F2

using queue_ptr = void*;

template <typename T, psz_platform P = PROPER_GPU_BACKEND>
T* malloc_device(size_t len, queue_ptr stream = nullptr)
{
  T* __a;
#if defined(PSZ_USE_CUDA)
  cudaMallocAsync(&__a, len * sizeof(T), (cudaStream_t)stream);
  cudaMemsetAsync(__a, 0, len * sizeof(T), (cudaStream_t)stream);
  cudaStreamSynchronize((cudaStream_t)stream);
#elif defined(PSZ_USE_HIP)
  hipMallocAsync(&__a, len * sizeof(T), (hipStream_t)stream);
  hipMemsetAsync(__a, 0, len * sizeof(T), (hipStream_t)stream);
  hipStreamSynchronize((hipStream_t)stream);
#elif defined(PSZ_USE_1API)
  if (not stream)
    throw std::runtime_error(
        "[psz::error] SYCL backend "
        "does not allow stream to be null.");
  __a = sycl::malloc_device<T>(len, *((sycl::queue*)stream));
  ((sycl::queue*)stream)->wait();
#endif
  return __a;
}

template <typename T, psz_platform P = PROPER_GPU_BACKEND>
T* malloc_host(size_t len, queue_ptr stream = nullptr)
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
        "[psz::error] SYCL backend "
        "does not allow stream to be null.");
  __a = sycl::malloc_host<T>(len, *((sycl::queue*)stream));
  ((sycl::queue*)stream)->wait();
#endif
  return __a;
}

template <typename T, psz_platform P = PROPER_GPU_BACKEND>
T* malloc_shared(size_t len, queue_ptr stream = nullptr)
{
  T* __a;
#if defined(PSZ_USE_CUDA)
  cudaMallocManaged(&__a, len * sizeof(T));
#elif defined(PSZ_USE_HIP)
  hipMallocManaged(&__a, len * sizeof(T));
#elif defined(PSZ_USE_1API)
  if (not stream)
    throw std::runtime_error(
        "[psz::error] SYCL backend "
        "does not allow stream to be null.");
  __a = sycl::malloc_shared<T>(len, *((sycl::queue*)stream));
  ((sycl::queue*)stream)->wait();
#endif
  return __a;
}

template <typename T, psz_platform P = PROPER_GPU_BACKEND>
void free_device(T* __a, queue_ptr stream = nullptr)
{
#if defined(PSZ_USE_CUDA)
  cudaFree(__a);
#elif defined(PSZ_USE_HIP)
  hipFree(__a);
#elif defined(PSZ_USE_1API)
  if (not stream)
    throw std::runtime_error(
        "[psz::error] SYCL backend "
        "does not allow stream to be null.");
  sycl::free(__a, *((sycl::queue*)stream));
#endif
}

template <typename T, psz_platform P = PROPER_GPU_BACKEND>
void free_host(T* __a, queue_ptr stream = nullptr)
{
#if defined(PSZ_USE_CUDA)
  cudaFree(__a);
#elif defined(PSZ_USE_HIP)
  hipHostFree(__a);
#elif defined(PSZ_USE_1API)
  if (not stream)
    throw std::runtime_error(
        "[psz::error] SYCL backend "
        "does not allow stream to be null.");
  sycl::free(__a, *((sycl::queue*)stream));
#endif
}

template <typename T, psz_platform P = PROPER_GPU_BACKEND>
void free_shared(T* __a, queue_ptr stream = nullptr)
{
  free_device(__a, stream);
}

#endif /* F8DE640C_EFD2_444C_992C_946B18F625F2 */
