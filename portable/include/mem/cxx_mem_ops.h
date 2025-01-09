template <_portable_mem_control>
struct _memcpy_direcion;

template <>
struct _memcpy_direcion<H2D> {
#if defined(_PORTABLE_USE_CUDA)
  static const cudaMemcpyKind direction = cudaMemcpyHostToHost;
#elif defined(_PORTABLE_USE_HIP)
  static const hipMemcpyKind direction = hipMemcpyHostToHost;
#elif defined(_PORTABLE_USE_1API)
// no need
#endif
};

template <>
struct _memcpy_direcion<D2H> {
#if defined(_PORTABLE_USE_CUDA)
  static const cudaMemcpyKind direction = cudaMemcpyDeviceToHost;
#elif defined(_PORTABLE_USE_HIP)
  static const hipMemcpyKind direction = hipMemcpyDeviceToHost;
#elif defined(_PORTABLE_USE_1API)
// no need
#endif
};

template <>
struct _memcpy_direcion<D2D> {
#if defined(_PORTABLE_USE_CUDA)
  static const cudaMemcpyKind direction = cudaMemcpyDeviceToDevice;
#elif defined(_PORTABLE_USE_HIP)
  static const hipMemcpyKind direction = hipMemcpyDeviceToDevice;
#elif defined(_PORTABLE_USE_1API)
// no need
#endif
};

#define malloc_d malloc_device
#define malloc_h malloc_host
#define malloc_u malloc_unified
#define free_d free_device
#define free_h free_host
#define free_u free_unified

template <typename T>
T* malloc_device(size_t const len, void* stream = nullptr)
{
  T* __a;
#if defined(_PORTABLE_USE_CUDA)
  cudaMalloc(&__a, len * sizeof(T));
  cudaMemset(__a, 0, len * sizeof(T));
#elif defined(_PORTABLE_USE_HIP)
  hipMalloc(&__a, len * sizeof(T));
  hipMemset(__a, 0, len * sizeof(T));
#elif defined(_PORTABLE_USE_1API)
  if (not stream)
    throw std::runtime_error("[psz::error] SYCL backend does not allow stream to be null.");
  __a = sycl::malloc_device<T>(len, *((sycl::queue*)stream));
  ((sycl::queue*)stream)->wait();
#endif
  return __a;
}

template <typename T>
T* malloc_host(size_t const len, void* stream = nullptr)
{
  T* __a;
#if defined(_PORTABLE_USE_CUDA)
  cudaMallocHost(&__a, len * sizeof(T));
  memset(__a, 0, len * sizeof(T));
#elif defined(_PORTABLE_USE_HIP)
  hipHostMalloc(&__a, len * sizeof(T));
  memset(__a, 0, len * sizeof(T));
#elif defined(_PORTABLE_USE_1API)
  if (not stream)
    throw std::runtime_error("[psz::error] SYCL backend does not allow stream to be null.");
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
#if defined(_PORTABLE_USE_CUDA)
  cudaMallocManaged(&__a, len * sizeof(T));
#elif defined(_PORTABLE_USE_HIP)
  hipMallocManaged(&__a, len * sizeof(T));
#elif defined(_PORTABLE_USE_1API)
  if (not stream)
    throw std::runtime_error("[psz::error] SYCL backend does not allow stream to be null.");
  __a = sycl::malloc_unified<T>(len, *((sycl::queue*)stream));
  ((sycl::queue*)stream)->wait();
#endif
  return __a;
}

template <typename T>
void free_device(T* __a, void* stream = nullptr)
{
#if defined(_PORTABLE_USE_CUDA)
  cudaFree(__a);
#elif defined(_PORTABLE_USE_HIP)
  hipFree(__a);
#elif defined(_PORTABLE_USE_1API)
  if (not stream)
    throw std::runtime_error("[psz::error] SYCL backend does not allow stream to be null.");
  sycl::free(__a, *((sycl::queue*)stream));
#endif
}

template <typename T>
void free_host(T* __a, void* stream = nullptr)
{
#if defined(_PORTABLE_USE_CUDA)
  cudaFreeHost(__a);
#elif defined(_PORTABLE_USE_HIP)
  hipHostFree(__a);
#elif defined(_PORTABLE_USE_1API)
  if (not stream)
    throw std::runtime_error("[psz::error] SYCL backend does not allow stream to be null.");
  sycl::free(__a, *((sycl::queue*)stream));
#endif
}

#define free_shared free_unified

template <typename T>
void free_unified(T* __a, void* stream = nullptr)
{
  free_device(__a, stream);
}

template <_portable_mem_control DIR, typename T>
void memcpy_allkinds(T* dst, T* src, size_t const len, void* stream = nullptr)
{
  static_assert(std::is_trivially_copyable_v<T>, "T must be a trivially copyable type.");

  constexpr auto direction = _memcpy_direcion<DIR>::direction;

#if defined(_PORTABLE_USE_CUDA)
  cudaMemcpy(dst, src, sizeof(T) * len, direction);
#elif defined(_PORTABLE_USE_HIP)
  hipMemcpy(dst, src, sizeof(T) * len, direction);
#elif defined(_PORTABLE_USE_1API)
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
void memcpy_allkinds_async(T* dst, T* src, size_t const len, void* stream = nullptr)
{
  static_assert(std::is_trivially_copyable_v<T>, "T must be a trivially copyable type.");

  constexpr auto direction = _memcpy_direcion<DIR>::direction;

#if defined(_PORTABLE_USE_CUDA)
  cudaMemcpyAsync(dst, src, sizeof(T) * len, direction, (cudaStream_t)stream);
#elif defined(_PORTABLE_USE_HIP)
  hipMemcpyAsync(dst, src, sizeof(T) * len, direction, (hipStream_t)stream);
#elif defined(_PORTABLE_USE_1API)
  ((sycl::queue*)stream)->memcpy(dst, src, sizeof(T) * len);
#endif
}

template <typename T>
void memset_device(T* __a, size_t const len, int value = 0)
{
  static_assert(std::is_trivially_copyable_v<T>, "T must be a trivially copyable type.");

#if defined(_PORTABLE_USE_CUDA)
  cudaMemset(__a, value, sizeof(T) * len);
#elif defined(_PORTABLE_USE_HIP)
  hipMemset(__a, value, sizeof(T) * len);
#elif defined(_PORTABLE_USE_1API)
  memset(__a, value, sizeof(T) * len);
#endif
}

template <typename T>
void memset_host(T* __a, size_t const len, int value = 0)
{
  static_assert(std::is_trivially_copyable_v<T>, "T must be a trivially copyable type.");

  memset(__a, value, sizeof(T) * len);
}
