#ifndef _PORTABLE_MEM_CXX_SMART_PTR_H
#define _PORTABLE_MEM_CXX_SMART_PTR_H

// Jianann Tian
// 24-12-22

#include <memory>

#define GPU_DELETER_D GPU_deleter_device
#define GPU_DELETER_H GPU_deleter_host
#define GPU_DELETER_U GPU_deleter_unified

#define MAKE_UNIQUE_HOST(TYPE, LEN) GPU_make_unique(malloc_h<TYPE>(LEN), GPU_DELETER_H())
#define MAKE_UNIQUE_DEVICE(TYPE, LEN) GPU_make_unique(malloc_d<TYPE>(LEN), GPU_DELETER_D())
#define MAKE_UNIQUE_UNIFIED(TYPE, LEN) GPU_make_unique(malloc_u<TYPE>(LEN), GPU_DELETER_U())

// smart pointer deleter for on-device buffer
struct GPU_deleter_device {
  void* stream;
  GPU_deleter_device(void* s = nullptr) : stream(s) {}
  void operator()(void* ptr) const
  {
#if defined(_PORTABLE_USE_CUDA)
    if (ptr) cudaFree(ptr);
#elif defined(_PORTABLE_USE_HIP)
    if (ptr) hipFree(ptr);
#elif defined(_PORTABLE_USE_1API)
    if (ptr) sycl::free(ptr, *((sycl::queue*)stream));
#endif
  }
};

// smart pointer deleter for on-host pinned memory buffer
struct GPU_deleter_host {
  void* stream;
  GPU_deleter_host(void* s = nullptr) : stream(s) {}
  void operator()(void* ptr) const
  {
#if defined(_PORTABLE_USE_CUDA)
    if (ptr) cudaFreeHost(ptr);
#elif defined(_PORTABLE_USE_HIP)
    if (ptr) hipHostFree(ptr);
#elif defined(_PORTABLE_USE_1API)
    if (ptr) sycl::free(ptr, *((sycl::queue*)stream));
#endif
  }
};

// smart pointer deleter for unifed memory
struct GPU_deleter_unified : GPU_deleter_device {
  GPU_deleter_unified(void* s = nullptr) : GPU_deleter_device(s) {}
};

// GPU unique_ptr checker (template): default to false
template <typename T>
struct is_unique_ptr : std::false_type {};

// GPU unique_ptr checker: specialization with std::true_type
template <typename T, typename Deleter>
struct is_unique_ptr<std::unique_ptr<T, Deleter>> : std::true_type {};
// GPU unique_ptr checker: ::value alias (C++17 onward)
template <typename T>
inline constexpr bool is_unique_ptr_v = is_unique_ptr<T>::value;

// GPU shared_ptr checker (template): default to false
template <typename T>
struct is_shared_ptr : std::false_type {};

// GPU shared_ptr checker: specialization with std::true_type
template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
// GPU shared_ptr checker: ::value alias (C++17 onward)
template <typename T>
inline constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;

// GPU smart pointer checker: unique_ptr or shared_ptr
template <typename T>
struct is_smart_ptr : std::bool_constant<is_unique_ptr_v<T> || is_shared_ptr_v<T>> {};

// GPU smart pointer checker: ::value alias (C++17 onward)
template <typename T>
inline constexpr bool is_smart_ptr_v = is_smart_ptr<T>::value;

template <typename T, typename Deleter>
std::unique_ptr<T[], Deleter> GPU_make_unique(T* ptr, Deleter d)
{
  return std::unique_ptr<T[], Deleter>(ptr, d);
}

template <typename T, typename Deleter>
std::shared_ptr<T[]> GPU_make_shared(T* ptr, Deleter d)
{
  return std::shared_ptr<T[]>(ptr, d);
}

template <typename T>
using GPU_unique_dptr = std::unique_ptr<T, GPU_deleter_device>;
template <typename T>
using GPU_unique_hptr = std::unique_ptr<T, GPU_deleter_host>;
template <typename T>
using GPU_unique_uptr = std::unique_ptr<T, GPU_deleter_unified>;

#endif /* _PORTABLE_MEM_CXX_SMART_PTR_H */
