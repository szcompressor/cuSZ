/**
 * @file atomics.cuhip.inl
 * @brief Common atomic operations for CUDA/HIP backends
 * @date 2024-06-02 (original)
 * @date 2026-01-07 (refactored to common file)
 */

#ifndef EVAL_DETAIL_ATOMICS_CUHIP_INL
#define EVAL_DETAIL_ATOMICS_CUHIP_INL

#include <cmath>
#include <type_traits>

namespace psz {

namespace {

// Type matching utilities for atomic operations
template <int bytewidth>
struct matchby;

template <>
struct matchby<4> {
  using utype = unsigned int;
  using itype = int;
  using ftype = float;
};

template <>
struct matchby<8> {
  using utype = unsigned long long;
  using itype = long long;
  using ftype = double;
};

#define __ATOMIC_PLUGIN                                                                 \
  constexpr auto bytewidth = sizeof(T);                                                 \
  using itype = typename matchby<bytewidth>::itype;                                     \
  using utype = typename matchby<bytewidth>::utype;                                     \
  using ftype = typename matchby<bytewidth>::ftype;                                     \
  static_assert(std::is_same<T, ftype>::value, "T and ftype don't match.");             \
  auto fp_as_int = [](T fpval) -> itype { return *reinterpret_cast<itype*>(&fpval); };  \
  auto fp_as_uint = [](T fpval) -> utype { return *reinterpret_cast<utype*>(&fpval); }; \
  auto int_as_fp = [](itype ival) -> T { return *reinterpret_cast<T*>(&ival); };        \
  auto uint_as_fp = [](utype uval) -> T { return *reinterpret_cast<T*>(&uval); };

// Modified from https://stackoverflow.com/a/51549250 (CC BY-SA 4.0)
// https://stackoverflow.com/a/72461459
template <typename T>
__device__ __forceinline__ T atomicMinFp(T* addr, T value)
{
  __ATOMIC_PLUGIN
  auto old = !signbit(value) ? int_as_fp(atomicMin((itype*)addr, fp_as_int(value)))
                             : uint_as_fp(atomicMax((utype*)addr, fp_as_uint(value)));
  return old;
}

template <typename T>
__device__ __forceinline__ T atomicMaxFp(T* addr, T value)
{
  __ATOMIC_PLUGIN
  auto old = !signbit(value) ? int_as_fp(atomicMax((itype*)addr, fp_as_int(value)))
                             : uint_as_fp(atomicMin((utype*)addr, fp_as_uint(value)));
  return old;
}

template <typename T>
__device__ __forceinline__ T atomicAddFp(T* addr, T value)
{
  if constexpr (std::is_same<T, float>::value) { return atomicAdd(addr, value); }
  else if constexpr (std::is_same<T, double>::value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 600)
    return atomicAdd(addr, value);
#else
    auto addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *addr_as_ull, assumed;
    do {
      assumed = old;
      double next = __longlong_as_double(assumed) + value;
      old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(next));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
  }
  else {
    return atomicAdd(addr, value);
  }
}

}  // namespace

}  // namespace psz

#endif  // EVAL_DETAIL_ATOMICS_CUHIP_INL
