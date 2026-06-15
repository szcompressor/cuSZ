#ifndef _PTB_MEM_CXX_VTYPE_CUH
#define _PTB_MEM_CXX_VTYPE_CUH

#include "cxx_v.cuh"

namespace _ptb {

// clang-format off

template <typename T, int N> 
                      __device__ __forceinline__ _v<T, N> ld_v(const T* p) { return *reinterpret_cast<const _v<T, N>*>(p); }
template <typename T> __device__ __forceinline__ auto ld_128b (const T* p) { return ld_v<T, _128b<T>::N>(p); }
template <typename T> __device__ __forceinline__ auto ld_64b  (const T* p) { return ld_v<T,  _64b<T>::N>(p); }
template <typename T> __device__ __forceinline__ auto ld_32b  (const T* p) { return ld_v<T,  _32b<T>::N>(p); }

// clang-format on

template <typename T, int IPT>
struct _vn {
  static constexpr auto N = _128b<T>::N;  
  static constexpr auto M = IPT / N;
  static_assert(IPT % N == 0, "IPT must fulfill M * 128 bits.");
  _v<T, N> g[M];
  __device__ __forceinline__ T  operator[](int i) const { return reinterpret_cast<const T*>(g)[i]; }
  __device__ __forceinline__ T& operator[](int i)       { return reinterpret_cast<      T*>(g)[i]; }
};

template <typename T, int IPT>
__device__ __forceinline__ _vn<T, IPT> ld_vn(const T* p) {
  using L = _vn<T, IPT>;
  L r;
#pragma unroll
  for (int k = 0; k < L::M; k++) r.g[k] = ld_v<T, L::N>(p + k * L::N);
  return r;
}

}  // namespace _ptb

#endif /* _PTB_MEM_CXX_VTYPE_CUH */
