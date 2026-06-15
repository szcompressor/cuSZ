#ifndef _PTB_MEM_CXX_V_CUH
#define _PTB_MEM_CXX_V_CUH

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <vector_types.h>

#include <type_traits>

#include "c_type.h"

#define CAN_BE(_input) std::is_same_v<T, _input>

using f1_43 = __nv_fp8_e4m3;
using f1_52 = __nv_fp8_e5m2;
using f2    = half;

namespace _ptb {

// N-element-of-T low-level internals
// FIXME: cannot take even lower-precision types
template <typename T, int N_>
struct alignas(N_ * sizeof(T)) _v {
  static constexpr int N = N_;
  static_assert(
      N * sizeof(T) == 4 or N * sizeof(T) == 8 or N * sizeof(T) == 16,
      "N * sizeof(T) must be 4/8/16 bytes.");

  static_assert(
      CAN_BE(u1) or CAN_BE(i1) or CAN_BE(f1_43) or CAN_BE(f1_52) or  //
          CAN_BE(u2) or CAN_BE(i2) or CAN_BE(f2) or                  //
          CAN_BE(u4) or CAN_BE(i4) or CAN_BE(f4) or                  //
          CAN_BE(u8) or CAN_BE(ull) or CAN_BE(i8) or CAN_BE(f8),
      "Wrong typing.");

  union {
    T  x[N];
    u1 _[N * sizeof(T)];
  };

  __device__ __forceinline__ T operator[](int i) const { return x[i]; }
  __device__ __forceinline__ T& operator[](int i) { return x[i]; }
};

// clang-format off

// N-element-of-T facade for *-bit single load
template <typename T> using _128b = _v<T, 16 / sizeof(T)>;
template <typename T> using  _64b = _v<T,  8 / sizeof(T)>;
template <typename T> using  _32b = _v<T,  4 / sizeof(T)>;

// clang-format on

// exhaustive list of hw-supported vtypes
// 1B: {u,i,f}1
// using u1x1  = _v<u1, 1>;
// using u1x2  = _v<u1, 2>;
using u1x4  = _v<u1, 4>;
using u1x8  = _v<u1, 8>;
using u1x16 = _v<u1, 16>;

// using i1x1  = _v<i1, 1>;
// using i1x2  = _v<i1, 2>;
using i1x4  = _v<i1, 4>;
using i1x8  = _v<i1, 8>;
using i1x16 = _v<i1, 16>;

// using f1x1_43  = _v<f1_43, 1>;
// using f1x2_43  = _v<f1_43, 2>;
using f1x4_43  = _v<f1_43, 4>;
using f1x8_43  = _v<f1_43, 8>;
using f1x16_43 = _v<f1_43, 16>;

// using f1x1_52  = _v<f1_52, 1>;
// using f1x2_52  = _v<f1_52, 2>;
using f1x4_52  = _v<f1_52, 4>;
using f1x8_52  = _v<f1_52, 8>;
using f1x16_52 = _v<f1_52, 16>;

// 2B: {u,i,f}2
// using u2x1 = _v<u2, 1>;
using u2x2 = _v<u2, 2>;
using u2x4 = _v<u2, 4>;
using u2x8 = _v<u2, 8>;

// using i2x1 = _v<i2, 1>;
using i2x2 = _v<i2, 2>;
using i2x4 = _v<i2, 4>;
using i2x8 = _v<i2, 8>;

// using f2x1 = _v<f2, 1>;
using f2x2 = _v<f2, 2>;
using f2x4 = _v<f2, 4>;
using f2x8 = _v<f2, 8>;

// 4B: {u,i,f}4
using u4x1 = _v<u4, 1>;
using u4x2 = _v<u4, 2>;
using u4x4 = _v<u4, 4>;

using i4x1 = _v<i4, 1>;
using i4x2 = _v<i4, 2>;
using i4x4 = _v<i4, 4>;

using f4x1 = _v<f4, 1>;
using f4x2 = _v<f4, 2>;
using f4x4 = _v<f4, 4>;

// 8B: {u,i,f}8
using u8x1 = _v<u8, 1>;
using u8x2 = _v<u8, 2>;

using i8x1 = _v<i8, 1>;
using i8x2 = _v<i8, 2>;

using f8x1 = _v<f8, 1>;
using f8x2 = _v<f8, 2>;

}  // namespace _ptb

#endif /* _PTB_MEM_CXX_V_CUH */
