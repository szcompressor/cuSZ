/**
 * @file extrema_g.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-08-19
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef E94048A9_2F2B_4A97_AB6E_1B8A3DD6E760
#define E94048A9_2F2B_4A97_AB6E_1B8A3DD6E760

#include <math.h>
#include <stdio.h>

#include <type_traits>
namespace psz {

namespace {

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

#define __ATOMIC_PLUGIN                                                     \
  constexpr auto bytewidth = sizeof(T);                                     \
  using itype = typename matchby<bytewidth>::itype;                         \
  using utype = typename matchby<bytewidth>::utype;                         \
  using ftype = typename matchby<bytewidth>::ftype;                         \
  static_assert(std::is_same<T, ftype>::value, "T and ftype don't match."); \
  auto fp_as_int = [](T fpval) -> itype {                                   \
    return *reinterpret_cast<itype *>(&fpval);                              \
  };                                                                        \
  auto fp_as_uint = [](T fpval) -> utype {                                  \
    return *reinterpret_cast<utype *>(&fpval);                              \
  };                                                                        \
  auto int_as_fp = [](itype ival) -> T {                                    \
    return *reinterpret_cast<T *>(&ival);                                   \
  };                                                                        \
  auto uint_as_fp = [](utype uval) -> T {                                   \
    return *reinterpret_cast<T *>(&uval);                                   \
  };

// modifed from https://stackoverflow.com/a/51549250 (CC BY-SA 4.0)
// https://stackoverflow.com/a/72461459
template <typename T>
__device__ __forceinline__ T atomicMinFp(T *addr, T value)
{
  __ATOMIC_PLUGIN
  auto old = !signbit(value)
                 ? int_as_fp(atomicMin((itype *)addr, fp_as_int(value)))
                 : uint_as_fp(atomicMax((utype *)addr, fp_as_uint(value)));
  return old;
}

template <typename T>
__device__ __forceinline__ T atomicMaxFp(T *addr, T value)
{
  __ATOMIC_PLUGIN
  auto old = !signbit(value)
                 ? int_as_fp(atomicMax((itype *)addr, fp_as_int(value)))
                 : uint_as_fp(atomicMin((utype *)addr, fp_as_uint(value)));
  return old;
}

}  // namespace

template <typename T>
__global__ void extrema_cu(
    T *in, size_t const len, T *minel, T *maxel, T const failsafe, int const R)
{
  __shared__ T shared_minv, shared_maxv;
  T tp_minv, tp_maxv;

  auto entry = (blockDim.x * R) * blockIdx.x + threadIdx.x;
  auto _idx = [&](auto r) { return entry + (r * blockDim.x); };

  // failsafe; require external setup
  tp_minv = failsafe, tp_maxv = failsafe;
  if (threadIdx.x == 0) shared_minv = failsafe, shared_maxv = failsafe;

  __syncthreads();

  for (auto r = 0; r < R; r++) {
    auto idx = _idx(r);
    if (idx < len) {
      auto val = in[idx];

      tp_minv = min(tp_minv, val);
      tp_maxv = max(tp_maxv, val);
    }
  }
  __syncthreads();

  atomicMinFp<T>(&shared_minv, tp_minv);
  atomicMaxFp<T>(&shared_maxv, tp_maxv);
  __syncthreads();

  if (threadIdx.x == 0) {
    auto oldmin = atomicMinFp<T>(minel, shared_minv);
    auto oldmax = atomicMaxFp<T>(maxel, shared_maxv);
  }
}

}  // namespace psz

#endif /* E94048A9_2F2B_4A97_AB6E_1B8A3DD6E760 */
