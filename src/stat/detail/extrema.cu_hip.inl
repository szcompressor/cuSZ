/**
 * @file extrema.cu_hip.inl
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
__global__ void extrema_kernel(
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

namespace psz {

namespace cu_hip {

template <typename T>
void extrema(T *in, size_t len, T res[4])
{
  static const int MINVAL = 0;
  static const int MAXVAL = 1;
  //   static const int AVGVAL = 2;  // TODO
  static const int RNG = 3;

  // TODO use external stream
  GpuStreamT stream;
  GpuStreamCreate(&stream);

  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto chunk = 32768;
  auto nworker = 128;
  auto R = chunk / nworker;

  T h_min, h_max, failsafe;
  T *d_minel, *d_maxel;

  CHECK_GPU(GpuMalloc(&d_minel, sizeof(T)));
  CHECK_GPU(GpuMalloc(&d_maxel, sizeof(T)));

  // failsafe init
  CHECK_GPU(GpuMemcpy(&failsafe, in, sizeof(T), GpuMemcpyD2H));
  CHECK_GPU(GpuMemcpy(d_minel, in, sizeof(T), GpuMemcpyD2D));
  CHECK_GPU(GpuMemcpy(d_maxel, in, sizeof(T), GpuMemcpyD2D));

// launch
#if defined(PSZ_USE_CUDA)
  psz::extrema_kernel<T><<<div(len, chunk), nworker, sizeof(T) * 2, stream>>>(
      in, len, d_minel, d_maxel, failsafe, R);
#elif defined(PSZ_USE_HIP)
#warning \
    "[psz::warning::caveat] `if-constexpr`-required C++17 is not specified in cmake file, but clang can handle it well."
  if constexpr (std::is_same<T, float>::value) {
    psz::extrema_kernel<float>
        <<<div(len, chunk), nworker, sizeof(float) * 2, stream>>>(
            in, len, d_minel, d_maxel, failsafe, R);
  }
  else {
    throw std::runtime_error(
        "As of now (5.5.30202), HIP does not support 64-bit integer atomic "
        "operation.");
  }
#endif

  GpuStreamSync(stream);

  // collect results
  CHECK_GPU(GpuMemcpy(&h_min, d_minel, sizeof(T), GpuMemcpyD2H));
  CHECK_GPU(GpuMemcpy(&h_max, d_maxel, sizeof(T), GpuMemcpyD2H));

  res[MINVAL] = h_min;
  res[MAXVAL] = h_max;
  res[RNG] = h_max - h_min;

  CHECK_GPU(GpuFree(d_minel));
  CHECK_GPU(GpuFree(d_maxel));

  GpuStreamDestroy(stream);
}

}  // namespace cu_hip
}  // namespace psz

#endif /* E94048A9_2F2B_4A97_AB6E_1B8A3DD6E760 */
