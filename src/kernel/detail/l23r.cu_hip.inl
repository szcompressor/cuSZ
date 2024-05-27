/**
 * @file l23r.cu_hip.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-04
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef AAC905A6_6314_4E1E_B5CD_BBBA9005A448
#define AAC905A6_6314_4E1E_B5CD_BBBA9005A448

#include <stdint.h>

#include <type_traits>

#include "kernel/lrz.hh"
#include "mem/compact.hh"
#include "port.hh"

#define SETUP_ZIGZAG                                                         \
  using EqUint = typename psz::typing::UInt<sizeof(Eq)>::T;                  \
  using EqInt = typename psz::typing::Int<sizeof(Eq)>::T;                    \
  static_assert(                                                             \
      std::is_same<Eq, EqUint>::value, "Eq must be unsigned integer type."); \
  auto zigzag_encode = [](EqInt x) -> EqUint {                               \
    return (2 * (x)) ^ ((x) >> (sizeof(Eq) * 8 - 1));                        \
  };

namespace psz {
namespace rolling {

template <
    typename T, typename Eq = uint32_t, typename Fp = T, int TileDim = 256,
    int Seq = 8, typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, bool ZigZag = false>
__global__ void c_lorenzo_1d1l(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn)
{
  constexpr auto NumThreads = TileDim / Seq;

  SETUP_ZIGZAG;

  __shared__ T s_data[TileDim];
  __shared__ EqUint s_eq_uint[TileDim];

  T _thp_data[Seq + 1] = {0};
  auto prev = [&]() -> T& { return _thp_data[0]; };
  auto thp_data = [&](auto i) -> T& { return _thp_data[i + 1]; };

  auto id_base = blockIdx.x * TileDim;

// dram.data to shmem.data
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < len3.x)
      s_data[threadIdx.x + ix * NumThreads] = round(data[id] * ebx2_r);
  }
  __syncthreads();

// shmem.data to private.data
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++)
    thp_data(ix) = s_data[threadIdx.x * Seq + ix];
  if (threadIdx.x > 0)
    prev() = s_data[threadIdx.x * Seq - 1];  // from last thread
  __syncthreads();

  // quantize & write back to shmem.eq
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    T delta = thp_data(ix) - thp_data(ix - 1);
    bool quantizable = fabs(delta) < radius;
    T candidate;

    if constexpr (ZigZag) {
      candidate = delta;
      s_eq_uint[ix + threadIdx.x * Seq] =
          zigzag_encode(quantizable * (EqInt)candidate);
    }
    else {
      candidate = delta + radius;
      s_eq_uint[ix + threadIdx.x * Seq] = quantizable * (EqUint)candidate;
    }

    if (not quantizable) {
      auto cur_idx = atomicAdd(cn, 1);
      cidx[cur_idx] = id_base + threadIdx.x * Seq + ix;
      cval[cur_idx] = candidate;
    }
  }
  __syncthreads();

// write from shmem.eq to dram.eq
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < len3.x) eq[id] = s_eq_uint[threadIdx.x + ix * NumThreads];
  }

  // end of kernel
}

template <
    typename T, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, bool ZigZag = false>
__global__ void c_lorenzo_2d1l(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn)
{
  constexpr auto TileDim = 16;
  constexpr auto Yseq = 8;

  SETUP_ZIGZAG;

  // NW  N       first el <- 0
  //  W  center
  T center[Yseq + 1] = {0};
  // auto prev = [&]() -> T& { return _center[0]; };
  // auto center = [&](auto i) -> T& { return _center[i + 1]; };
  // auto last = [&]() -> T& { return _center[Yseq]; };

  // BDX == TileDim == 16, BDY * Yseq = TileDim == 16
  auto gix = blockIdx.x * TileDim + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim + threadIdx.y * Yseq;
  auto g_id = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

  // use a warp as two half-warps
  // block_dim = (16, 2, 1) makes a full warp internally

// read to private.data (center)
#pragma unroll
  for (auto iy = 0; iy < Yseq; iy++) {
    if (gix < len3.x and giy_base + iy < len3.y)
      center[iy + 1] = round(data[g_id(iy)] * ebx2_r);
  }
  // same-warp, next-16
  auto tmp = __shfl_up_sync(0xffffffff, center[Yseq], 16, 32);
  if (threadIdx.y == 1) center[0] = tmp;

// prediction (apply Lorenzo filter)
#pragma unroll
  for (auto i = Yseq; i > 0; i--) {
    // with center[i-1] intact in this iteration
    center[i] -= center[i - 1];
    // within a halfwarp (32/2)
    auto west = __shfl_up_sync(0xffffffff, center[i], 1, 16);
    if (threadIdx.x > 0) center[i] -= west;  // delta
  }
  __syncthreads();

#pragma unroll
  for (auto i = 1; i < Yseq + 1; i++) {
    auto gid = g_id(i - 1);

    if (gix < len3.x and giy_base + (i - 1) < len3.y) {
      bool quantizable = fabs(center[i]) < radius;
      T candidate;

      if constexpr (ZigZag) {
        candidate = center[i];
        eq[gid] = zigzag_encode(quantizable * (EqInt)candidate);
      }
      else {
        candidate = center[i] + radius;
        eq[gid] = quantizable * (EqUint)candidate;
      }

      if (not quantizable) {
        auto cur_idx = atomicAdd(cn, 1);
        cidx[cur_idx] = gid;
        cval[cur_idx] = candidate;
      }
    }
  }

  // end of kernel
}

template <
    typename T, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, bool ZigZag = false>
__global__ void c_lorenzo_3d1l(
    T* data, dim3 len3, dim3 stride3, int radius, Fp ebx2_r, Eq* eq,
    CompactVal* cval, CompactIdx* cidx, CompactNum* cn)
{
  SETUP_ZIGZAG;

  constexpr auto TileDim = 8;
  __shared__ T s[9][33];
  T delta[TileDim + 1] = {0};  // first el = 0

  const auto gix = blockIdx.x * (TileDim * 4) + threadIdx.x;
  const auto giy = blockIdx.y * TileDim + threadIdx.y;
  const auto giz_base = blockIdx.z * TileDim;
  const auto base_id = gix + giy * stride3.y + giz_base * stride3.z;

  auto giz = [&](auto z) { return giz_base + z; };
  auto gid = [&](auto z) { return base_id + z * stride3.z; };

  auto load_prequant_3d = [&]() {
    if (gix < len3.x and giy < len3.y) {
      for (auto z = 0; z < TileDim; z++)
        if (giz(z) < len3.z)
          delta[z + 1] =
              round(data[gid(z)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_compact_write = [&](T delta, auto x, auto y, auto z,
                                    auto gid) {
    bool quantizable = fabs(delta) < radius;

    if (x < len3.x and y < len3.y and z < len3.z) {
      T candidate;

      if constexpr (ZigZag) {
        candidate = delta;
        eq[gid] = zigzag_encode(quantizable * (EqInt)candidate);
      }
      else {
        candidate = delta + radius;
        eq[gid] = quantizable * (EqUint)candidate;
      }

      if (not quantizable) {
        auto cur_idx = atomicAdd(cn, 1);
        cidx[cur_idx] = gid;
        cval[cur_idx] = candidate;
      }
    }
  };

  ////////////////////////////////////////////////////////////////////////////

  load_prequant_3d();

  for (auto z = TileDim; z > 0; z--) {
    // z-direction
    delta[z] -= delta[z - 1];

    // x-direction
    auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
    if (threadIdx.x % TileDim > 0) delta[z] -= prev_x;

    // y-direction, exchange via shmem
    // ghost padding along y
    s[threadIdx.y + 1][threadIdx.x] = delta[z];
    __syncthreads();

    delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

    // now delta[z] is delta
    quantize_compact_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    __syncthreads();
  }
}

}  // namespace rolling
}  // namespace psz

#endif /* AAC905A6_6314_4E1E_B5CD_BBBA9005A448 */
