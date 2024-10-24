/**
 * @file l23r.cuhip.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-04-04
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef E3BBDC36_B394_4F63_9E7C_CE601F1C5CA5
#define E3BBDC36_B394_4F63_9E7C_CE601F1C5CA5

#include <cstdint>
#include <type_traits>

#include "cusz/type.h"
#include "kernel/lrz.hh"
#include "mem/compact.hh"
#include "port.hh"
#include "typing.hh"
#include "utils/err.hh"
#include "utils/timer.hh"

#define SETUP_ZIGZAG                                                         \
  using EqUint = typename psz::UInt<sizeof(Eq)>::T;                          \
  using EqInt = typename psz::SInt<sizeof(Eq)>::T;                           \
  static_assert(                                                             \
      std::is_same<Eq, EqUint>::value, "Eq must be unsigned integer type."); \
  auto zigzag_encode = [](EqInt x) -> EqUint {                               \
    return (2 * (x)) ^ ((x) >> (sizeof(Eq) * 8 - 1));                        \
  };

namespace psz {

// TODO (241024) the necessity to keep Fp=T, which triggered double type that
// significantly slowed down the kernel on non-HPC GPU
template <
    typename T, int TileDim, int Seq, typename Eq = uint16_t,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, typename Fp = T, bool ZigZag = false>
__global__ void KERNEL_CUHIP_c_lorenzo_1d1l(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3,
    Eq* const out_eq, CompactVal* const out_cval, CompactIdx* const out_cidx,
    CompactNum* const out_cn, uint16_t const radius, Fp const ebx2_r)
{
  constexpr auto NumThreads = TileDim / Seq;

  SETUP_ZIGZAG;

  __shared__ T s_data[TileDim];
  __shared__ EqUint s_eq_uint[TileDim];

  T _thp_data[Seq + 1] = {0};
  auto prev = [&]() -> T& { return _thp_data[0]; };
  auto thp_data = [&](auto i) -> T& { return _thp_data[i + 1]; };

  auto id_base = blockIdx.x * TileDim;

// dram.in_data to shmem.in_data
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < data_len3.x)
      s_data[threadIdx.x + ix * NumThreads] = round(in_data[id] * ebx2_r);
  }
  __syncthreads();

// shmem.in_data to private.in_data
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
      auto cur_idx = atomicAdd(out_cn, 1);
      out_cidx[cur_idx] = id_base + threadIdx.x * Seq + ix;
      out_cval[cur_idx] = candidate;
    }
  }
  __syncthreads();

// write from shmem.eq to dram.eq
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < data_len3.x)
      out_eq[id] = s_eq_uint[threadIdx.x + ix * NumThreads];
  }

  // end of kernel
}

template <
    typename T, typename Eq = uint16_t, typename CompactVal = T,
    typename CompactIdx = uint32_t, typename CompactNum = uint32_t,
    typename Fp = T, bool ZigZag = false>
__global__ void KERNEL_CUHIP_c_lorenzo_2d1l(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3,
    Eq* const out_eq, CompactVal* const out_cval, CompactIdx* const out_cidx,
    CompactNum* const out_cn, uint16_t const radius, Fp const ebx2_r)
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
  auto g_id = [&](auto i) { return (giy_base + i) * data_leap3.y + gix; };

  // use a warp as two half-warps
  // block_dim = (16, 2, 1) makes a full warp internally

// read to private.in_data (center)
#pragma unroll
  for (auto iy = 0; iy < Yseq; iy++) {
    if (gix < data_len3.x and giy_base + iy < data_len3.y)
      center[iy + 1] = round(in_data[g_id(iy)] * ebx2_r);
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

    if (gix < data_len3.x and giy_base + (i - 1) < data_len3.y) {
      bool quantizable = fabs(center[i]) < radius;
      T candidate;

      if constexpr (ZigZag) {
        candidate = center[i];
        out_eq[gid] = zigzag_encode(quantizable * (EqInt)candidate);
      }
      else {
        candidate = center[i] + radius;
        out_eq[gid] = quantizable * (EqUint)candidate;
      }

      if (not quantizable) {
        auto cur_idx = atomicAdd(out_cn, 1);
        out_cidx[cur_idx] = gid;
        out_cval[cur_idx] = candidate;
      }
    }
  }

  // end of kernel
}

template <
    typename T, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, bool ZigZag = false>
__global__ void KERNEL_CUHIP_c_lorenzo_3d1l(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3,
    Eq* const out_eq, CompactVal* const out_cval, CompactIdx* const out_cidx,
    CompactNum* const out_cn, uint16_t const radius, Fp const ebx2_r)
{
  SETUP_ZIGZAG;

  constexpr auto TileDim = 8;
  __shared__ T s[9][33];
  T delta[TileDim + 1] = {0};  // first el = 0

  const auto gix = blockIdx.x * (TileDim * 4) + threadIdx.x;
  const auto giy = blockIdx.y * TileDim + threadIdx.y;
  const auto giz_base = blockIdx.z * TileDim;
  const auto base_id = gix + giy * data_leap3.y + giz_base * data_leap3.z;

  auto giz = [&](auto z) { return giz_base + z; };
  auto gid = [&](auto z) { return base_id + z * data_leap3.z; };

  auto load_prequant_3d = [&]() {
    if (gix < data_len3.x and giy < data_len3.y) {
      for (auto z = 0; z < TileDim; z++)
        if (giz(z) < data_len3.z)
          delta[z + 1] =
              round(in_data[gid(z)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_compact_write = [&](T delta, auto x, auto y, auto z,
                                    auto gid) {
    bool quantizable = fabs(delta) < radius;

    if (x < data_len3.x and y < data_len3.y and z < data_len3.z) {
      T candidate;

      if constexpr (ZigZag) {
        candidate = delta;
        out_eq[gid] = zigzag_encode(quantizable * (EqInt)candidate);
      }
      else {
        candidate = delta + radius;
        out_eq[gid] = quantizable * (EqUint)candidate;
      }

      if (not quantizable) {
        auto cur_idx = atomicAdd(out_cn, 1);
        out_cidx[cur_idx] = gid;
        out_cval[cur_idx] = candidate;
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

template <
    typename TIN, typename TOUT, bool ReverseProcess = false,
    typename Fp = TIN, int TileDim = 256, int Seq = 8>
__global__ void KERNEL_CUHIP_lorenzo_prequant(
    TIN* const in, size_t const in_len, Fp const ebx2_r, Fp const ebx2,
    TOUT* const out)
{
  constexpr auto NumThreads = TileDim / Seq;
  auto id_base = blockIdx.x * TileDim;

#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    // dram to dram
    if constexpr (not ReverseProcess) {
      if (id < in_len) out[id] = round(in[id] * ebx2_r);
    }
    else {
      if (id < in_len) out[id] = in[id] * ebx2;
    }
  }
}

}  // namespace psz

namespace psz::cuhip {

template <typename T, typename Eq, bool ZigZag>
pszerror GPU_c_lorenzo_nd_with_outlier(
    T* const in_data, dim3 const data_len3, Eq* const out_eq,
    void* out_outlier, f8 const eb, uint16_t const radius, f4* time_elapsed,
    void* stream)
{
  using Compact = typename CompactDram<PROPER_GPU_BACKEND, T>::Compact;
  using namespace psz::kernelconfig;

  auto ot = (Compact*)out_outlier;
  auto d = lorenzo_utils::ndim(data_len3);

  // error bound
  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;
  auto leap3 = dim3(1, data_len3.x, data_len3.x * data_len3.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING((cudaStream_t)stream);

  if (d == 1) {
    psz::KERNEL_CUHIP_c_lorenzo_1d1l<
        T, c_lorenzo<1>::tile.x, c_lorenzo<1>::sequentiality.x, Eq>
        <<<c_lorenzo<1>::thread_grid(data_len3), c_lorenzo<1>::thread_block, 0,
           (cudaStream_t)stream>>>(
            in_data, data_len3, leap3, out_eq, ot->val(), ot->idx(), ot->num(),
            radius, (T)ebx2_r);
  }
  else if (d == 2) {
    psz::KERNEL_CUHIP_c_lorenzo_2d1l<T, Eq>
        <<<c_lorenzo<2>::thread_grid(data_len3), c_lorenzo<2>::thread_block, 0,
           (cudaStream_t)stream>>>(
            in_data, data_len3, leap3, out_eq, ot->val(), ot->idx(), ot->num(),
            radius, (T)ebx2_r);
  }
  else if (d == 3) {
    psz::KERNEL_CUHIP_c_lorenzo_3d1l<T, Eq>
        <<<c_lorenzo<3>::thread_grid(data_len3), c_lorenzo<3>::thread_block, 0,
           (cudaStream_t)stream>>>(
            in_data, data_len3, leap3, out_eq, ot->val(), ot->idx(), ot->num(),
            radius, (T)ebx2_r);
  }

  STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

template <typename TIN, typename TOUT, bool ReverseProcess>
pszerror GPU_lorenzo_prequant(
    TIN* const in, size_t const len, PROPER_EB const eb, TOUT* const out,
    float* time_elapsed, void* stream)
{
  using namespace psz::kernelconfig;
  // error bound
  auto ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING((cudaStream_t)stream);

  psz::KERNEL_CUHIP_lorenzo_prequant<
      TIN, TOUT, ReverseProcess, TIN, c_lorenzo<1>::tile.x,
      c_lorenzo<1>::sequentiality.x>
      <<<c_lorenzo<1>::thread_grid(dim3(len)), c_lorenzo<1>::thread_block, 0,
         (cudaStream_t)stream>>>(in, len, ebx2_r, ebx2, out);

  STOP_GPUEVENT_RECORDING((cudaStream_t)stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time_elapsed);
  DESTROY_GPUEVENT_PAIR;

  return CUSZ_SUCCESS;
}

}  // namespace psz::cuhip

// -----------------------------------------------------------------------------
#define INSTANCIATE_GPU_L23R_3params(T, Eq, ZIGZAG)                           \
  template pszerror psz::cuhip::GPU_c_lorenzo_nd_with_outlier<T, Eq, ZIGZAG>( \
      T* const in_data, dim3 const data_len3, Eq* const out_eq,               \
      void* out_outlier, f8 const eb, uint16_t const radius,                  \
      f4* time_elapsed, void* stream);

#define INSTANCIATE_GPU_L23R_2params(T, Eq)   \
  INSTANCIATE_GPU_L23R_3params(T, Eq, false); \
  INSTANCIATE_GPU_L23R_3params(T, Eq, true);

#define INSTANCIATE_GPU_L23R_1param(T) \
  INSTANCIATE_GPU_L23R_2params(T, u2); \
  INSTANCIATE_GPU_L23R_2params(T, u4);

// -----------------------------------------------------------------------------

#define INSTANCIATE_GPU_L23_PREQ_3params(TIN, TOUT, REV)                    \
  template pszerror psz::cuhip::GPU_lorenzo_prequant<TIN, TOUT, REV>(       \
      TIN* const in, size_t const len, PROPER_EB const eb, TOUT* const out, \
      float* time_elapsed, void* stream);

#define INSTANCIATE_GPU_L23_PREQ_2params(TIN, REV)     \
  INSTANCIATE_GPU_L23_PREQ_3params(TIN, int32_t, REV); \
  INSTANCIATE_GPU_L23_PREQ_3params(TIN, int16_t, REV); \
  INSTANCIATE_GPU_L23_PREQ_3params(TIN, int8_t, REV);

#define INSTANCIATE_GPU_L23_PREQ_1param(T)    \
  INSTANCIATE_GPU_L23_PREQ_2params(T, false); \
  INSTANCIATE_GPU_L23_PREQ_2params(T, true);

//  -----------------------------------------------------------------------------

#endif /* E3BBDC36_B394_4F63_9E7C_CE601F1C5CA5 */
