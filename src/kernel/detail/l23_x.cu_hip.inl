/**
 * @file lorenzo23.inl
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#ifndef D1C4C282_1485_4677_BC6B_F3DB79ED853E
#define D1C4C282_1485_4677_BC6B_F3DB79ED853E

#include "cusz/suint.hh"
#include "port.hh"
// #include "subr.cu_hip.inl"
#include "wave32.cu_hip.inl"

namespace psz {
namespace cuda_hip {
namespace __kernel {

template <typename T, typename Eq, typename FP = T, int BLOCK, int SEQ>
__global__ void x_lorenzo_1d1l(  //
    Eq* eq, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata)
{
  constexpr auto NTHREAD = BLOCK / SEQ;  // equiv. to blockDim.x

  __shared__ T scratch[BLOCK];  // for data and outlier
  __shared__ Eq s_eq[BLOCK];
  __shared__ T exch_in[NTHREAD / 32];
  __shared__ T exch_out[NTHREAD / 32];

  T thp_data[SEQ];

  auto id_base = blockIdx.x * BLOCK;

  auto load_fuse_1d = [&]() {
#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
      auto local_id = threadIdx.x + i * NTHREAD;
      auto id = id_base + local_id;
      if (id < len3.x)
        scratch[local_id] = outlier[id] + static_cast<T>(eq[id]) - radius;
    }
    __syncthreads();

#pragma unroll
    for (auto i = 0; i < SEQ; i++)
      thp_data[i] = scratch[threadIdx.x * SEQ + i];
    __syncthreads();
  };

  auto block_scan_1d = [&]() {
    namespace wave32 = psz::cu_hip::wave32;
    wave32::intrawarp_inclscan_1d<T, SEQ>(thp_data);
    wave32::intrablock_exclscan_1d<T, SEQ, NTHREAD>(
        thp_data, exch_in, exch_out);

    // put back to shmem
#pragma unroll
    for (auto i = 0; i < SEQ; i++)
      scratch[threadIdx.x * SEQ + i] = thp_data[i] * ebx2;
    __syncthreads();
  };

  auto write_1d = [&]() {
#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
      auto local_id = threadIdx.x + i * NTHREAD;
      auto id = id_base + local_id;
      if (id < len3.x) xdata[id] = scratch[local_id];
    }
  };

  /*-----------*/

  load_fuse_1d();
  block_scan_1d();
  write_1d();
}

//   2D partial sum: memory layout
//
//       ------> gix (x)
//
//   |   t(0,0)       t(0,1)       t(0,2)       t(0,3)       ... t(0,f)
//   |
//   |   thp(0,0)[0]  thp(0,0)[0]  thp(0,0)[0]  thp(0,0)[0]
//  giy  thp(0,0)[1]  thp(0,0)[1]  thp(0,0)[1]  thp(0,0)[1]
//  (y)  |            |            |            |
//       thp(0,0)[7]  thp(0,0)[7]  thp(0,0)[7]  thp(0,0)[7]
//
//   |   t(1,0)       t(1,1)       t(1,2)       t(1,3)       ... t(1,f)
//   |
//   |   thp(1,0)[0]  thp(1,0)[0]  thp(1,0)[0]  thp(1,0)[0]
//  giy  thp(1,0)[1]  thp(1,0)[1]  thp(1,0)[1]  thp(1,0)[1]
//  (y)  |            |            |            |
//       thp(1,0)[7]  thp(1,0)[7]  thp(1,0)[7]  thp(1,0)[7]

template <typename T, typename Eq, typename FP = T>
__global__ void x_lorenzo_2d1l(  //
    Eq* eq, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata)
{
  constexpr auto BLOCK = 16;
  constexpr auto YSEQ = BLOCK / 2;  // sequentiality in y direction
  static_assert(BLOCK == 16, "In one case, we need BLOCK for 2D == 16");

  __shared__ T scratch[BLOCK];  // TODO use warp shuffle to eliminate this
  T thp_data[YSEQ] = {0};

  auto gix = blockIdx.x * BLOCK + threadIdx.x;
  auto giy_base =
      blockIdx.y * BLOCK + threadIdx.y * YSEQ;  // BDY * YSEQ = BLOCK == 16

  auto get_gid = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

  auto load_fuse_2d = [&]() {

#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      auto gid = get_gid(i);
      if (gix < len3.x and (giy_base + i) < len3.y)
        thp_data[i] = outlier[gid] + static_cast<T>(eq[gid]) - radius;  // fuse
    }
  };

  // partial-sum along y-axis, sequantially
  // then, in-warp partial-sum along x-axis
  auto block_scan_2d = [&]() {
    for (auto i = 1; i < YSEQ; i++) thp_data[i] += thp_data[i - 1];
    // two-pass: store for cross-thread-private update
    // TODO shuffle up by 16 in the same warp
    if (threadIdx.y == 0) scratch[threadIdx.x] = thp_data[YSEQ - 1];
    __syncthreads();
    // broadcast the partial-sum result from a previous segment
    if (threadIdx.y == 1) {
      auto tmp = scratch[threadIdx.x];
#pragma unroll
      for (auto i = 0; i < YSEQ; i++)
        thp_data[i] += tmp;  // regression as pointer
    }
    // implicit sync as there is half-warp divergence

#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      for (auto d = 1; d < BLOCK; d *= 2) {
        T n = __shfl_up_sync(
            0xffffffff, thp_data[i], d, 16);  // half-warp shuffle
        if (threadIdx.x >= d) thp_data[i] += n;
      }
      thp_data[i] *= ebx2;  // scale accordingly
    }
  };

  auto decomp_write_2d = [&]() {
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      auto gid = get_gid(i);
      if (gix < len3.x and (giy_base + i) < len3.y) xdata[gid] = thp_data[i];
    }
  };

  /*-----------*/

  load_fuse_2d();
  block_scan_2d();
  decomp_write_2d();
}

// 32x8x8 data block maps to 32x1x8 thread block
template <typename T, typename Eq, typename FP = T>
__global__ void x_lorenzo_3d1l(  //
    Eq* eq, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata)
{
  constexpr auto BLOCK = 8;
  constexpr auto YSEQ = BLOCK;
  static_assert(BLOCK == 8, "In one case, we need BLOCK for 3D == 8");

  __shared__ T scratch[BLOCK][4][8];
  T thread_private[YSEQ] = {0};

  auto seg_id = threadIdx.x / 8;
  auto seg_tix = threadIdx.x % 8;

  auto gix = blockIdx.x * (4 * BLOCK) + threadIdx.x;
  auto giy_base = blockIdx.y * BLOCK;
  auto giy = [&](auto y) { return giy_base + y; };
  auto giz = blockIdx.z * BLOCK + threadIdx.z;
  auto gid = [&](auto y) {
    return giz * stride3.z + (giy_base + y) * stride3.y + gix;
  };

  auto load_fuse_3d = [&]() {
  // load to thread-private array (fuse at the same time)
#pragma unroll
    for (auto y = 0; y < YSEQ; y++) {
      if (gix < len3.x and giy_base + y < len3.y and giz < len3.z)
        thread_private[y] =
            outlier[gid(y)] + static_cast<T>(eq[gid(y)]) - radius;  // fuse
    }
  };

  auto block_scan_3d = [&]() {
    // partial-sum along y-axis, sequentially
    for (auto y = 1; y < YSEQ; y++) thread_private[y] += thread_private[y - 1];

#pragma unroll
    for (auto i = 0; i < BLOCK; i++) {
      // ND partial-sums along x- and z-axis
      // in-warp shuffle used: in order to perform, it's transposed after
      // X-partial sum
      T val = thread_private[i];

      for (auto dist = 1; dist < BLOCK; dist *= 2) {
        auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
        if (seg_tix >= dist) val += addend;
      }

      // x-z transpose
      scratch[threadIdx.z][seg_id][seg_tix] = val;
      __syncthreads();
      val = scratch[seg_tix][seg_id][threadIdx.z];
      __syncthreads();

      for (auto dist = 1; dist < BLOCK; dist *= 2) {
        auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
        if (seg_tix >= dist) val += addend;
      }

      scratch[threadIdx.z][seg_id][seg_tix] = val;
      __syncthreads();
      val = scratch[seg_tix][seg_id][threadIdx.z];
      __syncthreads();

      thread_private[i] = val;
    }
  };

  auto decomp_write_3d = [&]() {
#pragma unroll
    for (auto y = 0; y < YSEQ; y++)
      if (gix < len3.x and giy(y) < len3.y and giz < len3.z)
        xdata[gid(y)] = thread_private[y] * ebx2;
  };

  ////////////////////////////////////////////////////////////////////////////
  load_fuse_3d();
  block_scan_3d();
  decomp_write_3d();
}

}  // namespace __kernel
}  // namespace cuda_hip
}  // namespace psz

#endif /* D1C4C282_1485_4677_BC6B_F3DB79ED853E */
