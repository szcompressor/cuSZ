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

#include "cusz/suint.hh"
#include "port.hh"
#include "subr.cu_hip.inl"

namespace psz {
namespace cuda_hip {
namespace __kernel {
namespace delta_only {

template <typename T, typename Eq, typename FP, int BLOCK, int SEQ>
__global__ void c_lorenzo_1d1l(
    T* data, dim3 len3, dim3 stride3, FP ebx2_r, Eq* delta);

template <typename T, typename Eq, typename FP, int BLOCK, int SEQ>
__global__ void x_lorenzo_1d1l(
    Eq* delta, dim3 len3, dim3 stride3, FP ebx2, T* xdata);

template <typename T, typename Eq, typename FP>
__global__ void c_lorenzo_2d1l(
    T* data, dim3 len3, dim3 stride3, FP ebx2_r, Eq* delta);

template <typename T, typename Eq, typename FP>
__global__ void x_lorenzo_2d1l(
    Eq* delta, dim3 len3, dim3 stride3, FP ebx2, T* xdata);

template <typename T, typename Eq, typename FP>
__global__ void c_lorenzo_3d1l(
    T* data, dim3 len3, dim3 stride3, FP ebx2_r, Eq* eq);

template <typename T, typename Eq, typename FP>
__global__ void x_lorenzo_3d1l(
    Eq* eq, dim3 len3, dim3 stride3, FP ebx2, T* xdata);

}  // namespace delta_only
}  // namespace __kernel
}  // namespace cuda_hip
}  // namespace psz

template <typename T, typename Eq, typename FP, int BLOCK, int SEQ>
__global__ void psz::cuda_hip::__kernel::delta_only::c_lorenzo_1d1l(
    T* data, dim3 len3, dim3 stride3, FP ebx2_r, Eq* eq)
{
  namespace subr_v0 = psz::cuda_hip;

  constexpr auto NTHREAD = BLOCK / SEQ;

  __shared__ T scratch[BLOCK];  // for data and outlier
  __shared__ Eq s_eq[BLOCK];

  T prev{0};
  T thp_data[SEQ];

  auto id_base = blockIdx.x * BLOCK;

  subr_v0::load_prequant_1d<T, FP, NTHREAD, SEQ>(
      data, len3.x, id_base, scratch, thp_data, prev, ebx2_r);
  subr_v0::predict_quantize__no_outlier_1d<T, Eq, SEQ, true>(
      thp_data, s_eq, prev);
  subr_v0::predict_quantize__no_outlier_1d<T, Eq, SEQ, false>(thp_data, s_eq);
  subr_v0::write_1d<Eq, T, NTHREAD, SEQ, false>(
      s_eq, nullptr, len3.x, id_base, eq, nullptr);
}

template <typename T, typename Eq, typename FP, int BLOCK, int SEQ>
__global__ void psz::cuda_hip::__kernel::x_lorenzo_1d1l(  //
    Eq* eq, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata)
{
  namespace subr_v0 = psz::cuda_hip;
  namespace wave32 = psz::cu_hip::wave32;

  constexpr auto NTHREAD = BLOCK / SEQ;  // equiv. to blockDim.x

  __shared__ T scratch[BLOCK];  // for data and outlier
  __shared__ Eq s_eq[BLOCK];
  __shared__ T exch_in[NTHREAD / 32];
  __shared__ T exch_out[NTHREAD / 32];

  T thp_data[SEQ];

  auto id_base = blockIdx.x * BLOCK;

  subr_v0::load_fuse_1d<T, Eq, NTHREAD, SEQ>(
      eq, outlier, len3.x, id_base, radius, scratch, thp_data);
  subr_v0::block_scan_1d<T, SEQ, NTHREAD>(
      thp_data, ebx2, exch_in, exch_out, scratch);
  subr_v0::write_1d<T, T, NTHREAD, SEQ, true>(
      scratch, nullptr, len3.x, id_base, xdata, nullptr);
}

template <typename T, typename Eq, typename FP, int BLOCK, int SEQ>
__global__ void psz::cuda_hip::__kernel::delta_only::x_lorenzo_1d1l(  //
    Eq* eq, dim3 len3, dim3 stride3, FP ebx2, T* xdata)
{
  namespace subr_v0 = psz::cuda_hip;

  constexpr auto NTHREAD = BLOCK / SEQ;  // equiv. to blockDim.x

  __shared__ T scratch[BLOCK];  // for data and outlier
  __shared__ Eq s_eq[BLOCK];
  // compat for wave32 and 64
  __shared__ T exch_in[NTHREAD / 32];
  __shared__ T exch_out[NTHREAD / 32];

  T thp_data[SEQ];

  auto id_base = blockIdx.x * BLOCK;

  subr_v0::delta_only::load_1d<T, Eq, NTHREAD, SEQ>(
      eq, len3.x, id_base, scratch, thp_data);
  subr_v0::block_scan_1d<T, SEQ, NTHREAD>(
      thp_data, ebx2, exch_in, exch_out, scratch);
  subr_v0::write_1d<T, T, NTHREAD, SEQ, true>(
      scratch, nullptr, len3.x, id_base, xdata, nullptr);
}

template <typename T, typename Eq, typename FP>
__global__ void psz::cuda_hip::__kernel::delta_only::c_lorenzo_2d1l(
    T* data, dim3 len3, dim3 stride3, FP ebx2_r, Eq* eq)
{
  namespace subr_v0 = psz::cuda_hip;

  constexpr auto BLOCK = 16;
  constexpr auto YSEQ = 8;

  T center[YSEQ + 1] = {0};  // NW  N       first element <- 0
                             //  W  center

  auto gix = blockIdx.x * BLOCK + threadIdx.x;  // BDX == BLOCK == 16
  auto giy_base =
      blockIdx.y * BLOCK + threadIdx.y * YSEQ;  // BDY * YSEQ = BLOCK == 16

  subr_v0::load_prequant_2d<T, FP, YSEQ>(
      data, len3.x, gix, len3.y, giy_base, stride3.y, ebx2_r, center);
  subr_v0::predict_2d<T, Eq, YSEQ>(center);
  subr_v0::delta_only::quantize_write_2d<T, Eq, YSEQ>(
      center, len3.x, gix, len3.y, giy_base, stride3.y, eq);
}

// 16x16 data block maps to 16x2 (one warp) thread block
template <typename T, typename Eq, typename FP>
__global__ void psz::cuda_hip::__kernel::x_lorenzo_2d1l(  //
    Eq* eq, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata)
{
  namespace subr_v0 = psz::cuda_hip;

  constexpr auto BLOCK = 16;
  constexpr auto YSEQ = BLOCK / 2;  // sequentiality in y direction
  static_assert(BLOCK == 16, "In one case, we need BLOCK for 2D == 16");

  __shared__ T scratch[BLOCK];  // TODO use warp shuffle to eliminate this
  T thread_private[YSEQ];

  auto gix = blockIdx.x * BLOCK + threadIdx.x;
  auto giy_base =
      blockIdx.y * BLOCK + threadIdx.y * YSEQ;  // BDY * YSEQ = BLOCK == 16

  auto get_gid = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

  subr_v0::load_fuse_2d<T, Eq, YSEQ>(
      eq, outlier, len3.x, gix, len3.y, giy_base, stride3.y, radius,
      thread_private);
  subr_v0::block_scan_2d<T, Eq, FP, YSEQ>(thread_private, scratch, ebx2);
  subr_v0::decomp_write_2d<T, YSEQ>(
      thread_private, len3.x, gix, len3.y, giy_base, stride3.y, xdata);
}

// 16x16 data block maps to 16x2 (one warp) thread block
template <typename T, typename Eq, typename FP>
__global__ void psz::cuda_hip::__kernel::delta_only::x_lorenzo_2d1l(  //
    Eq* eq, dim3 len3, dim3 stride3, FP ebx2, T* xdata)
{
  namespace subr_v0 = psz::cuda_hip;

  constexpr auto BLOCK = 16;
  constexpr auto YSEQ = BLOCK / 2;  // sequentiality in y direction
  static_assert(BLOCK == 16, "In one case, we need BLOCK for 2D == 16");

  __shared__ T scratch[BLOCK];  // TODO use warp shuffle to eliminate this
  T thread_private[YSEQ];

  auto gix = blockIdx.x * BLOCK + threadIdx.x;
  auto giy_base =
      blockIdx.y * BLOCK + threadIdx.y * YSEQ;  // BDY * YSEQ = BLOCK == 16

  auto get_gid = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

  subr_v0::delta_only::load_2d<T, Eq, YSEQ>(
      eq, len3.x, gix, len3.y, giy_base, stride3.y, thread_private);
  subr_v0::block_scan_2d<T, Eq, FP, YSEQ>(thread_private, scratch, ebx2);
  subr_v0::decomp_write_2d<T, YSEQ>(
      thread_private, len3.x, gix, len3.y, giy_base, stride3.y, xdata);
}

template <typename T, typename Eq, typename FP>
__global__ void psz::cuda_hip::__kernel::delta_only::c_lorenzo_3d1l(  //
    T* data, dim3 len3, dim3 stride3, FP ebx2_r, Eq* eq)
{
  constexpr auto BLOCK = 8;
  __shared__ T s[9][33];
  T delta[BLOCK + 1] = {0};  // first el = 0

  const auto gix = blockIdx.x * (BLOCK * 4) + threadIdx.x;
  const auto giy = blockIdx.y * BLOCK + threadIdx.y;
  const auto giz_base = blockIdx.z * BLOCK;
  const auto base_id = gix + giy * stride3.y + giz_base * stride3.z;

  auto giz = [&](auto z) { return giz_base + z; };
  auto gid = [&](auto z) { return base_id + z * stride3.z; };

  auto load_prequant_3d = [&]() {
    if (gix < len3.x and giy < len3.y) {
      for (auto z = 0; z < BLOCK; z++)
        if (giz(z) < len3.z)
          delta[z + 1] =
              round(data[gid(z)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_write = [&](T delta, auto x, auto y, auto z, auto gid) {
    if (x < len3.x and y < len3.y and z < len3.z)
      eq[gid] = static_cast<Eq>(delta);
  };

  ////////////////////////////////////////////////////////////////////////////

  load_prequant_3d();

  for (auto z = BLOCK; z > 0; z--) {
    // z-direction
    delta[z] -= delta[z - 1];

    // x-direction
    auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
    if (threadIdx.x % BLOCK > 0) delta[z] -= prev_x;

    // y-direction, exchange via shmem
    // ghost padding along y
    s[threadIdx.y + 1][threadIdx.x] = delta[z];
    __syncthreads();

    delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

    // now delta[z] is delta
    quantize_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    __syncthreads();
  }
}

// 32x8x8 data block maps to 32x1x8 thread block
template <typename T, typename Eq, typename FP>
__global__ void psz::cuda_hip::__kernel::delta_only::x_lorenzo_3d1l(  //
    Eq* eq, dim3 len3, dim3 stride3, FP ebx2, T* xdata)
{
  constexpr auto BLOCK = 8;
  constexpr auto YSEQ = BLOCK;
  static_assert(BLOCK == 8, "In one case, we need BLOCK for 3D == 8");

  __shared__ T scratch[BLOCK][4][8];
  T thread_private[YSEQ];

  auto seg_id = threadIdx.x / 8;
  auto seg_tix = threadIdx.x % 8;

  auto gix = blockIdx.x * (4 * BLOCK) + threadIdx.x;
  auto giy_base = blockIdx.y * BLOCK;
  auto giy = [&](auto y) { return giy_base + y; };
  auto giz = blockIdx.z * BLOCK + threadIdx.z;
  auto gid = [&](auto y) {
    return giz * stride3.z + (giy_base + y) * stride3.y + gix;
  };

  auto load_3d = [&]() {
  // load to thread-private array (fuse at the same time)
#pragma unroll
    for (auto y = 0; y < YSEQ; y++) {
      if (gix < len3.x and giy_base + y < len3.y and giz < len3.z)
        thread_private[y] = static_cast<T>(eq[gid(y)]);  // fuse
      else
        thread_private[y] = 0;
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
  load_3d();
  block_scan_3d();
  decomp_write_3d();
}
