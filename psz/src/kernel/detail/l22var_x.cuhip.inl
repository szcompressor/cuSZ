/**
 * @file l23_fzgpu.cuhip.inl
 * @author Jiannan Tian
 * @brief Adapted from l21conf for FZ-GPU (HPDC '23) (decompression part). This
 * only reflects WIP rather than the final version for FZ-GPU.
 * @version 0.4
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "port.hh"
#include "subr_legacy.cuhip.inl"

namespace psz {

template <typename T, int TileDim, int Seq>
__global__ void KERNEL_CUHIP_x_lorenzo_1d1l_FZGPU(
    FzgpuDeltaType* const in_delta, T* const out_data, dim3 const data_len3,
    dim3 const data_leap3, T const ebx2)
{
  namespace subr_v0 = psz::cuda_hip;
  constexpr auto NTHREAD = TileDim / Seq;

  __shared__ T scratch[TileDim];
  __shared__ Eq s_eq[TileDim];
  // compat for wave32 and 64
  __shared__ T exch_in[NTHREAD / 32];
  __shared__ T exch_out[NTHREAD / 32];

  T thp_data[Seq];

  auto id_base = blockIdx.x * TileDim;

  subr_v0::delta_only::load_1d<T, Eq, NTHREAD, Seq>(
      in_delta, data_len3.x, id_base, scratch, thp_data);
  subr_v0::block_scan_1d<T, Seq, NTHREAD>(
      thp_data, ebx2, exch_in, exch_out, scratch);
  subr_v0::write_1d<T, T, NTHREAD, Seq, true>(
      scratch, nullptr, data_len3.x, id_base, out_data, nullptr);
}

// 16x16 data block maps to 16x2 (one warp) thread block
template <typename T>
__global__ void KERNEL_CUHIP_x_lorenzo_2d1l_FZGPU(
    FzgpuDeltaType* const in_delta, T* const out_data, dim3 const data_len3,
    dim3 const data_leap3, T const ebx2)
{
  namespace subr_v0 = psz::cuda_hip;

  constexpr auto TileDim = 16;
  constexpr auto YSeq = TileDim / 2;
  static_assert(TileDim == 16, "In one case, we need TileDim for 2D == 16");

  __shared__ T scratch[TileDim];  // TODO use warp shuffle to eliminate this
  T thread_private[YSeq];

  auto gix = blockIdx.x * TileDim + threadIdx.x;
  auto giy_base =
      blockIdx.y * TileDim + threadIdx.y * YSeq;  // BDY * YSeq = TileDim == 16

  auto get_gid = [&](auto i) { return (giy_base + i) * data_leap3.y + gix; };

  subr_v0::delta_only::load_2d<T, Eq, YSeq>(
      in_delta, data_len3.x, gix, data_len3.y, giy_base, data_leap3.y,
      thread_private);
  subr_v0::block_scan_2d<T, Eq, Fp, YSeq>(thread_private, scratch, ebx2);
  subr_v0::decomp_write_2d<T, YSeq>(
      thread_private, data_len3.x, gix, data_len3.y, giy_base, data_leap3.y,
      out_data);
}

// 32x8x8 data block maps to 32x1x8 thread block
template <typename T>
__global__ void KERNEL_CUHIP_x_lorenzo_3d1l_delta_only(
    FzgpuDeltaType* const in_delta, T* const out_data, dim3 const data_len3,
    dim3 const data_leap3, T const ebx2)
{
  constexpr auto TileDim = 8;
  constexpr auto YSeq = TileDim;
  static_assert(TileDim == 8, "In one case, we need TileDim for 3D == 8");

  __shared__ T scratch[TileDim][4][8];
  T thread_private[YSeq];

  auto seg_id = threadIdx.x / 8;
  auto seg_tix = threadIdx.x % 8;

  auto gix = blockIdx.x * (4 * TileDim) + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim;
  auto giy = [&](auto y) { return giy_base + y; };
  auto giz = blockIdx.z * TileDim + threadIdx.z;
  auto gid = [&](auto y) {
    return giz * data_leap3.z + (giy_base + y) * data_leap3.y + gix;
  };

  auto load_3d = [&]() {
  // load to thread-private array (fuse at the same time)
#pragma unroll
    for (auto y = 0; y < YSeq; y++) {
      if (gix < data_len3.x and giy_base + y < data_len3.y and
          giz < data_len3.z)
        thread_private[y] = static_cast<T>(in_delta[gid(y)]);  // fuse
      else
        thread_private[y] = 0;
    }
  };

  auto block_scan_3d = [&]() {
    // partial-sum along y-axis, sequentially
    for (auto y = 1; y < YSeq; y++) thread_private[y] += thread_private[y - 1];

#pragma unroll
    for (auto i = 0; i < TileDim; i++) {
      // ND partial-sums along x- and z-axis
      // in-warp shuffle used: in order to perform, it's transposed after
      // X-partial sum
      T val = thread_private[i];

      for (auto dist = 1; dist < TileDim; dist *= 2) {
        auto addend = __shfl_up_sync(0xffffffff, val, dist, 8);
        if (seg_tix >= dist) val += addend;
      }

      // x-z transpose
      scratch[threadIdx.z][seg_id][seg_tix] = val;
      __syncthreads();
      val = scratch[seg_tix][seg_id][threadIdx.z];
      __syncthreads();

      for (auto dist = 1; dist < TileDim; dist *= 2) {
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
    for (auto y = 0; y < YSeq; y++)
      if (gix < data_len3.x and giy(y) < data_len3.y and giz < data_len3.z)
        out_data[gid(y)] = thread_private[y] * ebx2;
  };

  ////////////////////////////////////////////////////////////////////////////
  load_3d();
  block_scan_3d();
  decomp_write_3d();
}

}  // namespace psz
