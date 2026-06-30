#include "cusz/type.h"
#include "kernel.hh"
#include "kernel/launch.inl"
#include "mem/buf_comp.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"
#include "wave32.cu.inl"

#define TYPES_SETUP_KERN                \
  using T = typename Types::T;          \
  using Eq = typename Types::Eq;        \
  using ZigZag = psz::ZigZag<Eq>;       \
  using EqUInt = typename ZigZag::UInt; \
  using EqSInt = typename ZigZag::SInt;

namespace psz {

template <class Types, class Features, class Perf>
__global__ void KCU_x_lorenzo_1d(
    typename Types::T* const out_data, dim3 const extent, uint16_t const radius,
    typename Types::Fp const ebx2)
{
  TYPES_SETUP_KERN;

  constexpr auto TileDim = Perf::TileDim;
  constexpr auto Seq = Perf::Seq;
  constexpr auto NTHREAD = TileDim / Seq;  // equiv. to blockDim.x

  __shared__ T scratch[TileDim];  // for fused candidates
  __shared__ typename Types::Eq s_eq[TileDim];
  __shared__ T exch_in[NTHREAD / 32];
  __shared__ T exch_out[NTHREAD / 32];

  T thp_data[Seq];

  auto id_base = blockIdx.x * TileDim;

  auto load_fuse_1d = [&]() {
#pragma unroll
    for (auto i = 0; i < Seq; i++) {
      auto local_id = threadIdx.x + i * NTHREAD;
      auto id = id_base + local_id;
      if (id < extent.x) {
        if constexpr (Features::UseZigZag == Toggle::ZigZag_Off)
          scratch[local_id] = out_data[id] - radius;
        else
          scratch[local_id] = out_data[id];
      }
    }
    __syncthreads();

#pragma unroll
    for (auto i = 0; i < Seq; i++) thp_data[i] = scratch[threadIdx.x * Seq + i];
    __syncthreads();
  };

  auto block_scan_1d = [&]() {
    psz::SUBR_CUHIP_WAVE32_intrawarp_inclscan_1d<T, Seq>(thp_data);
    psz::SUBR_CUHIP_WAVE32_intrablock_exclscan_1d<T, Seq, NTHREAD>(thp_data, exch_in, exch_out);

    // put back to shmem
#pragma unroll
    for (auto i = 0; i < Seq; i++) scratch[threadIdx.x * Seq + i] = thp_data[i] * ebx2;
    __syncthreads();
  };

  auto write_1d = [&]() {
#pragma unroll
    for (auto i = 0; i < Seq; i++) {
      auto local_id = threadIdx.x + i * NTHREAD;
      auto id = id_base + local_id;
      if (id < extent.x) out_data[id] = scratch[local_id];
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

template <class Types, class Features, class Perf>
__global__ void KCU_x_lorenzo_2d__32x32(  //
    typename Types::T* const out_data, dim3 const extent, uint32_t leapy, uint16_t const radius, typename Types::Fp const ebx2)
{
  TYPES_SETUP_KERN;

  constexpr auto TileDim = Perf::TileDim;
  constexpr auto NumWarps = 4;
  constexpr auto YSEQ = TileDim / NumWarps;  // sequentiality in y direction

  static_assert(Perf::SeqY == YSEQ, "wrong SeqY");

  __shared__ T scratch[NumWarps - 1][TileDim + 1];
  T thp_data[YSEQ] = {0};

  auto gix = blockIdx.x * TileDim + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim + threadIdx.y * YSEQ;
  auto get_gid = [&](auto i) { return (giy_base + i) * leapy + gix; };

  auto load_fuse_2d = [&]() {
  // fuse outlier + eq
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      auto gid = get_gid(i);
      if (gix < extent.x and (giy_base + i) < extent.y) {
        if constexpr (Features::UseZigZag == Toggle::ZigZag_Off)
          thp_data[i] = out_data[gid] - radius;
        else
          thp_data[i] = out_data[gid];
      }
    }
  };

  auto block_scan_2d = [&]() {
    // partial-sum along y-axis, sequantially
    for (auto i = 1; i < YSEQ; i++) thp_data[i] += thp_data[i - 1];

    // 0, 1, 2
    if (threadIdx.y < NumWarps - 1) scratch[threadIdx.y][threadIdx.x] = thp_data[YSEQ - 1];
    __syncthreads();

    // cross-wrap scan

    if (threadIdx.y == 0) {
      T warp_accum[NumWarps - 1];  // 0, 1, 2
#pragma unroll
      for (auto i = 0; i < NumWarps - 1; i++) {  // load thp_data[YSEQ - 1] from each warp
        warp_accum[i] = scratch[i][threadIdx.x];
      }
#pragma unroll
      for (auto i = 1; i < NumWarps - 1; i++) {  // exclusive scan
        warp_accum[i] += warp_accum[i - 1];
      }
#pragma unroll
      for (auto i = 1; i < NumWarps - 1; i++) {  // determine the final addends
        scratch[i][threadIdx.x] = warp_accum[i];
      }
    }
    __syncthreads();

    if (threadIdx.y > 0) {
      auto addend = scratch[threadIdx.y - 1][threadIdx.x];
#pragma unroll
      for (auto i = 0; i < YSEQ; i++) thp_data[i] += addend;  // regression as pointer
    }
    __syncthreads();

    // then, in-warp partial-sum along x-axis
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      for (auto d = 1; d < TileDim; d *= 2) {
        T n = __shfl_up_sync(0xffffffff, thp_data[i], d, 32);  // full-warp shuffle
        if (threadIdx.x >= d) thp_data[i] += n;
      }
      thp_data[i] *= ebx2;  // scale accordingly
    }
  };

  auto decomp_write_2d = [&]() {
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      auto gid = get_gid(i);
      if (gix < extent.x and (giy_base + i) < extent.y) out_data[gid] = thp_data[i];
    }
  };

  /*-----------*/

  load_fuse_2d();
  block_scan_2d();
  decomp_write_2d();
}

// 32x8x8 data block maps to 32x1x8 thread block
template <class Types, class Features, class Perf>
__global__ void KCU_x_lorenzo_3d(  //
    typename Types::T* const out_data, dim3 const extent, uint32_t leapy, uint32_t leapz,
    uint16_t const radius, typename Types::Fp const ebx2)
{
  TYPES_SETUP_KERN;

  // TODO check SeqY
  constexpr auto TileDim = 8;
  constexpr auto YSEQ = TileDim;

  __shared__ T scratch[TileDim][4][8];
  T thread_private[YSEQ] = {0};

  auto seg_id = threadIdx.x / 8;
  auto seg_tix = threadIdx.x % 8;

  auto gix = blockIdx.x * (4 * TileDim) + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim;
  auto giy = [&](auto y) { return giy_base + y; };
  auto giz = blockIdx.z * TileDim + threadIdx.z;
  auto gid = [&](auto y) { return giz * leapz + (giy_base + y) * leapy + gix; };

  auto load_fuse_3d = [&]() {
  // load to thread-private array (fuse at the same time)
#pragma unroll
    for (auto y = 0; y < YSEQ; y++) {
      if (gix < extent.x and giy_base + y < extent.y and giz < extent.z) {
        if constexpr (Features::UseZigZag == Toggle::ZigZag_Off)
          thread_private[y] = out_data[gid(y)] - radius;
        else
          thread_private[y] = out_data[gid(y)];
      }
    }
  };

  auto block_scan_3d = [&]() {
    // partial-sum along y-axis, sequentially
    for (auto y = 1; y < YSEQ; y++) thread_private[y] += thread_private[y - 1];

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
    for (auto y = 0; y < YSEQ; y++)
      if (gix < extent.x and giy(y) < extent.y and giz < extent.z)
        out_data[gid(y)] = thread_private[y] * ebx2;
  };

  ////////////////////////////////////////////////////////////////////////////
  load_fuse_3d();
  block_scan_3d();
  decomp_write_3d();
}

}  // namespace psz

namespace psz::module {

template <class Types, class Features>
int GPU_x_lorenzo_nd<Types, Features>::kernel(
    typename Types::Buf_Comp* buf, typename Types::T* out, f8 const eb, u2 const radius,
    void* stream)
{
  auto extent = LEN_TO_DIM3(buf->eq_len3());
  auto d = psz::config::utils::ndim(extent);
  auto ebx2 = eb * 2;
  auto leapy = extent.x;
  auto leapz = extent.x * extent.y;

  if (d == 1) {
    using lrz1 = config::x_lorenzo<1>;
    psz::KCU_x_lorenzo_1d<Types, Features, lrz1::Perf>
        <<<lrz1::thread_grid(extent), lrz1::thread_block, 0, (cudaStream_t)stream>>>(
            out, extent, radius, (T)ebx2);
  }
  else if (d == 2) {
    using lrz2 = config::x_lorenzo<2, 32>;
    psz::KCU_x_lorenzo_2d__32x32<Types, Features, lrz2::Perf>
        <<<lrz2::thread_grid(extent), lrz2::thread_block, 0, (cudaStream_t)stream>>>(
            out, extent, leapy, radius, (T)ebx2);
  }
  else if (d == 3) {
    using lrz3 = config::x_lorenzo<3>;
    psz::KCU_x_lorenzo_3d<Types, Features, lrz3::Perf>
        <<<lrz3::thread_grid(extent), lrz3::thread_block, 0, (cudaStream_t)stream>>>(
            out, extent, leapy, leapz, radius, (T)ebx2);
  }
  else
    return PSZ_ABORT_UNSUPPORTED_DIMENSION;

  return CUSZ_SUCCESS;
}

}  // namespace psz::module
