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
    typename Types::Eq* const in_eq, typename Types::T* const in_outlier,
    typename Types::T* const out_data, dim3 const extent, uint16_t const radius,
    typename Types::Fp const ebx2, uint8_t const* incomp_flag = nullptr)
{
  TYPES_SETUP_KERN;

  constexpr auto TileDim = Perf::TileDim;
  constexpr auto Seq = Perf::Seq;
  constexpr auto NTHREAD = TileDim / Seq;  // equiv. to blockDim.x

  __shared__ T scratch[TileDim];  // for data and in_outlier
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
        if constexpr (Features::UseZigZag == 0b0)
          scratch[local_id] = in_outlier[id] - radius;
        else
          scratch[local_id] = in_outlier[id];
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

  // unpred-incomp blocks now ship the residual (delta+radius) like 2D/3D and
  // decode through the normal recurrence (in_outlier - radius -> inverse Lorenzo).
  (void)incomp_flag;

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
    typename Types::Eq* const in_eq, typename Types::T* const in_outlier, typename Types::T* const out_data, dim3 const extent, uint32_t leapy, uint16_t const radius, typename Types::Fp const ebx2)
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
  // the fused predict-input is laid out per-tile (tile == 1Ki chunk).
  auto tile_base =
      (size_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * (TileDim * TileDim);

  auto load_fuse_2d = [&]() {
    // fuse outlier and error-quant
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      auto gid = get_gid(i);
      if (gix < extent.x and (giy_base + i) < extent.y) {
        auto src = in_outlier[tile_base + (threadIdx.y * YSEQ + i) * TileDim + threadIdx.x];
        if constexpr (Features::UseZigZag == 0b0)
          thp_data[i] = src - radius;
        else
          thp_data[i] = src;
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
    typename Types::Eq* const in_eq, typename Types::T* const in_outlier, typename Types::T* const out_data,
    dim3 const extent, uint32_t leapy, uint32_t leapz,
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
  // the 32x8x8 CTA == one 2Ki per-tile slab; in-tile offset is
  // local_z*256 + local_y*32 + local_x (the encode's slab order, local_x fast).
  auto tile_base =
      (size_t)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * 2048u;

  auto load_fuse_3d = [&]() {
  // load to thread-private array (fuse at the same time)
#pragma unroll
    for (auto y = 0; y < YSEQ; y++) {
      if (gix < extent.x and giy_base + y < extent.y and giz < extent.z) {
        auto src = in_outlier[tile_base + threadIdx.z * 256u + y * 32u + threadIdx.x];
        if constexpr (Features::UseZigZag == 0b0)
          thread_private[y] = src - radius;
        else
          thread_private[y] = src;
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
  auto in_eq = buf->eq_d();
  // 2D/3D: the fused predict-input lives in the per-tile scratch (1D stays in-place linear).
  auto in_fused = buf->decode_fused_d();
  // per-block unpred-incomp message from the HF decoder (null for non-HFR / non-PBK paths).
  auto incomp_flag = buf->buf_hf() ? buf->buf_hf()->incomp_flag_d() : nullptr;

  if (d == 1) {
    using lrz1 = config::x_lorenzo<1>;
    psz::KCU_x_lorenzo_1d<Types, Features, lrz1::Perf>
        <<<lrz1::thread_grid(extent), lrz1::thread_block, 0, (cudaStream_t)stream>>>(
            in_eq, out, out, extent, radius, (T)ebx2, incomp_flag);
  }
  else if (d == 2) {
    using lrz2 = config::x_lorenzo<2, 32>;
    psz::KCU_x_lorenzo_2d__32x32<Types, Features, lrz2::Perf>
        <<<lrz2::thread_grid(extent), lrz2::thread_block, 0, (cudaStream_t)stream>>>(
            in_eq, in_fused, out, extent, leapy, radius, (T)ebx2);
  }
  else if (d == 3) {
    using lrz3 = config::x_lorenzo<3>;
    psz::KCU_x_lorenzo_3d<Types, Features, lrz3::Perf>
        <<<lrz3::thread_grid(extent), lrz3::thread_block, 0, (cudaStream_t)stream>>>(
            in_eq, in_fused, out, extent, leapy, leapz, radius, (T)ebx2);
  }
  else
    return PSZ_ABORT_UNSUPPORTED_DIMENSION;

  return CUSZ_SUCCESS;
}

}  // namespace psz::module
