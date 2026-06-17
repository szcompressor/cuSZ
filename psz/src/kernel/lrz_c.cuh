#include <cooperative_groups.h>

#include "kernel.hh"
#include "kernel/launch.inl"
#include "mem/buf_comp.hh"
#include "mem/cxx_sp_gpu.h"

namespace cg = cooperative_groups;

#define COUNT_LOCAL_STAT(DELTA, IS_VALID_RANGE)           \
  int is_zero = IS_VALID_RANGE ? (DELTA == 0) : 0;        \
  unsigned int mask = __ballot_sync(0xffffffff, is_zero); \
  if (threadIdx.x % 32 == 0) thp_top1_count += __popc(mask);

#define TYPES_SETUP_KERN                \
  using T = typename Types::T;          \
  using Eq = typename Types::Eq;        \
  using ZigZag = psz::ZigZag<Eq>;       \
  using EqUInt = typename ZigZag::UInt; \
  using EqSInt = typename ZigZag::SInt;

#define TYPES_SETUP_MODULE              \
  using T = typename Types::T;          \
  using Eq = typename Types::Eq;        \
  using Buf = typename Types::Buf_Comp; \
  using Compact2 = _ptb::compact_GPU_DRAM2<T, u4>;

namespace psz {

__device__ __forceinline__ u4 linear_block_idx()
{
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}

// TODO (241024) the necessity to keep Fp=T, which triggered double type that
// significantly slowed down the kernel on non-HPC GPU
template <class Types, class Features, class Perf>
__global__ void KCU_c_lorenzo_1d(
    typename Types::T* const in_data, dim3 const extent, typename Types::Eq* const out_eq,
    typename Types::CompactValIdx* const out_cval_cidx, typename Types::CN* const out_cn,
    const size_t cn_max_allowed, uint16_t const radius, typename Types::Fp const ebx2_r,
    typename Types::M* top_count = nullptr)
{
  TYPES_SETUP_KERN;

  constexpr auto TileDim = Perf::TileDim;
  constexpr auto Seq = Perf::Seq;
  constexpr auto NumThreads = TileDim / Seq;

  __shared__ uint32_t s_top1_counts[1];
  if (threadIdx.x == 0) s_top1_counts[0] = 0;

  __shared__ T s_data[TileDim];
  __shared__ typename Types::EqUInt s_eq_uint[TileDim];

  T _thp_data[Seq + 1] = {0};
  auto prev = [&]() -> T& { return _thp_data[0]; };
  auto thp_data = [&](auto i) -> T& { return _thp_data[i + 1]; };

  auto const id_base = blockIdx.x * TileDim;

// dram.in_data to shmem.in_data
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < extent.x) s_data[threadIdx.x + ix * NumThreads] = round(in_data[id] * ebx2_r);
  }
  __syncthreads();

// shmem.in_data to private.in_data
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) thp_data(ix) = s_data[threadIdx.x * Seq + ix];
  if (threadIdx.x > 0) prev() = s_data[threadIdx.x * Seq - 1];  // from last thread
  __syncthreads();

  u4 thp_top1_count{0};

  // quantize & write back to shmem.eq
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    T delta = thp_data(ix) - thp_data(ix - 1);
    bool quantizable = fabs(delta) < radius;

    if constexpr (Features::UseH1L == Toggle::H1L_On) {
      bool is_valid_range = id_base + threadIdx.x * Seq + ix < extent.x;
      COUNT_LOCAL_STAT(delta, is_valid_range);
    }

    T candidate;
    if constexpr (Features::UseZigZag == Toggle::ZigZag_On) {
      candidate = delta;
      s_eq_uint[threadIdx.x * Seq + ix] =
          ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
    }
    else {
      candidate = delta + radius;
      s_eq_uint[threadIdx.x * Seq + ix] = quantizable * static_cast<EqUInt>(candidate);
    }

    if (not quantizable) {
      auto cur_idx = atomicAdd(out_cn, 1);
      if (cur_idx <= cn_max_allowed)
        out_cval_cidx[cur_idx] = {(float)candidate, id_base + threadIdx.x * Seq + ix};
    }
  }
  __syncthreads();

  if constexpr (Features::UseH1L == Toggle::H1L_On) {
    if (threadIdx.x % 32 == 0) atomicAdd(s_top1_counts, thp_top1_count);
    __syncthreads();

    if constexpr (Features::UseH1G == Toggle::H1G_On) {
      if (threadIdx.x == 0) atomicAdd(top_count, s_top1_counts[0]);
    }
    else {
      if (threadIdx.x == 0) top_count[linear_block_idx()] = s_top1_counts[0];
    }
  }

// write from shmem.eq to dram.eq
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < extent.x) out_eq[id] = s_eq_uint[threadIdx.x + ix * NumThreads];
  }

  // end of kernel
}

template <typename T, bool UseZigZag, typename Eq = uint16_t, typename Fp = T>
__global__ [[deprecated]] void KCU_c_lorenzo_2d1l(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3, Eq* const out_eq,
    T* const out_cval, uint32_t* const out_cidx, uint32_t* const out_cn, uint16_t const radius,
    Fp const ebx2_r)
{
  using ZigZag = psz::ZigZag<Eq>;
  using EqUInt = typename ZigZag::UInt;
  using EqSInt = typename ZigZag::SInt;

  constexpr auto TileDim = 16;
  constexpr auto Yseq = 8;

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

      if constexpr (UseZigZag) {
        candidate = center[i];
        out_eq[gid] = ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
      }
      else {
        candidate = center[i] + radius;
        out_eq[gid] = quantizable * (EqUInt)candidate;
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

template <class Types, class Features, class Perf>
__global__ void KCU_c_lorenzo_2d__32x32(
    typename Types::T* const in_data, dim3 const extent, uint32_t const leapy,
    typename Types::Eq* const out_eq, typename Types::CompactValIdx* const out_cval_cidx,
    typename Types::CN* const out_cn, const size_t cn_max_allowed, uint16_t const radius,
    typename Types::Fp const ebx2_r, typename Types::M* top_count = nullptr)
{
  TYPES_SETUP_KERN;

  constexpr auto TileDim = Perf::TileDim;
  constexpr auto Yseq = Perf::SeqY;
  constexpr auto NumWarps = 4;
  static_assert(NumWarps == TileDim * TileDim / Yseq / 32, "wrong TileDim");

  __shared__ uint32_t s_top1_counts[1];
  if (cg::this_thread_block().thread_rank() == 0) s_top1_counts[0] = 0;

  __shared__ T exchange[NumWarps - 1][TileDim + 1];

  T center[Yseq + 1] = {0};

  // BDX == TileDim == 32 (a full warp), BDY * Yseq = TileDim == 32
  auto gix = blockIdx.x * TileDim + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim + threadIdx.y * Yseq;
  auto g_id = [&](auto i) { return (giy_base + i) * leapy + gix; };

// read to private.in_data (center)
#pragma unroll
  for (auto iy = 0; iy < Yseq; iy++) {
    if (gix < extent.x and giy_base + iy < extent.y)
      center[iy + 1] = round(in_data[g_id(iy)] * ebx2_r);
  }
  if (threadIdx.y < NumWarps - 1) exchange[threadIdx.y][threadIdx.x] = center[Yseq];
  __syncthreads();
  if (threadIdx.y > 0) center[0] = exchange[threadIdx.y - 1][threadIdx.x];
  __syncthreads();

  u4 thp_top1_count{0};

#pragma unroll
  for (auto i = Yseq; i > 0; i--) {
    // 1) prediction (apply Lorenzo filter)
    center[i] -= center[i - 1];
    auto west = __shfl_up_sync(0xffffffff, center[i], 1, 32);
    if (threadIdx.x > 0) center[i] -= west;

    // 2) store quant-code
    auto gid = g_id(i - 1);

    bool quantizable = fabs(center[i]) < radius;
    bool is_valid_range = (gix < extent.x and (giy_base + i - 1) < extent.y);

    if constexpr (Features::UseH1L == Toggle::H1L_On) {
      COUNT_LOCAL_STAT(center[i], is_valid_range);
    }

    T candidate;

    if constexpr (Features::UseZigZag == Toggle::ZigZag_On) {
      candidate = center[i];
      if (is_valid_range)
        out_eq[gid] = ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
    }
    else {
      candidate = center[i] + radius;
      if (is_valid_range) out_eq[gid] = quantizable * static_cast<EqUInt>(candidate);
    }

    if (not quantizable) {
      if (gix < extent.x and (giy_base + i - 1) < extent.y) {
        auto cur_idx = atomicAdd(out_cn, 1);
        if (cur_idx <= cn_max_allowed) out_cval_cidx[cur_idx] = {(float)candidate, gid};
      }
    }
  }

  if constexpr (Features::UseH1L == Toggle::H1L_On) {
    if (cg::this_thread_block().thread_rank() % 32 == 0) atomicAdd(s_top1_counts, thp_top1_count);
    __syncthreads();

    if constexpr (Features::UseH1G == Toggle::H1G_On) {
      if (cg::this_thread_block().thread_rank() == 0) atomicAdd(top_count, s_top1_counts[0]);
    }
    else {
      if (cg::this_thread_block().thread_rank() == 0)
        top_count[linear_block_idx()] = s_top1_counts[0];
    }
  }

  // end of kernel
}

template <class Types, class Features, class Perf>
__global__ void KCU_c_lorenzo_3d(
    typename Types::T* const in_data, dim3 const extent, uint32_t const leapy,
    uint32_t const leapz, typename Types::Eq* const out_eq,
    typename Types::CompactValIdx* const out_cval_cidx, typename Types::CN* const out_cn,
    const size_t cn_max_allowed, uint16_t const radius, typename Types::Fp const ebx2_r,
    typename Types::M* top_count = nullptr)
{
  TYPES_SETUP_KERN;

  constexpr auto TileDim = Perf::TileDim;
  // constexpr auto NumWarps = 8;

  __shared__ uint32_t s_top1_counts[1];
  if (cg::this_thread_block().thread_rank() == 0) s_top1_counts[0] = 0;

  __shared__ T s[9][33];

  T delta[TileDim + 1] = {0};  // first el = 0

  const auto gix = blockIdx.x * (TileDim * 4) + threadIdx.x;
  const auto giy = blockIdx.y * TileDim + threadIdx.y;
  const auto giz_base = blockIdx.z * TileDim;
  const auto base_id = gix + giy * leapy + giz_base * leapz;

  auto giz = [&](auto z) { return giz_base + z; };
  auto gid = [&](auto z) { return base_id + z * leapz; };

  auto load_prequant_3d = [&]() {
    if (gix < extent.x and giy < extent.y) {
      for (auto z = 0; z < TileDim; z++)
        if (giz(z) < extent.z)
          delta[z + 1] = round(in_data[gid(z)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_compact_write = [&](T delta, auto x, auto y, auto z, auto gid) {
    bool quantizable = fabs(delta) < radius;

    if (x < extent.x and y < extent.y and z < extent.z) {
      T candidate;

      if constexpr (Features::UseZigZag == Toggle::ZigZag_On) {
        candidate = delta;
        out_eq[gid] = Types::ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
      }
      else {
        candidate = delta + radius;
        out_eq[gid] = quantizable * static_cast<EqUInt>(candidate);
      }

      if (not quantizable) {
        auto cur_idx = atomicAdd(out_cn, 1);
        if (cur_idx <= cn_max_allowed) out_cval_cidx[cur_idx] = {(float)candidate, gid};
      }
    }
  };

  ////////////////////////////////////////////////////////////////////////////

  load_prequant_3d();

  u4 thp_top1_count{0};

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

    // ty==0 must NOT read s[0][..]: it is never written; the prior `0*x` idiom NaN-leaked.
    if (threadIdx.y > 0) delta[z] -= s[threadIdx.y][threadIdx.x];

    if constexpr (Features::UseH1L == Toggle::H1L_On) {
      auto is_valid_range = (gix < extent.x and giy < extent.y and giz(z - 1) < extent.z);
      COUNT_LOCAL_STAT(delta[z], is_valid_range);
    }

    // now delta[z] is delta
    quantize_compact_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    __syncthreads();
  }

  if constexpr (Features::UseH1L == Toggle::H1L_On) {
    if (cg::this_thread_block().thread_rank() % 32 == 0) atomicAdd(s_top1_counts, thp_top1_count);
    __syncthreads();

    if constexpr (Features::UseH1G == Toggle::H1G_On) {
      if (cg::this_thread_block().thread_rank() == 0) atomicAdd(top_count, s_top1_counts[0]);
    }
    else {
      if (cg::this_thread_block().thread_rank() == 0)
        top_count[linear_block_idx()] = s_top1_counts[0];
    }
  }
}

}  // namespace psz

namespace psz::module {

template <class Types, class Features>
int GPU_c_lorenzo_nd<Types, Features>::kernel(
    typename Types::Buf_Comp* buf, host::view<typename Types::T> in_data, f8 const eb,
    uint16_t const radius, void* stream)
{
  using Compact2 = _ptb::compact_GPU_DRAM2<T, u4>;
  auto extent = LEN_TO_DIM3(in_data.extent);
  auto d = psz::config::utils::ndim(extent);
  auto ebx2_r = 1 / (eb * 2);
  auto leapy = extent.x;
  auto leapz = extent.x * extent.y;
  auto ot = (Compact2*)buf->buf_outlier2();
  auto out_eq = buf->eq_d();
  auto out_top1 = buf->top1_d();

  if (d == 1) {
    using lrz1 = config::c_lorenzo<1>;
    KCU_c_lorenzo_1d<Types, Features, lrz1::Perf>
        <<<lrz1::thread_grid(extent), lrz1::thread_block, 0, (cudaStream_t)stream>>>(
            in_data.ptr, extent, out_eq, ot->val_idx_d(), ot->num_d(), ot->max_allowed_num(),
            radius, (T)ebx2_r, out_top1);
  }
  else if (d == 2) {
    using lrz2 = config::c_lorenzo<2, 32, 32>;
    KCU_c_lorenzo_2d__32x32<Types, Features, lrz2::Perf>
        <<<lrz2::thread_grid(extent), lrz2 ::thread_block, 0, (cudaStream_t)stream>>>(
            in_data.ptr, extent, leapy, out_eq, ot->val_idx_d(), ot->num_d(),
            ot->max_allowed_num(), radius, (T)ebx2_r, out_top1);
  }
  else if (d == 3) {
    using lrz3 = config::c_lorenzo<3>;
    KCU_c_lorenzo_3d<Types, Features, lrz3::Perf>
        <<<lrz3::thread_grid(extent), lrz3::thread_block, 0, (cudaStream_t)stream>>>(
            in_data.ptr, extent, leapy, leapz, out_eq, ot->val_idx_d(), ot->num_d(),
            ot->max_allowed_num(), radius, (T)ebx2_r, out_top1);
  }
  else
    return PSZ_ABORT_UNSUPPORTED_DIMENSION;

  return CUSZ_SUCCESS;
}

}  // namespace psz::module
