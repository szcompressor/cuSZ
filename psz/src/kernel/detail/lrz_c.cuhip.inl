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

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define COUNT_LOCAL_STAT(DELTA, IS_VALID_RANGE)           \
  int is_zero = IS_VALID_RANGE ? (DELTA == 0) : 0;        \
  unsigned int mask = __ballot_sync(0xffffffff, is_zero); \
  if (threadIdx.x % 32 == 0) thp_top1_count += __popc(mask);

#include "detail/composite.hh"
#include "kernel/launch.hh"
#include "kernel/predictor.hh"
#include "mem/cxx_sp_gpu.h"

#define Z(LEN3) LEN3[2]
#define Y(LEN3) LEN3[1]
#define X(LEN3) LEN3[0]
#define TO_DIM3(LEN3) dim3(X(LEN3), Y(LEN3), Z(LEN3))

namespace psz {

// TODO (241024) the necessity to keep Fp=T, which triggered double type that
// significantly slowed down the kernel on non-HPC GPU
template <typename T, class PC, class Perf>
__global__ void KERNEL_CUHIP_c_lorenzo_1d(
    T* const in_data, size_t const data_len, typename PC::Eq* const out_eq,
    typename PC::C2VI* const out_cval_cidx, typename PC::CN* const out_cn,
    const size_t cn_max_allowed, uint16_t const radius, typename PC::Fp const ebx2_r,
    typename PC::M* top_count = nullptr)
{
  constexpr auto TileDim = Perf::TileDim;
  constexpr auto Seq = Perf::Seq;
  constexpr auto NumThreads = TileDim / Seq;

  __shared__ uint32_t s_top1_counts[1];
  if (threadIdx.x == 0) s_top1_counts[0] = 0;

  __shared__ T s_data[TileDim];
  __shared__ typename PC::EqUInt s_eq_uint[TileDim];

  T _thp_data[Seq + 1] = {0};
  auto prev = [&]() -> T& { return _thp_data[0]; };
  auto thp_data = [&](auto i) -> T& { return _thp_data[i + 1]; };

  auto const id_base = blockIdx.x * TileDim;

// dram.in_data to shmem.in_data
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < data_len) s_data[threadIdx.x + ix * NumThreads] = round(in_data[id] * ebx2_r);
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

    if constexpr (PC::UseStatLocal == Toggle::StatLocalEnabled) {
      bool is_valid_range = id_base + threadIdx.x * Seq + ix < data_len;
      COUNT_LOCAL_STAT(delta, is_valid_range);
    }

    T candidate;
    if constexpr (PC::UseZigZag == Toggle::ZigZagEnabled) {
      candidate = delta;
      s_eq_uint[threadIdx.x * Seq + ix] =
          PC::ZigZag::encode(static_cast<typename PC::EqSInt>(quantizable * candidate));
    }
    else {
      candidate = delta + radius;
      s_eq_uint[threadIdx.x * Seq + ix] =
          quantizable * static_cast<typename PC::EqUInt>(candidate);
    }

    if (not quantizable) {
      auto cur_idx = atomicAdd(out_cn, 1);
      if (cur_idx <= cn_max_allowed)
        out_cval_cidx[cur_idx] = {(float)candidate, id_base + threadIdx.x * Seq + ix};
    }
  }
  __syncthreads();

  if constexpr (PC::UseStatLocal == Toggle::StatLocalEnabled) {
    if (threadIdx.x % 32 == 0) atomicAdd(s_top1_counts, thp_top1_count);
    __syncthreads();

    if constexpr (PC::UseGlobalStat == Toggle::StatGlobalEnabled)
      if (threadIdx.x == 0) atomicAdd(top_count, s_top1_counts[0]);
  }

// write from shmem.eq to dram.eq
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < data_len) out_eq[id] = s_eq_uint[threadIdx.x + ix * NumThreads];
  }

  // end of kernel
}

template <typename T, bool UseZigZag, typename Eq = uint16_t, typename Fp = T>
__global__ [[deprecated]] void KERNEL_CUHIP_c_lorenzo_2d1l(
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

template <typename T, class PC, class Perf>
__global__ void KERNEL_CUHIP_c_lorenzo_2d__32x32(
    T* const in_data, uint32_t const data_lenx, uint32_t const data_leny,
    uint32_t const data_leapy, typename PC::Eq* const out_eq,
    typename PC::C2VI* const out_cval_cidx, typename PC::CN* const out_cn,
    const size_t cn_max_allowed, uint16_t const radius, typename PC::Fp const ebx2_r,
    typename PC::M* top_count = nullptr)
{
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
  auto g_id = [&](auto i) { return (giy_base + i) * data_leapy + gix; };

// read to private.in_data (center)
#pragma unroll
  for (auto iy = 0; iy < Yseq; iy++) {
    if (gix < data_lenx and giy_base + iy < data_leny)
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
    bool is_valid_range = (gix < data_lenx and (giy_base + i - 1) < data_leny);

    if constexpr (PC::UseStatLocal == Toggle::StatLocalEnabled) {
      COUNT_LOCAL_STAT(center[i], is_valid_range);
    }

    T candidate;

    if constexpr (PC::UseZigZag == Toggle::ZigZagEnabled) {
      candidate = center[i];
      if (is_valid_range)
        out_eq[gid] =
            PC::ZigZag::encode(static_cast<typename PC::EqSInt>(quantizable * candidate));
    }
    else {
      candidate = center[i] + radius;
      if (is_valid_range) out_eq[gid] = quantizable * static_cast<typename PC::EqUInt>(candidate);
    }

    if (not quantizable) {
      if (gix < data_lenx and (giy_base + i - 1) < data_leny) {
        auto cur_idx = atomicAdd(out_cn, 1);
        if (cur_idx <= cn_max_allowed) out_cval_cidx[cur_idx] = {(float)candidate, gid};
      }
    }
  }

  if constexpr (PC::UseStatLocal == Toggle::StatLocalEnabled) {
    if (cg::this_thread_block().thread_rank() % 32 == 0) atomicAdd(s_top1_counts, thp_top1_count);
    __syncthreads();

    if constexpr (PC::UseGlobalStat == Toggle::StatGlobalEnabled) {
      if (cg::this_thread_block().thread_rank() == 0) atomicAdd(top_count, s_top1_counts[0]);
    }
  }

  // end of kernel
}

template <typename T, class PC, class Perf>
__global__ void KERNEL_CUHIP_c_lorenzo_3d(
    T* const in_data, uint32_t const data_lenx, uint32_t const data_leny,
    uint32_t const data_leapy, uint32_t const data_lenz, uint32_t const data_leapz,
    typename PC::Eq* const out_eq, typename PC::C2VI* const out_cval_cidx,
    typename PC::CN* const out_cn, const size_t cn_max_allowed, uint16_t const radius,
    typename PC::Fp const ebx2_r, typename PC::M* top_count = nullptr)
{
  constexpr auto TileDim = Perf::TileDim;
  // constexpr auto NumWarps = 8;

  __shared__ uint32_t s_top1_counts[1];
  if (cg::this_thread_block().thread_rank() == 0) s_top1_counts[0] = 0;

  __shared__ T s[9][33];

  T delta[TileDim + 1] = {0};  // first el = 0

  const auto gix = blockIdx.x * (TileDim * 4) + threadIdx.x;
  const auto giy = blockIdx.y * TileDim + threadIdx.y;
  const auto giz_base = blockIdx.z * TileDim;
  const auto base_id = gix + giy * data_leapy + giz_base * data_leapz;

  auto giz = [&](auto z) { return giz_base + z; };
  auto gid = [&](auto z) { return base_id + z * data_leapz; };

  auto load_prequant_3d = [&]() {
    if (gix < data_lenx and giy < data_leny) {
      for (auto z = 0; z < TileDim; z++)
        if (giz(z) < data_lenz)
          delta[z + 1] = round(in_data[gid(z)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_compact_write = [&](T delta, auto x, auto y, auto z, auto gid) {
    bool quantizable = fabs(delta) < radius;

    if (x < data_lenx and y < data_leny and z < data_lenz) {
      T candidate;

      if constexpr (PC::UseZigZag == Toggle::ZigZagEnabled) {
        candidate = delta;
        out_eq[gid] =
            PC::ZigZag::encode(static_cast<typename PC::EqSInt>(quantizable * candidate));
      }
      else {
        candidate = delta + radius;
        out_eq[gid] = quantizable * static_cast<typename PC::EqUInt>(candidate);
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

    delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

    if constexpr (PC::UseStatLocal == Toggle::StatLocalEnabled) {
      auto is_valid_range = (gix < data_lenx and giy < data_leny and giz(z - 1) < data_lenz);
      COUNT_LOCAL_STAT(delta[z], is_valid_range);
    }

    // now delta[z] is delta
    quantize_compact_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    __syncthreads();
  }

  if constexpr (PC::UseStatLocal == Toggle::StatLocalEnabled) {
    if (cg::this_thread_block().thread_rank() % 32 == 0) atomicAdd(s_top1_counts, thp_top1_count);
    __syncthreads();

    if constexpr (PC::UseGlobalStat == Toggle::StatGlobalEnabled) {
      if (cg::this_thread_block().thread_rank() == 0) atomicAdd(top_count, s_top1_counts[0]);
    }
  }
}

template <
    typename TIN, typename TOUT, bool ReverseProcess = false, typename Fp = TIN, int TileDim = 256,
    int Seq = 8>
__global__ [[deprecated]] void KERNEL_CUHIP_lorenzo_prequant(
    TIN* const in, size_t const in_len, Fp const ebx2_r, Fp const ebx2, TOUT* const out)
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

namespace psz::module {

template <typename TIN, typename TOUT, bool ReverseProcess>
[[deprecated]] int GPU_lorenzo_prequant(
    TIN* const in, size_t const len, f8 const ebx2, f8 const ebx2_r, TOUT* const out, void* stream)
{
  using namespace psz::config;

  psz::KERNEL_CUHIP_lorenzo_prequant<
      TIN, TOUT, ReverseProcess, TIN, c_lorenzo<1>::tile.x, c_lorenzo<1>::sequentiality.x>
      <<<c_lorenzo<1>::thread_grid(dim3(len)), c_lorenzo<1>::thread_block, 0,
         (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(in, len, ebx2_r, ebx2, out);

  return CUSZ_SUCCESS;
}

template <typename T, class PC>
struct GPU_c_lorenzo_1d {
  static int kernel(
      T* const in_data, stdlen3 const data_len3, typename PC::Eq* const out_eq, void* out_outlier,
      u4* out_top1, f8 const ebx2_r, uint16_t const radius, void* stream)
  {
    using Compact2 = _portable::compact_GPU_DRAM2<T, u4>;
    auto ot = (Compact2*)out_outlier;
    using lrz1 = config::c_lorenzo<1>;

    psz::KERNEL_CUHIP_c_lorenzo_1d<T, PC, lrz1::Perf>
        <<<lrz1::thread_grid(dim3(data_len3[0], 1, 1)), lrz1::thread_block, 0,
           (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(
            in_data, data_len3[0], out_eq, ot->val_idx_d(), ot->num_d(), ot->max_allowed_num(),
            radius, (T)ebx2_r, out_top1);

    return CUSZ_SUCCESS;
  }
};

template <typename T, class PC>
struct GPU_c_lorenzo_2d {
  static int kernel(
      T* const in_data, stdlen3 const _data_len3, typename PC::Eq* const out_eq, void* out_outlier,
      u4* out_top1, f8 const ebx2_r, uint16_t const radius, void* stream)
  {
    using Compact2 = _portable::compact_GPU_DRAM2<T, u4>;
    auto ot = (Compact2*)out_outlier;
    using lrz2 = config::c_lorenzo<2, 32, 32>;

    auto data_len3 = TO_DIM3(_data_len3);
    auto leap3 = dim3(1, data_len3.x, data_len3.x * data_len3.y);

    psz::KERNEL_CUHIP_c_lorenzo_2d__32x32<T, PC, lrz2::Perf>
        <<<lrz2::thread_grid(data_len3), lrz2 ::thread_block, 0,
           (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(
            in_data, data_len3.x, data_len3.y, leap3.y, out_eq, ot->val_idx_d(), ot->num_d(),
            ot->max_allowed_num(), radius, (T)ebx2_r, out_top1);

    return CUSZ_SUCCESS;
  }
};

template <typename T, class PC>
struct GPU_c_lorenzo_3d {
  static int kernel(
      T* const in_data, stdlen3 const _data_len3, typename PC::Eq* const out_eq, void* out_outlier,
      u4* out_top1, f8 const ebx2_r, uint16_t const radius, void* stream)
  {
    using Compact2 = _portable::compact_GPU_DRAM2<T, u4>;
    auto ot = (Compact2*)out_outlier;

    using lrz3 = config::c_lorenzo<3>;

    auto data_len3 = TO_DIM3(_data_len3);
    auto leap3 = dim3(1, data_len3.x, data_len3.x * data_len3.y);

    psz::KERNEL_CUHIP_c_lorenzo_3d<T, PC, lrz3::Perf>
        <<<lrz3::thread_grid(data_len3), lrz3::thread_block, 0,
           (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(
            in_data, data_len3.x, data_len3.y, leap3.y, data_len3.z, leap3.z, out_eq,
            ot->val_idx_d(), ot->num_d(), ot->max_allowed_num(), radius, (T)ebx2_r, out_top1);

    return CUSZ_SUCCESS;
  }
};

template <typename T, class PC, class Buf>
int GPU_c_lorenzo_nd<T, PC, Buf>::kernel(
    T* const in_data, stdlen3 const _data_len3, typename PC::Eq* const out_eq, void* out_outlier,
    u4* out_top1, f8 const eb, uint16_t const radius, void* stream)
{
  auto data_len3 = TO_DIM3(_data_len3);
  auto d = psz::config::utils::ndim(data_len3);

  auto eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  if (d == 1)
    GPU_c_lorenzo_1d<T, PC>::kernel(
        in_data, _data_len3, out_eq, out_outlier, out_top1, ebx2_r, radius, stream);
  else if (d == 2)
    GPU_c_lorenzo_2d<T, PC>::kernel(
        in_data, _data_len3, out_eq, out_outlier, out_top1, ebx2_r, radius, stream);
  else if (d == 3)
    GPU_c_lorenzo_3d<T, PC>::kernel(
        in_data, _data_len3, out_eq, out_outlier, out_top1, ebx2_r, radius, stream);
  else
    return PSZ_ABORT_UNSUPPORTED_DIMENSION;

  return CUSZ_SUCCESS;
}

template <typename T, class PC, class Buf>
int GPU_c_lorenzo_nd<T, PC, Buf>::compressor_kernel(
    Buf* buf, T* const in_data, stdlen3 const _data_len3, f8 const eb, uint16_t const radius,
    void* stream)
{
  auto data_len3 = TO_DIM3(_data_len3);
  auto d = psz::config::utils::ndim(data_len3);

  auto eb_r = 1 / eb, ebx2 = eb * 2, ebx2_r = 1 / ebx2;

  if (d == 1)
    GPU_c_lorenzo_1d<T, PC>::kernel(
        in_data, _data_len3, buf->eq_d(), buf->buf_outlier2(), buf->top1_d(), ebx2_r, radius,
        stream);
  else if (d == 2)
    GPU_c_lorenzo_2d<T, PC>::kernel(
        in_data, _data_len3, buf->eq_d(), buf->buf_outlier2(), buf->top1_d(), ebx2_r, radius,
        stream);
  else if (d == 3)
    GPU_c_lorenzo_3d<T, PC>::kernel(
        in_data, _data_len3, buf->eq_d(), buf->buf_outlier2(), buf->top1_d(), ebx2_r, radius,
        stream);
  else
    return PSZ_ABORT_UNSUPPORTED_DIMENSION;

  return CUSZ_SUCCESS;
}

}  // namespace psz::module
