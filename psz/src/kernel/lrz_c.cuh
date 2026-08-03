#include <cooperative_groups.h>

#include "kernel.hh"
#include "kernel/blk_fb.cuh"
#include "kernel/launch.inl"
#include "mem/buf_comp.hh"
#include "mem/cxx_sp_gpu.h"

namespace cg = cooperative_groups;

#define TYPES_SETUP_ZIGZAG()            \
  using ZigZag = psz::ZigZag<Eq>;       \
  using EqUInt = typename ZigZag::UInt; \
  using EqSInt = typename ZigZag::SInt; \
  using UInt = typename ZigZag::UInt;   \
  using SInt = typename ZigZag::SInt;

#define TYPES_SETUP_KERN                                      \
  using T = typename Types::T;                                \
  using Eq = typename Types::Eq;                              \
  TYPES_SETUP_ZIGZAG();                                       \
  constexpr auto UseZigZag = Features::UseZigZag == 0b1;      \
  constexpr auto UseH1L = (Features::UseH1GL & 0b01) == 0b01; \
  constexpr auto UseH1G = (Features::UseH1GL & 0b10) == 0b10; \
  constexpr auto UseIncompL = (Features::UnpredIncomp & 0b01) == 0b01;

#define TYPES_SETUP_MODULE              \
  using T = typename Types::T;          \
  using Eq = typename Types::Eq;        \
  using Buf = typename Types::Buf_Comp; \
  using Compact2 = _ptb::compact_GPU_DRAM2<T, u4>;

namespace psz {

__device__ __forceinline__ u4 linear_block_idx()
{ return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z); }

// padding eq (> eq_len)
// set_eq_padded: max(ALIGN_4Ki(linear), ntile*chunk); 1D: ALIGN_4Ki.
template <typename Eq>
__device__ __forceinline__ OutlierCell* blk_cells_tail(
    Eq* out_eq, size_t linear, size_t tiles_x_chunk)
{
  size_t aligned = (linear + 4095u) & ~size_t(4095);
  return (OutlierCell*)(out_eq +
                                       (aligned > tiles_x_chunk ? aligned : tiles_x_chunk));
}

template <bool UseZigZag, typename T, typename Eq>
__device__ __forceinline__ T
lrz_quantize_normal(T residual, u2 radius, bool quantizable, Eq& eq_out)
{
  TYPES_SETUP_ZIGZAG();

  T candidate;
  if constexpr (UseZigZag) {
    candidate = residual;
    eq_out = ZigZag::encode(static_cast<SInt>(quantizable * candidate));
  }
  else {
    candidate = residual + radius;
    eq_out = quantizable * static_cast<UInt>(candidate);
  }
  return candidate;
}

template <bool UseH1L, bool UseH1G, typename M>
__device__ __forceinline__ void hist1(u4 p_top1_count, u4* s_top1_counts, M* top_count)
{
  if constexpr (UseH1L) {
    auto rank = cg::this_thread_block().thread_rank();
    if (rank % 32 == 0) atomicAdd(s_top1_counts, p_top1_count);
    __syncthreads();
    if constexpr (UseH1G) {
      if (rank == 0) atomicAdd(top_count, s_top1_counts[0]);
    }
    else {
      if (rank == 0) top_count[linear_block_idx()] = s_top1_counts[0];
    }
  }
}

template <typename T>
__device__ __forceinline__ void count_local_stat(T delta, bool is_valid_range, u4& p_top1_count)
{
  int is_zero = is_valid_range ? (delta == 0) : 0;
  unsigned int mask = __ballot_sync(0xffffffff, is_zero);
  if (threadIdx.x % 32 == 0) p_top1_count += __popc(mask);
}

// TODO (241024) Fp=T can inherit all-fp8 path, lowering perf.
template <class Types, class Features, class Perf>
__global__ void KCU_c_lorenzo_1d(
    typename Types::T* const in_data, dim3 const extent, typename Types::Eq* const out_eq,
    uint16_t const radius, typename Types::Fp const ebx2_r, u4* out_bheader,
    typename Types::CVI* _compat_out_cvi = nullptr, typename Types::CN* _compat_out_cn = nullptr,
    size_t _compat_cn_max_allowed = 0, typename Types::M* top_count = nullptr)
{
  TYPES_SETUP_KERN;

  constexpr auto TileDim = Perf::TileDim;
  constexpr auto Seq = Perf::Seq;
  constexpr auto NumThreads = TileDim / Seq;

  __shared__ u4 s_top1_counts[1];
  if (threadIdx.x == 0) s_top1_counts[0] = 0;

  __shared__ T s_data[TileDim];
  __shared__ EqUInt s_eq_uint[TileDim];
  __shared__ u4 s_nout;
  __shared__ bool s_incomp;
  __shared__ OutlierCell s_cells[C::MaxNumUnpred];
  if (threadIdx.x == 0) s_nout = 0;

  T _thp_data[Seq + 1] = {0};
  auto prev = [&]() -> T& { return _thp_data[0]; };
  auto p_data = [&](auto i) -> T& { return _thp_data[i + 1]; };

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
  for (auto ix = 0; ix < Seq; ix++) p_data(ix) = s_data[threadIdx.x * Seq + ix];
  if (threadIdx.x > 0) prev() = s_data[threadIdx.x * Seq - 1];  // from last thread
  __syncthreads();

  u4 p_top1_count{0};

  // quantize & write back to shmem.eq
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    T delta = p_data(ix) - p_data(ix - 1);
    bool quantizable = fabs(delta) < radius;
    auto gid = id_base + threadIdx.x * Seq + ix;
    bool is_valid_range = gid < extent.x;

    if constexpr (UseH1L) count_local_stat(delta, is_valid_range, p_top1_count);

    Eq eq;
    T candidate = lrz_quantize_normal<UseZigZag>(delta, radius, quantizable, eq);
    s_eq_uint[threadIdx.x * Seq + ix] = eq;

    s_data[threadIdx.x * Seq + ix] = candidate;  // buffer dn-output

    if (not quantizable and is_valid_range) {
      if constexpr (Features::UnpredIncomp == 0b10)  // G1 L0
        fb_overflow_global(
            _compat_out_cvi, _compat_out_cn, _compat_cn_max_allowed, (f4)candidate, (u4)gid);
      else {
        auto local = atomicAdd(&s_nout, 1u);
        if constexpr (UseIncompL)
          if (local < C::MaxNumUnpred) s_cells[local] = {(f4)candidate, (u2)(gid - id_base)};
      }
    }
  }
  __syncthreads();

  // M10 unpred-incomp (pre)
  if (threadIdx.x == 0) {
    u4 nout = s_nout;
    s_incomp = UseIncompL and (nout > C::MaxNumUnpred);
    out_bheader[2u * blockIdx.x] =
        s_incomp ? ((u4)C::CodeIncompUnpred << EncIdShift) : (nout & (u4)C::MaxNumUnpred);
  }
  __syncthreads();

  // M10 unpred-incomp
  if constexpr (UseIncompL) {
    if (not s_incomp and threadIdx.x < s_nout and threadIdx.x < C::MaxNumUnpred) {
      auto out_cells = blk_cells_tail(out_eq, (size_t)extent.x, (size_t)gridDim.x * TileDim);
      out_cells[blockIdx.x * C::MaxNumUnpred + threadIdx.x] = s_cells[threadIdx.x];
    }
  }

  hist1<UseH1L, UseH1G>(p_top1_count, s_top1_counts, top_count);

  // M10 unpred-incomp
  if (s_incomp) {
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++) {
      auto local_id = threadIdx.x + ix * NumThreads;
      auto id = id_base + local_id;
      if (id < extent.x) out_eq[id] = psz::incomp_pack<Eq>((f4)s_data[local_id]);
    }
  }
  else {
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++) {
      auto local_id = threadIdx.x + ix * NumThreads;
      auto id = id_base + local_id;
      if (id < extent.x) out_eq[id] = (Eq)s_eq_uint[local_id];
    }
  }

  // end of kernel
}

template <class Types, class Features, class Perf>
__global__ void KCU_c_lorenzo_2d__32x32(
    typename Types::T* const in_data, dim3 const extent, u4 const leapy,
    typename Types::Eq* const out_eq, uint16_t const radius, typename Types::Fp const ebx2_r,
    u4* out_bheader, typename Types::CVI* _compat_out_cvi = nullptr,
    typename Types::CN* _compat_out_cn = nullptr, size_t _compat_cn_max_allowed = 0,
    typename Types::M* top_count = nullptr)
{
  TYPES_SETUP_KERN;

  constexpr auto TileDim = Perf::TileDim;
  constexpr auto Yseq = Perf::SeqY;
  constexpr auto NumWarps = 4;

  static_assert(NumWarps == TileDim * TileDim / Yseq / 32, "wrong TileDim");

  __shared__ u4 s_top1_counts[1];
  if (cg::this_thread_block().thread_rank() == 0) s_top1_counts[0] = 0;

  // Per-tile outliers are buffered locally.
  __shared__ u4 s_nout;
  if (cg::this_thread_block().thread_rank() == 0) s_nout = 0;
  __shared__ OutlierCell s_cells[C::MaxNumUnpred];

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

  u4 p_top1_count{0};

#pragma unroll
  for (auto i = Yseq; i > 0; i--) {
    // prediction
    center[i] -= center[i - 1];
    auto west = __shfl_up_sync(0xffffffff, center[i], 1, 32);
    if (threadIdx.x > 0) center[i] -= west;

    // quant-code
    auto gid = g_id(i - 1);
    bool quantizable = fabs(center[i]) < radius;
    bool is_valid_range = (gix < extent.x and (giy_base + i - 1) < extent.y);

    if constexpr (UseH1L) count_local_stat(center[i], is_valid_range, p_top1_count);

    Eq eq;
    T candidate = lrz_quantize_normal<UseZigZag>(center[i], radius, quantizable, eq);

    auto local_idx = (u2)((threadIdx.y * Yseq + (i - 1)) * TileDim + threadIdx.x);

    out_eq[linear_block_idx() * (TileDim * TileDim) + local_idx] =
        is_valid_range ? eq : (UseZigZag ? (Eq)0 : (Eq)radius);

    if (not quantizable and is_valid_range) {
      if constexpr (Features::UnpredIncomp == 0b10)  // G1 L0
        fb_overflow_global(
            _compat_out_cvi, _compat_out_cn, _compat_cn_max_allowed, (f4)candidate,
            (u4)(linear_block_idx() * (TileDim * TileDim) + local_idx));
      else {
        auto local = atomicAdd(&s_nout, 1u);
        if constexpr (UseIncompL)
          if (local < C::MaxNumUnpred) s_cells[local] = {(f4)candidate, local_idx};
      }
    }
  }
  __syncthreads();

  bool incomp = UseIncompL and (s_nout > C::MaxNumUnpred);
  if (cg::this_thread_block().thread_rank() == 0) {
    out_bheader[2u * linear_block_idx()] =
        incomp ? ((u4)C::CodeIncompUnpred << EncIdShift) : (s_nout & 0x7u);
  }

  // M10 unpred-incomp
  if constexpr (UseIncompL) {
    auto rank = cg::this_thread_block().thread_rank();
    if (not incomp and rank < s_nout and rank < C::MaxNumUnpred) {
      auto out_cells = blk_cells_tail(
          out_eq, (size_t)extent.x * extent.y,
          (size_t)gridDim.x * gridDim.y * (TileDim * TileDim));
      out_cells[linear_block_idx() * C::MaxNumUnpred + rank] = s_cells[rank];
    }
  }

  // M10 unpred-incomp
  if (incomp) {
#pragma unroll
    for (auto i = Yseq; i > 0; i--) {
      bool is_valid_range = (gix < extent.x and (giy_base + i - 1) < extent.y);
      T candidate = UseZigZag ? center[i] : center[i] + radius;
      auto local_idx = (u2)((threadIdx.y * Yseq + (i - 1)) * TileDim + threadIdx.x);
      out_eq[linear_block_idx() * (TileDim * TileDim) + local_idx] =
          psz::incomp_pack<Eq>(is_valid_range ? (f4)candidate : (f4)0);
    }
  }

  hist1<UseH1L, UseH1G>(p_top1_count, s_top1_counts, top_count);

  // end of kernel
}

template <class Types, class Features, class Perf>
__global__ void KCU_c_lorenzo_3d(
    typename Types::T* const in_data, dim3 const extent, u4 const leapy, u4 const leapz,
    typename Types::Eq* const out_eq, uint16_t const radius, typename Types::Fp const ebx2_r,
    u4* out_bheader, typename Types::CVI* _compat_out_cvi = nullptr,
    typename Types::CN* _compat_out_cn = nullptr, size_t _compat_cn_max_allowed = 0,
    typename Types::M* top_count = nullptr)
{
  TYPES_SETUP_KERN;

  constexpr auto TileDim = Perf::TileDim;
  // constexpr auto NumWarps = 8;

  __shared__ u4 s_top1_counts[1];
  if (cg::this_thread_block().thread_rank() == 0) s_top1_counts[0] = 0;

  using KC11 = psz::HFR_PBK_C11;  // M11, 2Ki
  __shared__ u4 s_nout;
  if (cg::this_thread_block().thread_rank() == 0) s_nout = 0;
  __shared__ OutlierCell s_cells[KC11::MaxNumUnpred];

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

  auto quantize_compact_write = [&](T delta, bool is_valid_range, auto gid, u2 local_idx) {
    bool quantizable = fabs(delta) < radius;

    Eq eq;
    T candidate = lrz_quantize_normal<UseZigZag>(delta, radius, quantizable, eq);

    out_eq[linear_block_idx() * 2048u + local_idx] =
        is_valid_range ? eq : (UseZigZag ? (Eq)0 : (Eq)radius);

    if (not quantizable and is_valid_range) {
      if constexpr (Features::UnpredIncomp == 0b10)  // G1 L0
        fb_overflow_global(
            _compat_out_cvi, _compat_out_cn, _compat_cn_max_allowed, (f4)candidate,
            (u4)(linear_block_idx() * 2048u + local_idx));
      else {
        // one 2Ki tile == one HF block (magnitude 11).
        u4 local = atomicAdd(&s_nout, 1u);
        if constexpr (UseIncompL)
          if (local < KC11::MaxNumUnpred) s_cells[local] = {(f4)candidate, local_idx};
      }
    }
  };

  ////////////////////////////////////////////////////////////////////////////

  load_prequant_3d();

  u4 p_top1_count{0};

  for (auto z = TileDim; z > 0; z--) {
    // z-direction
    delta[z] -= delta[z - 1];

    // x-direction
    auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
    if (threadIdx.x % TileDim > 0) delta[z] -= prev_x;

    // y-direction, exchange via shmem ghost padding along y
    s[threadIdx.y + 1][threadIdx.x] = delta[z];
    __syncthreads();

    // ty==0 must NOT read s[0][..]: it is never written; the prior `0*x` idiom NaN-leaked.
    if (threadIdx.y > 0) delta[z] -= s[threadIdx.y][threadIdx.x];

    bool is_valid_range = (gix < extent.x and giy < extent.y and giz(z - 1) < extent.z);
    if constexpr (UseH1L) count_local_stat(delta[z], is_valid_range, p_top1_count);

    // 2 1Ki-halves, z: 0..3 / 4..7).
    auto local_idx = (u2)((z - 1) * 256u + threadIdx.y * 32u + threadIdx.x);
    quantize_compact_write(delta[z], is_valid_range, gid(z - 1), local_idx);
    __syncthreads();
  }

  // M11 unpred-incomp (pre)
  constexpr auto EncIdShift11 = (u4)(KC11::BitsMaxNumUnpred + KC11::BitsMaxNumBreaks);
  bool incomp = UseIncompL and (s_nout > KC11::MaxNumUnpred);
  if (cg::this_thread_block().thread_rank() == 0) {
    out_bheader[2u * linear_block_idx()] =
        incomp ? ((u4)KC11::CodeIncompUnpred << EncIdShift11) : (s_nout & (u4)KC11::MaxNumUnpred);
  }

  // M11 unpred-incomp
  if constexpr (UseIncompL) {
    auto rank = cg::this_thread_block().thread_rank();
    if (not incomp and rank < s_nout and rank < KC11::MaxNumUnpred) {
      auto out_cells = blk_cells_tail(
          out_eq, (size_t)extent.x * extent.y * extent.z,
          (size_t)gridDim.x * gridDim.y * gridDim.z * 2048u);
      out_cells[linear_block_idx() * KC11::MaxNumUnpred + rank] = s_cells[rank];
    }
  }

  if (incomp) {
    for (auto z = TileDim; z > 0; z--) {
      bool is_valid_range = (gix < extent.x and giy < extent.y and giz(z - 1) < extent.z);
      T candidate = UseZigZag ? delta[z] : delta[z] + radius;
      auto local_idx = (u2)((z - 1) * 256u + threadIdx.y * 32u + threadIdx.x);
      out_eq[linear_block_idx() * 2048u + local_idx] =
          psz::incomp_pack<Eq>(is_valid_range ? (f4)candidate : (f4)0);
    }
  }

  hist1<UseH1L, UseH1G>(p_top1_count, s_top1_counts, top_count);
}

}  // namespace psz

namespace psz::module {

template <class Types, class Features>
int GPU_c_lorenzo_nd<Types, Features>::kernel(
    typename Types::Buf_Comp* buf, host::view<typename Types::T> in_data, f8 const eb,
    uint16_t const radius, bool enable_incomp, bool enable_global, void* stream)
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
  auto out_bheader = buf->buf_hf() ? (u4*)buf->buf_hf()->pbk_headers_d() : nullptr;

  auto go = [&](auto ui_const) {
    constexpr int UI = decltype(ui_const)::value;
    using F = psz::PredictorFeature<Features::UseZigZag, Features::UseH1GL, UI>;
    if (d == 1) {
      using lrz1 = config::c_lorenzo<1>;
      KCU_c_lorenzo_1d<Types, F, lrz1::Perf>
          <<<lrz1::thread_grid(extent), lrz1::thread_block, 0, (cudaStream_t)stream>>>(
              in_data.ptr, extent, out_eq, radius, (T)ebx2_r, out_bheader, ot->val_idx_d(),
              ot->num_d(), ot->max_allowed_num(), out_top1);
    }
    else if (d == 2) {
      using lrz2 = config::c_lorenzo<2, 32, 32>;
      KCU_c_lorenzo_2d__32x32<Types, F, lrz2::Perf>
          <<<lrz2::thread_grid(extent), lrz2 ::thread_block, 0, (cudaStream_t)stream>>>(
              in_data.ptr, extent, leapy, out_eq, radius, (T)ebx2_r, out_bheader, ot->val_idx_d(),
              ot->num_d(), ot->max_allowed_num(), out_top1);
    }
    else if (d == 3) {
      using lrz3 = config::c_lorenzo<3>;
      KCU_c_lorenzo_3d<Types, F, lrz3::Perf>
          <<<lrz3::thread_grid(extent), lrz3::thread_block, 0, (cudaStream_t)stream>>>(
              in_data.ptr, extent, leapy, leapz, out_eq, radius, (T)ebx2_r, out_bheader,
              ot->val_idx_d(), ot->num_d(), ot->max_allowed_num(), out_top1);
    }
  };
  if (d < 1 or d > 3) return PSZ_ABORT_UNSUPPORTED_DIMENSION;
  int const ui = (enable_global ? 0b10 : 0b00) | (enable_incomp ? 0b01 : 0b00);
  if (ui == 0b10)
    go(std::integral_constant<int, 0b10>{});
  else if (ui == 0b01)
    go(std::integral_constant<int, 0b01>{});
  else
    go(std::integral_constant<int, 0b00>{});

  return CUSZ_SUCCESS;
}

}  // namespace psz::module
