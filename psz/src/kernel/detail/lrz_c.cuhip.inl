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

#define CHECK_LOCAL_STAT_TOGGLE                                 \
  static_assert(                                                \
      not((UseLocalStat == false) and (UseGlobalStat == true)), \
      "UseLocalStat must be true when UseGlobalStat is set to true.");

#define COUNT_LOCAL_STAT(DELTA, IS_VALID_RANGE)           \
  int is_zero = IS_VALID_RANGE ? (DELTA == 0) : 0;        \
  unsigned int mask = __ballot_sync(0xffffffff, is_zero); \
  if (threadIdx.x % 32 == 0) thp_top1_count += __popc(mask);

#include <cstdint>
#include <type_traits>

#include "cusz/suint.hh"
#include "cusz/type.h"
#include "hfword.hh"
#include "kernel/predictor.hh"
#include "mem/cxx_sp_gpu.h"
#include "utils/err.hh"
#include "utils/timer.hh"

#define SETUP_ZIGZAG                    \
  using ZigZag = psz::ZigZag<Eq>;       \
  using EqUInt = typename ZigZag::UInt; \
  using EqSInt = typename ZigZag::SInt;

#define Z(LEN3) LEN3[2]
#define Y(LEN3) LEN3[1]
#define X(LEN3) LEN3[0]
#define TO_DIM3(LEN3) dim3(X(LEN3), Y(LEN3), Z(LEN3))

using CompactIdx = uint32_t;
using CompactNum = uint32_t;
#define CompactVal T
#define CV CompactVal
#define CI CompactIdx
#define CN CompactNum

using Hf = u4;
using W = HuffmanWord<sizeof(Hf)>;

namespace psz {

__device__ __constant__ float probs_lookup[] = {
    0.9986978877668036, 0.8958236229793216,  0.7240624144538735,  0.5539072788884121,
    0.4002056851980645, 0.2829302218900982,  0.21690651474233288, 0.17925002959630637,
    0.1556048834628191, 0.13656879865036847, 0.1197345152387365};

__device__ void find_closest(float prob, volatile u4* closest_index) {}

// TODO (241024) the necessity to keep Fp=T, which triggered double type that
// significantly slowed down the kernel on non-HPC GPU
template <
    typename T, int TileDim, int Seq, bool UseZigZag, typename Eq = uint16_t, typename Fp = T,
    bool UseLocalStat = true, bool UseGlobalStat = true, bool EncodeInPlace = false,
    bool EIP_DBG = true>
__global__ void KERNEL_CUHIP_c_lorenzo_1d1l(
    T* const in_data, size_t const gx,                         // input
    Eq* const out_eq,                                          // output: dense
    CV* const out_cval, CI* const out_cidx, CN* const out_cn,  // output: sparse
    uint16_t const radius, Fp const ebx2_r,                    // config
    uint32_t* top_count = nullptr,                             // opt: feature 1
    Hf* pbk = nullptr,                                         // opt: feature 2
    u1* pbk_tree_IDs = nullptr, u4* pbk_bitstream = nullptr, u2* pbk_bits = nullptr,
    u4* pbk_entries = nullptr, size_t* pbk_loc = nullptr,
    // breaking handling
    Eq* const brval = nullptr, CI* const bridx = nullptr, CN* const brnum = nullptr)
{
  constexpr auto NumThreads = TileDim / Seq;
  // constexpr auto NumWarps = NumThreads / 32;

  CHECK_LOCAL_STAT_TOGGLE;
  SETUP_ZIGZAG;

  __shared__ uint32_t s_top1_counts[1];
  if (threadIdx.x == 0) s_top1_counts[0] = 0;

  __shared__ T s_data[TileDim];
  __shared__ EqUInt s_eq_uint[TileDim];

  T _thp_data[Seq + 1] = {0};
  auto prev = [&]() -> T& { return _thp_data[0]; };
  auto thp_data = [&](auto i) -> T& { return _thp_data[i + 1]; };

  auto const id_base = blockIdx.x * TileDim;

// dram.in_data to shmem.in_data
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + threadIdx.x + ix * NumThreads;
    if (id < gx) {
      auto val = in_data[id];
      s_data[threadIdx.x + ix * NumThreads] = round(val * ebx2_r);
    }
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

    if constexpr (UseLocalStat) {
      bool is_valid_range = id_base + threadIdx.x * Seq + ix < gx;
      COUNT_LOCAL_STAT(delta, is_valid_range);
    }

    T candidate;
    if constexpr (UseZigZag) {
      candidate = delta;
      s_eq_uint[threadIdx.x * Seq + ix] =
          ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
    }
    else {
      candidate = delta + radius;
      s_eq_uint[threadIdx.x * Seq + ix] = quantizable * (EqUInt)candidate;
    }

    if (not quantizable) {
      auto cur_idx = atomicAdd(out_cn, 1);
      out_cidx[cur_idx] = id_base + threadIdx.x * Seq + ix;
      out_cval[cur_idx] = candidate;
    }
  }
  __syncthreads();

  if constexpr (UseLocalStat) {
    if (threadIdx.x % 32 == 0) atomicAdd(s_top1_counts, thp_top1_count);
    __syncthreads();

    if constexpr (UseGlobalStat)
      if (threadIdx.x == 0) atomicAdd(top_count, s_top1_counts[0]);
  }

  if constexpr (UseLocalStat and EncodeInPlace) {
    constexpr auto ChunkSize = TileDim;  // 1D
    constexpr auto ShardSize = Seq;
    constexpr auto ReduceTimes = 2u;
    constexpr auto ShuffleTimes = 8u;
    constexpr auto BITWIDTH = 32;
#define s_to_encode s_eq_uint

    static_assert(ShardSize == 1 << ReduceTimes, "Wrong reduce times.");
    static_assert(ChunkSize == 1 << (ReduceTimes + ShuffleTimes), "Wrong shuffle times.");

    constexpr auto Radius = 64;
    constexpr auto BkLen = Radius * 2;
    constexpr auto _INKERNEL_PBK_LEN = 11;
    __shared__ Hf s_book[BkLen];
    __shared__ Hf s_reduced[NumThreads];
    __shared__ u4 s_bitcount[NumThreads];
    __shared__ u4 s_closest_idx;

    auto bitcount_of = [](Hf* _w) { return reinterpret_cast<W*>(_w)->bitcount; };
    auto entry = [&]() -> size_t { return ChunkSize * blockIdx.x; };
    auto allowed_len = [&]() { return min((size_t)ChunkSize, gx - entry()); };

    ////////// find the right book
    if (threadIdx.x < 32) {
      float prob = 0.0;
      if (threadIdx.x == 0) { prob = s_top1_counts[0] * 1.0 / TileDim; }
      prob = __shfl_sync(0xffffffff, prob, 0);
      float min_diff = fabsf(probs_lookup[threadIdx.x] - prob);
      int min_idx = threadIdx.x;

      unsigned mask = __ballot_sync(0xffffffff, threadIdx.x < _INKERNEL_PBK_LEN);
      for (int offset = 16; offset > 0; offset /= 2) {
        float shfl_min = __shfl_down_sync(mask, min_diff, offset);
        int shfl_idx = __shfl_down_sync(mask, min_idx, offset);
        if (shfl_min < min_diff and threadIdx.x < _INKERNEL_PBK_LEN)
          min_diff = shfl_min, min_idx = shfl_idx;
      }
      if (threadIdx.x == 0) { s_closest_idx = min_idx; }
    }
    __syncthreads();

    ////////// load codebook
    if (threadIdx.x < BkLen) {
      auto tree_idx = 0u;
      if (threadIdx.x % 32 == 0) { tree_idx = s_closest_idx; }
      tree_idx = __shfl_sync(0xffffffff, tree_idx, 0);
      auto w = pbk[threadIdx.x + tree_idx * BkLen];
      s_book[threadIdx.x] = w;
      if (threadIdx.x == 0) pbk_tree_IDs[blockIdx.x] = tree_idx;
    }
    __syncthreads();

    ////////// start of reduce-merge
    {
      auto p_bits{0u};
      Hf p_reduced{0x0};

      // per-thread loop, merge
      for (auto i = 0; i < ShardSize; i++) {
        auto idx = (threadIdx.x * ShardSize) + i;
        auto p_key = s_to_encode[idx];
        auto p_val = s_book[p_key];
        auto sym_bits = bitcount_of(&p_val);

        p_val <<= (BITWIDTH - sym_bits);
        p_reduced |= (p_val >> p_bits);
        p_bits += sym_bits * (idx < allowed_len());
      }

      // breaking handling
      if (p_bits > BITWIDTH) {
        p_bits = 0u;      // reset on breaking
        p_reduced = 0x0;  // reset on breaking, too
        auto p_val_ref = s_book[radius];
        auto const sym_bits = bitcount_of(&p_val_ref);
        auto br_gidx_start = atomicAdd(brnum, ShardSize);
#pragma unroll
        for (auto ix = 0u, br_lidx = (threadIdx.x * ShardSize); ix < ShardSize; ix++, br_lidx++) {
          bridx[br_gidx_start + ix] = id_base + br_lidx;
          brval[br_gidx_start + ix] = s_to_encode[br_lidx];

          auto p_val = p_val_ref;
          p_val <<= (BITWIDTH - sym_bits);
          p_reduced |= (p_val >> p_bits);
          p_bits += sym_bits * (br_lidx < allowed_len());
        }
      }

      // still for this thread only
      s_reduced[threadIdx.x] = p_reduced;
      s_bitcount[threadIdx.x] = p_bits;

      if constexpr (EIP_DBG) {
        // if (p_bits > 32)
        //   printf(
        //       "ERROR: bitcount %2u > 32 @threadIdx.x %3u of blockIdx.x %5u\n", p_bits,
        //       threadIdx.x, blockIdx.x);
      }
    }
    __syncthreads();

    ////////// end of reduce-merge; start of shuffle-merge

    for (auto sf = ShuffleTimes, stride = 1u; sf > 0; sf--, stride *= 2) {
      auto l = threadIdx.x / (stride * 2) * (stride * 2);
      auto r = l + stride;

      auto lbc = s_bitcount[l];
      u4 used__units = lbc / BITWIDTH;
      u4 used___bits = lbc % BITWIDTH;
      u4 unused_bits = BITWIDTH - used___bits;

      auto lend = (Hf*)(s_reduced + l + used__units);
      auto this_point = s_reduced[threadIdx.x];
      auto lsym = this_point >> used___bits;
      auto rsym = this_point << unused_bits;

      if (threadIdx.x >= r and threadIdx.x < r + stride)
        atomicAnd((Hf*)(s_reduced + threadIdx.x), 0x0);
      __syncthreads();

      if (threadIdx.x >= r and threadIdx.x < r + stride) {
        atomicOr(lend + (threadIdx.x - r) + 0, lsym);
        atomicOr(lend + (threadIdx.x - r) + 1, rsym);
      }

      if (threadIdx.x == l) s_bitcount[l] += s_bitcount[r];
      __syncthreads();
    }

    ////////// end of shuffle-merge, start of outputting
    __shared__ u4 s_wunits;
    __shared__ u4 s_wloc;
    ull p_wunits, p_wloc;

    {
      static_assert(BITWIDTH == 32, "Wrong bitwidth (!=32).");
      if (threadIdx.x == 0) {
        u4 p_bc = s_bitcount[0];
        p_wunits = (p_bc + 31) / 32;
        p_wloc = atomicAdd((ull*)pbk_loc, p_wunits);

        pbk_bits[blockIdx.x] = p_bc;
        pbk_entries[blockIdx.x] = p_wloc;

        s_wloc = p_wloc;
        s_wunits = p_wunits;
      }
      __syncthreads();

      if (threadIdx.x % 32 == 0 and threadIdx.x / 32 > 0) {
        p_wunits = s_wunits;
        p_wloc = s_wloc;
      }
      __syncthreads();

      p_wunits = __shfl_sync(0xffffffff, p_wunits, 0);
      p_wloc = __shfl_sync(0xffffffff, p_wloc, 0);

      for (auto i = threadIdx.x; i < p_wunits; i += blockDim.x) {
        // TODO align for coalesce
        Hf w = s_reduced[i];
        pbk_bitstream[p_wloc + i] = w;
      }
    }

    ////////// end of outputting the encoded
  }
  else {
// write from shmem.eq to dram.eq
#pragma unroll
    for (auto ix = 0; ix < Seq; ix++) {
      auto id = id_base + threadIdx.x + ix * NumThreads;
      if (id < gx) out_eq[id] = s_eq_uint[threadIdx.x + ix * NumThreads];
    }
  }

  // end of kernel
}

template <typename T, bool UseZigZag, typename Eq = uint16_t, typename Fp = T>
__global__ [[deprecated]] void KERNEL_CUHIP_c_lorenzo_2d1l(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3, Eq* const out_eq,
    CompactVal* const out_cval, CompactIdx* const out_cidx, CompactNum* const out_cn,
    uint16_t const radius, Fp const ebx2_r)
{
  SETUP_ZIGZAG;
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

template <
    typename T, bool UseZigZag, typename Eq = uint16_t, typename Fp = T, bool UseLocalStat = true,
    bool UseGlobalStat = true>
__global__ void KERNEL_CUHIP_c_lorenzo_2d1l__32x32(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3, Eq* const out_eq,
    CompactVal* const out_cval, CompactIdx* const out_cidx, CompactNum* const out_cn,
    uint16_t const radius, Fp const ebx2_r, uint32_t* top_count = nullptr)
{
  constexpr auto TileDim = 32;
  constexpr auto Yseq = 8;
  constexpr auto NumWarps = 4;

  CHECK_LOCAL_STAT_TOGGLE;
  SETUP_ZIGZAG;

  __shared__ uint32_t s_top1_counts[1];
  if (cg::this_thread_block().thread_rank() == 0) s_top1_counts[0] = 0;

  __shared__ T exchange[NumWarps - 1][TileDim + 1];

  T center[Yseq + 1] = {0};

  // BDX == TileDim == 32 (a full warp), BDY * Yseq = TileDim == 32
  auto gix = blockIdx.x * TileDim + threadIdx.x;
  auto giy_base = blockIdx.y * TileDim + threadIdx.y * Yseq;
  auto g_id = [&](auto i) { return (giy_base + i) * data_leap3.y + gix; };

// read to private.in_data (center)
#pragma unroll
  for (auto iy = 0; iy < Yseq; iy++) {
    if (gix < data_len3.x and giy_base + iy < data_len3.y)
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

    bool is_valid_range = (gix < data_len3.x and (giy_base + i - 1) < data_len3.y);

    if constexpr (UseLocalStat) { COUNT_LOCAL_STAT(center[i], is_valid_range); }

    T candidate;

    if constexpr (UseZigZag) {
      candidate = center[i];
      if (is_valid_range)
        out_eq[gid] = ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
    }
    else {
      candidate = center[i] + radius;
      if (is_valid_range) out_eq[gid] = quantizable * (EqUInt)candidate;
    }

    if (not quantizable) {
      if (gix < data_len3.x and (giy_base + i - 1) < data_len3.y) {
        auto cur_idx = atomicAdd(out_cn, 1);
        out_cidx[cur_idx] = gid;
        out_cval[cur_idx] = candidate;
      }
    }
  }

  if constexpr (UseLocalStat) {
    if (cg::this_thread_block().thread_rank() % 32 == 0) atomicAdd(s_top1_counts, thp_top1_count);
    __syncthreads();

    if constexpr (UseGlobalStat) {
      if (cg::this_thread_block().thread_rank() == 0) atomicAdd(top_count, s_top1_counts[0]);
    }
  }

  // end of kernel
}

template <
    typename T, bool UseZigZag, typename Eq = uint32_t, typename Fp = T, bool UseLocalStat = true,
    bool UseGlobalStat = true>
__global__ void KERNEL_CUHIP_c_lorenzo_3d1l(
    T* const in_data, dim3 const data_len3, dim3 const data_leap3, Eq* const out_eq,
    CompactVal* const out_cval, CompactIdx* const out_cidx, CompactNum* const out_cn,
    uint16_t const radius, Fp const ebx2_r, uint32_t* top_count = nullptr)
{
  constexpr auto TileDim = 8;
  // constexpr auto NumWarps = 8;

  CHECK_LOCAL_STAT_TOGGLE;
  SETUP_ZIGZAG;

  __shared__ uint32_t s_top1_counts[1];
  if (cg::this_thread_block().thread_rank() == 0) s_top1_counts[0] = 0;

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
          delta[z + 1] = round(in_data[gid(z)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_compact_write = [&](T delta, auto x, auto y, auto z, auto gid) {
    bool quantizable = fabs(delta) < radius;

    if (x < data_len3.x and y < data_len3.y and z < data_len3.z) {
      T candidate;

      if constexpr (UseZigZag) {
        candidate = delta;
        out_eq[gid] = ZigZag::encode(static_cast<EqSInt>(quantizable * candidate));
      }
      else {
        candidate = delta + radius;
        out_eq[gid] = quantizable * (EqUInt)candidate;
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

    if constexpr (UseLocalStat) {
      auto is_valid_range = (gix < data_len3.x and giy < data_len3.y and giz(z - 1) < data_len3.z);
      COUNT_LOCAL_STAT(delta[z], is_valid_range);
    }

    // now delta[z] is delta
    quantize_compact_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    __syncthreads();
  }

  if constexpr (UseLocalStat) {
    if (cg::this_thread_block().thread_rank() % 32 == 0) atomicAdd(s_top1_counts, thp_top1_count);
    __syncthreads();

    if constexpr (UseGlobalStat) {
      if (cg::this_thread_block().thread_rank() == 0) atomicAdd(top_count, s_top1_counts[0]);
    }
  }
}

template <
    typename TIN, typename TOUT, bool ReverseProcess = false, typename Fp = TIN, int TileDim = 256,
    int Seq = 8>
__global__ void KERNEL_CUHIP_lorenzo_prequant(
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

template <typename T, bool UseZigZag, typename Eq, bool EncodeInPlace>
int GPU_c_lorenzo_nd_with_outlier(
    T* const in_data, stdlen3 const _data_len3, Eq* const out_eq, void* out_outlier, u4* out_top1,
    f8 const ebx2, f8 const ebx2_r, uint16_t const radius, void* stream, Hf* pbk, u1* pbk_tree_IDs,
    Hf* pbk_bitstream, u2* pbk_bits, u4* pbk_entries, size_t* pbk_loc, Eq* const brval,
    uint32_t* const bridx, uint32_t* const brnum)
{
  using Compact = _portable::compact_gpu<T>;
  using namespace psz::kernelconfig;

  auto data_len3 = TO_DIM3(_data_len3);
  auto ot = (Compact*)out_outlier;
  auto d = lorenzo_utils::ndim(data_len3);

  // error bound
  auto leap3 = dim3(1, data_len3.x, data_len3.x * data_len3.y);

  if (d == 1) {
    if constexpr (EncodeInPlace) {
      // printf("%s, using PBK\n", __FILE__);
      // printf("input-pbk\t%x\tlegal? %d\n", pbk, pbk != nullptr);
      // printf("res-bitstream\t%x\tlegal? %d\n", pbk_bitstream, pbk_bitstream != nullptr);
      // printf("res-bits\t%x\tlegal? %d\n", pbk_bits, pbk_bits != nullptr);
      // printf("res-blkid\t%x\tlegal? %d\n", pbk_entries, pbk_entries != nullptr);
      // printf("res-loc\t\t%x\tlegal? %d\n", pbk_loc, pbk_loc != nullptr);
    }
    psz::KERNEL_CUHIP_c_lorenzo_1d1l<
        T, c_lorenzo<1>::tile.x, c_lorenzo<1>::sequentiality.x, UseZigZag, Eq, T, true, false,
        EncodeInPlace>
        <<<c_lorenzo<1>::thread_grid(data_len3), c_lorenzo<1>::thread_block, 0,
           (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(
            in_data, data_len3.x, out_eq, ot->val(), ot->idx(), ot->num(), radius, (T)ebx2_r,
            out_top1, pbk, pbk_tree_IDs, pbk_bitstream, pbk_bits, pbk_entries, pbk_loc, brval,
            bridx, brnum);
  }
  else if (d == 2) {
    // psz::KERNEL_CUHIP_c_lorenzo_2d1l<T, UseZigZag, Eq>
    //     <<<c_lorenzo<2>::thread_grid(data_len3), c_lorenzo<2>::thread_block, 0,
    //        (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(
    //         in_data, data_len3, leap3, out_eq, ot->val(), ot->idx(), ot->num(), radius,
    //         (T)ebx2_r);
    psz::KERNEL_CUHIP_c_lorenzo_2d1l__32x32<T, UseZigZag, Eq>
        <<<c_lorenzo<2, 32, 32>::thread_grid(data_len3), c_lorenzo<2, 32, 32>::thread_block, 0,
           (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(
            in_data, data_len3, leap3, out_eq, ot->val(), ot->idx(), ot->num(), radius, (T)ebx2_r,
            out_top1);
  }
  else if (d == 3)
    psz::KERNEL_CUHIP_c_lorenzo_3d1l<T, UseZigZag, Eq>
        <<<c_lorenzo<3>::thread_grid(data_len3), c_lorenzo<3>::thread_block, 0,
           (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(
            in_data, data_len3, leap3, out_eq, ot->val(), ot->idx(), ot->num(), radius, (T)ebx2_r,
            out_top1);
  else
    return CUSZ_NOT_IMPLEMENTED;

  return CUSZ_SUCCESS;
}

template <typename TIN, typename TOUT, bool ReverseProcess>
int GPU_lorenzo_prequant(
    TIN* const in, size_t const len, f8 const ebx2, f8 const ebx2_r, TOUT* const out, void* stream)
{
  using namespace psz::kernelconfig;

  psz::KERNEL_CUHIP_lorenzo_prequant<
      TIN, TOUT, ReverseProcess, TIN, c_lorenzo<1>::tile.x, c_lorenzo<1>::sequentiality.x>
      <<<c_lorenzo<1>::thread_grid(dim3(len)), c_lorenzo<1>::thread_block, 0,
         (GPU_BACKEND_SPECIFIC_STREAM)stream>>>(in, len, ebx2_r, ebx2, out);

  return CUSZ_SUCCESS;
}

}  // namespace psz::module

// -----------------------------------------------------------------------------
#define INSTANCIATE_GPU_L23R_3params(T, USE_ZIGZAG, Eq)                                         \
  template int psz::module::GPU_c_lorenzo_nd_with_outlier<T, USE_ZIGZAG, Eq, false>(            \
      T* const in_data, stdlen3 const data_len3, Eq* const out_eq, void* out_outlier, u4* top1, \
      f8 const ebx2, f8 const ebx2_r, uint16_t const radius, void* stream, Hf*, u1*, Hf*, u2*,  \
      u4*, size_t*, Eq* const, u4* const, u4* const);                                           \
  template int psz::module::GPU_c_lorenzo_nd_with_outlier<T, USE_ZIGZAG, Eq, true>(             \
      T* const in_data, stdlen3 const data_len3, Eq* const out_eq, void* out_outlier, u4* top1, \
      f8 const ebx2, f8 const ebx2_r, uint16_t const radius, void* stream, Hf*, u1*, Hf*, u2*,  \
      u4*, size_t*, Eq* const, u4* const, u4* const);

#define INSTANCIATE_GPU_L23R_2params(T, Eq)   \
  INSTANCIATE_GPU_L23R_3params(T, false, Eq); \
  INSTANCIATE_GPU_L23R_3params(T, true, Eq);

#define INSTANCIATE_GPU_L23R_1param(T) \
  INSTANCIATE_GPU_L23R_2params(T, u1); \
  INSTANCIATE_GPU_L23R_2params(T, u2);

// -----------------------------------------------------------------------------

#define INSTANCIATE_GPU_L23_PREQ_3params(TIN, TOUT, REV)                                \
  template int psz::module::GPU_lorenzo_prequant<TIN, TOUT, REV>(                       \
      TIN* const in, size_t const len, f8 const ebx2, f8 const ebx2_r, TOUT* const out, \
      void* stream);

#define INSTANCIATE_GPU_L23_PREQ_2params(TIN, REV)     \
  INSTANCIATE_GPU_L23_PREQ_3params(TIN, int32_t, REV); \
  INSTANCIATE_GPU_L23_PREQ_3params(TIN, int16_t, REV); \
  INSTANCIATE_GPU_L23_PREQ_3params(TIN, int8_t, REV);

#define INSTANCIATE_GPU_L23_PREQ_1param(T)    \
  INSTANCIATE_GPU_L23_PREQ_2params(T, false); \
  INSTANCIATE_GPU_L23_PREQ_2params(T, true);

//  -----------------------------------------------------------------------------

#undef SETUP_ZIGZAG
