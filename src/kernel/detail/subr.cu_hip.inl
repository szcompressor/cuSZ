/**
 * @file subr.cu_hip.inl
 * @author Jiannan Tian
 * @brief subroutines of kernels
 * @version 0.4
 * @date 2022-12-22
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include <stdint.h>

#include <type_traits>

#include "cusz/suint.hh"
#include "mem/compact.hh"
#include "port.hh"

namespace psz {
namespace cu_hip {
namespace wave32 {

template <typename T, int SEQ>
__forceinline__ __device__ void intrawarp_inclusivescan_1d(
    T private_buffer[SEQ])
{
  for (auto i = 1; i < SEQ; i++) private_buffer[i] += private_buffer[i - 1];
  T addend = private_buffer[SEQ - 1];

  // in-warp shuffle
  for (auto d = 1; d < 32; d *= 2) {
    T n = __shfl_up_sync(0xffffffff, addend, d, 32);
    if (threadIdx.x % 32 >= d) addend += n;
  }
  // exclusive scan
  T prev_addend = __shfl_up_sync(0xffffffff, addend, 1, 32);

  // propagate
  if (threadIdx.x % 32 > 0)
    for (auto i = 0; i < SEQ; i++) private_buffer[i] += prev_addend;
}

template <typename T, int SEQ, int NTHREAD>
__forceinline__ __device__ void intrablock_exclusivescan_1d(
    T private_buffer[SEQ], volatile T* exchange_in, volatile T* exchange_out)
{
  constexpr auto NWARP = NTHREAD / 32;
  static_assert(NWARP <= 32, "too big");

  auto warp_id = threadIdx.x / 32;
  auto lane_id = threadIdx.x % 32;

  if (lane_id == 31) exchange_in[warp_id] = private_buffer[SEQ - 1];
  __syncthreads();

  if (NWARP <= 8) {
    if (threadIdx.x == 0) {
      exchange_out[0] = 0;
      for (auto i = 1; i < NWARP; i++)
        exchange_out[i] = exchange_out[i - 1] + exchange_in[i - 1];
    }
  }
  else if (NWARP <= 32) {
    if (threadIdx.x <= 32) {
      auto addend = exchange_in[threadIdx.x];

      for (auto d = 1; d < 32; d *= 2) {
        T n = __shfl_up_sync(0xffffffff, addend, d, 32);
        if (threadIdx.x >= d) addend += n;
      }
      // exclusive scan
      T prev_addend = __shfl_up_sync(0xffffffff, addend, 1, 32);
      exchange_out[warp_id] = (warp_id > 0) * prev_addend;
    }
  }
  // else-case handled by static_assert
  __syncthreads();

  // propagate
  auto addend = exchange_out[warp_id];
  for (auto i = 0; i < SEQ; i++) private_buffer[i] += addend;
  __syncthreads();
}

}  // namespace wave32
}  // namespace cu_hip
}  // namespace psz

namespace psz {
namespace cuda_hip {

//////// 1D

// compression load
template <typename T, typename FP, int NTHREAD, int SEQ>
__forceinline__ __device__ void load_prequant_1d(
    T* data, uint32_t dimx, uint32_t id_base, volatile T* shmem,
    T private_buffer[SEQ], T& prev, FP ebx2_r);

// decompression load
template <typename T, typename EQ, int NTHREAD, int SEQ>
__forceinline__ __device__ void load_fuse_1d(
    EQ* quant, T* outlier, uint32_t dimx, uint32_t id_base, int radius,
    volatile T* shmem, T private_buffer[SEQ]);

namespace delta_only {

template <typename T, typename EQ, int NTHREAD, int SEQ>
__forceinline__ __device__ void load_1d(
    EQ* quant, uint32_t dimx, uint32_t id_base, volatile T* shmem,
    T private_buffer[SEQ]);

}

// compression and decompression store
template <typename T1, typename T2, int NTHREAD, int SEQ, bool NO_OUTLIER>
__forceinline__ __device__ void write_1d(  //
    volatile T1* shmem_a1, volatile T2* shmem_a2, uint32_t dimx,
    uint32_t id_base, T1* a1, T2* a2);

// compression pred-quant, method 1
template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void predict_quantize__no_outlier_1d(  //
    T private_buffer[SEQ], volatile EQ* shmem_quant, T prev = 0);

// compression pred-quant, method 2
template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void predict_quantize_1d(  //
    T private_buffer[SEQ], volatile EQ* shmem_quant, volatile T* shmem_outlier,
    int radius, T prev = 0);

// decompression pred-quant
template <typename T, int SEQ, int NTHREAD>
__forceinline__ __device__ void block_scan_1d(
    T private_buffer[SEQ], T ebx2, volatile T* exchange_in,
    volatile T* exchange_out, volatile T* shmem_buffer);

//////// 2D

template <typename T, typename FP, int YSEQ>
__forceinline__ __device__ void load_prequant_2d(
    T* data, uint32_t dimx, uint32_t gix, uint32_t dimy, uint32_t giy_base,
    uint32_t stridey, FP ebx2_r, T center[YSEQ + 1]);

template <typename T, typename FP, int YSEQ>
__forceinline__ __device__ void predict_2d(T center[YSEQ + 1]);

template <typename T, typename EQ, int YSEQ>
__forceinline__ __device__ void quantize_write_2d(
    T delta[YSEQ + 1], uint32_t dimx, uint32_t gix, uint32_t dimy,
    uint32_t giy_base, uint32_t stridey, int radius, EQ* quant, T* outlier);

namespace delta_only {

template <typename T, typename EQ, int YSEQ>
__forceinline__ __device__ void quantize_write_2d(
    T delta[YSEQ + 1], uint32_t dimx, uint32_t gix, uint32_t dimy,
    uint32_t giy_base, uint32_t stridey, EQ* quant);

}

// decompression load
template <typename T, typename EQ, int YSEQ>
__forceinline__ __device__ void load_fuse_2d(
    EQ* quant, T* outlier, uint32_t dimx, uint32_t gix, uint32_t dimy,
    uint32_t giy_base, uint32_t stridey, int radius, T private_buffer[YSEQ]);

namespace delta_only {
// decompression load
template <typename T, typename EQ, int YSEQ>
__forceinline__ __device__ void load_2d(
    EQ* quant, uint32_t dimx, uint32_t gix, uint32_t dimy, uint32_t giy_base,
    uint32_t stridey, T private_buffer[YSEQ]);

}  // namespace delta_only

template <typename T, typename EQ, typename FP, int YSEQ>
__forceinline__ __device__ void block_scan_2d(  //
    T thread_private[YSEQ], volatile T* intermediate, FP ebx2);

template <typename T, int YSEQ>
__forceinline__ __device__ void decomp_write_2d(
    T thread_private[YSEQ], uint32_t dimx, uint32_t gix, uint32_t dimy,
    uint32_t giy_base, uint32_t stridey, T* xdata);

//////// 3D

// TODO move subroutines for 3D here

}  // namespace cuda_hip
}  // namespace psz

////////////////////////////////////////////////////////////////////////////////

//////// 1D

template <typename T, typename FP, int NTHREAD, int SEQ>
__forceinline__ __device__ void psz::cuda_hip::load_prequant_1d(
    T* data, uint32_t dimx, uint32_t id_base, volatile T* shmem,
    T private_buffer[SEQ],
    T& prev,  // TODO use pointer?
    FP ebx2_r)
{
#pragma unroll
  for (auto i = 0; i < SEQ; i++) {
    auto id = id_base + threadIdx.x + i * NTHREAD;
    if (id < dimx) shmem[threadIdx.x + i * NTHREAD] = round(data[id] * ebx2_r);
  }
  __syncthreads();

#pragma unroll
  for (auto i = 0; i < SEQ; i++)
    private_buffer[i] = shmem[threadIdx.x * SEQ + i];
  if (threadIdx.x > 0) prev = shmem[threadIdx.x * SEQ - 1];
  __syncthreads();
}

template <typename T, typename EQ, int NTHREAD, int SEQ>
__forceinline__ __device__ void psz::cuda_hip::load_fuse_1d(
    EQ* quant, T* outlier, uint32_t dimx, uint32_t id_base, int radius,
    volatile T* shmem, T private_buffer[SEQ])
{
#pragma unroll
  for (auto i = 0; i < SEQ; i++) {
    auto local_id = threadIdx.x + i * NTHREAD;
    auto id = id_base + local_id;
    if (id < dimx)
      shmem[local_id] = outlier[id] + static_cast<T>(quant[id]) - radius;
  }
  __syncthreads();

#pragma unroll
  for (auto i = 0; i < SEQ; i++)
    private_buffer[i] = shmem[threadIdx.x * SEQ + i];
  __syncthreads();
}

template <typename T, typename EQ, int NTHREAD, int SEQ>
__forceinline__ __device__ void psz::cuda_hip::delta_only::load_1d(
    EQ* quant, uint32_t dimx, uint32_t id_base, volatile T* shmem,
    T private_buffer[SEQ])
{
#pragma unroll
  for (auto i = 0; i < SEQ; i++) {
    auto local_id = threadIdx.x + i * NTHREAD;
    auto id = id_base + local_id;
    if (id < dimx) shmem[local_id] = static_cast<T>(quant[id]);
  }
  __syncthreads();

#pragma unroll
  for (auto i = 0; i < SEQ; i++)
    private_buffer[i] = shmem[threadIdx.x * SEQ + i];
  __syncthreads();
}

template <
    typename T1, typename T2, int NTHREAD, int SEQ,
    bool NO_OUTLIER>  // TODO remove NO_OUTLIER, use nullable
__forceinline__ __device__ void psz::cuda_hip::write_1d(
    volatile T1* shmem_a1, volatile T2* shmem_a2, uint32_t dimx,
    uint32_t id_base, T1* a1, T2* a2)
{
#pragma unroll
  for (auto i = 0; i < SEQ; i++) {
    auto id = id_base + threadIdx.x + i * NTHREAD;
    if (id < dimx) {
      if (NO_OUTLIER) {  //
        a1[id] = shmem_a1[threadIdx.x + i * NTHREAD];
      }
      else {
        a1[id] = shmem_a1[threadIdx.x + i * NTHREAD];
        a2[id] = shmem_a2[threadIdx.x + i * NTHREAD];
      }
    }
  }
}

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void
psz::cuda_hip::predict_quantize__no_outlier_1d(  //
    T private_buffer[SEQ], volatile EQ* shmem_quant, T prev)
{
  auto quantize_1d = [&](T& cur, T& prev, uint32_t idx) {
    shmem_quant[idx + threadIdx.x * SEQ] = static_cast<EQ>(cur - prev);
  };

  if (FIRST_POINT) {  // i == 0
    quantize_1d(private_buffer[0], prev, 0);
  }
  else {
#pragma unroll
    for (auto i = 1; i < SEQ; i++)
      quantize_1d(private_buffer[i], private_buffer[i - 1], i);
    __syncthreads();
  }
}

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__forceinline__ __device__ void psz::cuda_hip::predict_quantize_1d(
    T private_buffer[SEQ], volatile EQ* shmem_quant, volatile T* shmem_outlier,
    int radius, T prev)
{
  auto quantize_1d = [&](T& cur, T& prev, uint32_t idx) {
    T delta = cur - prev;
    bool quantizable = fabs(delta) < radius;
    T candidate = delta + radius;

    // otherwise, need to reset shared memory (to 0)
    shmem_quant[idx + threadIdx.x * SEQ] =
        quantizable * static_cast<EQ>(candidate);
    shmem_outlier[idx + threadIdx.x * SEQ] = (not quantizable) * candidate;
  };

  if (FIRST_POINT) {  // i == 0
    quantize_1d(private_buffer[0], prev, 0);
  }
  else {
#pragma unroll
    for (auto i = 1; i < SEQ; i++)
      quantize_1d(private_buffer[i], private_buffer[i - 1], i);
    __syncthreads();
  }
}

// decompression pred-quant
template <typename T, int SEQ, int NTHREAD>
__forceinline__ __device__ void psz::cuda_hip::block_scan_1d(
    T private_buffer[SEQ], T ebx2, volatile T* exchange_in,
    volatile T* exchange_out, volatile T* shmem_buffer)
{
  namespace wave32 = psz::cu_hip::wave32;
  wave32::intrawarp_inclusivescan_1d<T, SEQ>(private_buffer);
  wave32::intrablock_exclusivescan_1d<T, SEQ, NTHREAD>(
      private_buffer, exchange_in, exchange_out);

  // put back to shmem
#pragma unroll
  for (auto i = 0; i < SEQ; i++)
    shmem_buffer[threadIdx.x * SEQ + i] = private_buffer[i] * ebx2;
  __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////

//////// 2D

template <typename T, typename FP, int YSEQ>
__forceinline__ __device__ void psz::cuda_hip::load_prequant_2d(
    // clang-format off
    T*       data,
    uint32_t dimx, uint32_t gix,
    uint32_t dimy, uint32_t giy_base, uint32_t stridey,
    FP ebx2_r,
    T  center[YSEQ + 1]
    // clang-format on
)
{
  auto g_id = [&](auto iy) { return (giy_base + iy) * stridey + gix; };

  // use a warp as two half-warps
  // block_dim = (16, 2, 1) makes a full warp internally

#pragma unroll
  for (auto iy = 0; iy < YSEQ; iy++) {
    if (gix < dimx and giy_base + iy < dimy)
      center[iy + 1] = round(data[g_id(iy)] * ebx2_r);
  }
  auto tmp =
      __shfl_up_sync(0xffffffff, center[YSEQ], 16, 32);  // same-warp, next-16
  if (threadIdx.y == 1) center[0] = tmp;
}

template <typename T, typename FP, int YSEQ>
__forceinline__ __device__ void psz::cuda_hip::predict_2d(T center[YSEQ + 1])
{
  /*
     Lorenzo 2D (1-layer) illustration
               NW N NE
     notation   W C E   "->" to predict
     --------  SW S SE

              normal data layout       |   considering register file
              col(k-1)    col(k)       |   thread(k-1)        thread(k)
                                       |
     r(i-1)  -west[i-1]  +center[i-1]  |  -center(k-1)[i-1]  +center(k)[i-1]
     r(i  )  +west[i]   ->center[i]    |  +center(k-1)[i]   ->center(k)[i]

     calculation
     -----------
     delta = center[i] - (center[i-1] + west[i] - west[i-1])
           = (center[i] - center[i-1]) - (west[i] - west[i-1])

     With center[i] -= center[i-1] and west[i] -= west[i-1],
     delta = center[i] - west[i]

     For thread(k),
     delta(k) = center(k)[i] - center(k-1)[i]
              = center(k)[i] - SHFL_UP(center(k)[i], 1, HALF_WARP)
   */

#pragma unroll
  for (auto i = YSEQ; i > 0; i--) {
    // with center[i-1] intact in this iteration
    center[i] -= center[i - 1];
    // within a halfwarp (32/2)
    auto west = __shfl_up_sync(0xffffffff, center[i], 1, 16);
    if (threadIdx.x > 0) center[i] -= west;  // delta
  }
  __syncthreads();
}

template <typename T, typename EQ, int YSEQ>
__forceinline__ __device__ void psz::cuda_hip::quantize_write_2d(
    // clang-format off
    T        delta[YSEQ + 1],
    uint32_t dimx,  uint32_t gix,
    uint32_t dimy,  uint32_t giy_base, uint32_t stridey,
    int      radius,
    EQ*      quant, 
    T*       outlier
    // clang-format on
)
{
  auto get_gid = [&](auto i) { return (giy_base + i) * stridey + gix; };

#pragma unroll
  for (auto i = 1; i < YSEQ + 1; i++) {
    auto gid = get_gid(i - 1);

    if (gix < dimx and giy_base + (i - 1) < dimy) {
      bool quantizable = fabs(delta[i]) < radius;
      T candidate = delta[i] + radius;

      // outlier array is not in sparse form in this version
      quant[gid] = quantizable * static_cast<EQ>(candidate);
      outlier[gid] = (not quantizable) * candidate;
    }
  }
}

template <typename T, typename EQ, int YSEQ>
__forceinline__ __device__ void psz::cuda_hip::delta_only::quantize_write_2d(
    // clang-format off
    T        delta[YSEQ + 1],
    uint32_t dimx,  uint32_t gix,
    uint32_t dimy,  uint32_t giy_base, uint32_t stridey,
    EQ*      quant
    // clang-format on
)
{
  auto get_gid = [&](auto i) { return (giy_base + i) * stridey + gix; };

#pragma unroll
  for (auto i = 1; i < YSEQ + 1; i++) {
    auto gid = get_gid(i - 1);
    if (gix < dimx and giy_base + (i - 1) < dimy)
      quant[gid] = static_cast<EQ>(delta[i]);
  }
}

// load to thread-private array (fuse at the same time)
template <typename T, typename EQ, int YSEQ>
__forceinline__ __device__ void psz::cuda_hip::load_fuse_2d(
    // clang-format off
    EQ*      quant,
    T*       outlier,
    uint32_t dimx, uint32_t gix,
    uint32_t dimy, uint32_t giy_base, uint32_t stridey,
    int      radius,
    T        thread_private[YSEQ]
    // clang-format on
)
{
  auto get_gid = [&](auto iy) { return (giy_base + iy) * stridey + gix; };

#pragma unroll
  for (auto i = 0; i < YSEQ; i++) {
    auto gid = get_gid(i);
    // even if we hit the else branch, all threads in a warp hit the y-boundary
    // simultaneously
    if (gix < dimx and (giy_base + i) < dimy)
      thread_private[i] =
          outlier[gid] + static_cast<T>(quant[gid]) - radius;  // fuse
    else
      thread_private[i] = 0;  // TODO set as init state?
  }
}

// load to thread-private array (fuse at the same time)
template <typename T, typename EQ, int YSEQ>
__forceinline__ __device__ void psz::cuda_hip::delta_only::load_2d(
    // clang-format off
    EQ*      quant,
    uint32_t dimx, uint32_t gix,
    uint32_t dimy, uint32_t giy_base, uint32_t stridey,
    T        thread_private[YSEQ]
    // clang-format on
)
{
  auto get_gid = [&](auto iy) { return (giy_base + iy) * stridey + gix; };

#pragma unroll
  for (auto i = 0; i < YSEQ; i++) {
    auto gid = get_gid(i);
    // even if we hit the else branch, all threads in a warp hit the y-boundary
    // simultaneously
    if (gix < dimx and (giy_base + i) < dimy)
      thread_private[i] = static_cast<T>(quant[gid]);
    else
      thread_private[i] = 0;  // TODO set as init state?
  }
}

// partial-sum along y-axis, sequantially
// then, in-warp partial-sum along x-axis
template <typename T, typename EQ, typename FP, int YSEQ>
__forceinline__ __device__ void psz::cuda_hip::block_scan_2d(
    T thread_private[YSEQ], volatile T* intermediate, FP ebx2)
{
  constexpr auto BLOCK = 16;

  for (auto i = 1; i < YSEQ; i++) thread_private[i] += thread_private[i - 1];
  // two-pass: store for cross-thread-private update
  // TODO shuffle up by 16 in the same warp
  if (threadIdx.y == 0) intermediate[threadIdx.x] = thread_private[YSEQ - 1];
  __syncthreads();
  // broadcast the partial-sum result from a previous segment
  if (threadIdx.y == 1) {
    auto tmp = intermediate[threadIdx.x];
#pragma unroll
    for (auto i = 0; i < YSEQ; i++)
      thread_private[i] += tmp;  // regression as pointer
  }
  // implicit sync as there is half-warp divergence

#pragma unroll
  for (auto i = 0; i < YSEQ; i++) {
    for (auto d = 1; d < BLOCK; d *= 2) {
      T n = __shfl_up_sync(
          0xffffffff, thread_private[i], d, 16);  // half-warp shuffle
      if (threadIdx.x >= d) thread_private[i] += n;
    }
    thread_private[i] *= ebx2;  // scale accordingly
  }
}

// write to DRAM
template <typename T, int YSEQ>
__forceinline__ __device__ void psz::cuda_hip::decomp_write_2d(
    // clang-format off
    T        thread_private[YSEQ],
    uint32_t dimx, uint32_t gix,
    uint32_t dimy, uint32_t giy_base, uint32_t stridey,
    T*       xdata
    // clang-format on
)
{
  auto get_gid = [&](auto iy) { return (giy_base + iy) * stridey + gix; };

#pragma unroll
  for (auto i = 0; i < YSEQ; i++) {
    auto gid = get_gid(i);
    if (gix < dimx and (giy_base + i) < dimy) xdata[gid] = thread_private[i];
  }
}

////////////////////////////////////////////////////////////////////////////////

//////// 3D
