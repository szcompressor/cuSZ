#include <stdint.h>

#include <cmath>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "cusz/suint.hh"
#include "mem/compact.hh"
#include "port.hh"

namespace psz::dpcpp::wave32 {

template <typename T, int SEQ, bool OneapiUseExperimental = false>
__dpct_inline__ void intrawarp_inclusivescan_1d(
    T private_buffer[SEQ], const sycl::nd_item<3>& item_ct1)
{
  for (auto i = 1; i < SEQ; i++) private_buffer[i] += private_buffer[i - 1];
  T addend = private_buffer[SEQ - 1];

  // in-warp shuffle
  for (auto d = 1; d < 32; d *= 2) {
    if constexpr (OneapiUseExperimental) {
      /* DPCT1108 */
      T n = dpct::experimental::shift_sub_group_right(
          0xffffffff, item_ct1.get_sub_group(), addend, d);
      if (item_ct1.get_local_id(2) % 32 >= d) addend += n;
    }
    else {
      /* DPCT1023 */
      /* DPCT1096 */
      T n = dpct::shift_sub_group_right(item_ct1.get_sub_group(), addend, d);
      if (item_ct1.get_local_id(2) % 32 >= d) addend += n;
    }
  }
  // exclusive scan
  if constexpr (OneapiUseExperimental) {
    /* DPCT1108 */
    T prev_addend = dpct::experimental::shift_sub_group_right(
        0xffffffff, item_ct1.get_sub_group(), addend, 1);
    // propagate
    if (item_ct1.get_local_id(2) % 32 > 0)
      for (auto i = 0; i < SEQ; i++) private_buffer[i] += prev_addend;
  }
  else {
    /* DPCT1023 */
    /* DPCT1096 */
    T prev_addend =
        dpct::shift_sub_group_right(item_ct1.get_sub_group(), addend, 1);
    // propagate
    if (item_ct1.get_local_id(2) % 32 > 0)
      for (auto i = 0; i < SEQ; i++) private_buffer[i] += prev_addend;
  }
}

template <typename T, int SEQ, int NTHREAD, bool OneapiUseExperimental = false>
__dpct_inline__ void intrablock_exclusivescan_1d(
    T private_buffer[SEQ], volatile T* exchange_in, volatile T* exchange_out,
    const sycl::nd_item<3>& item_ct1)
{
  constexpr auto NWARP = NTHREAD / 32;
  static_assert(NWARP <= 32, "too big");

  auto warp_id = item_ct1.get_local_id(2) / 32;
  auto lane_id = item_ct1.get_local_id(2) % 32;

  if (lane_id == 31) exchange_in[warp_id] = private_buffer[SEQ - 1];
  item_ct1.barrier(sycl::access::fence_space::local_space);

  if (NWARP <= 8) {
    if (item_ct1.get_local_id(2) == 0) {
      exchange_out[0] = 0;
      for (auto i = 1; i < NWARP; i++)
        exchange_out[i] = exchange_out[i - 1] + exchange_in[i - 1];
    }
  }
  else if (NWARP <= 32) {
    if (item_ct1.get_local_id(2) <= 32) {
      auto addend = exchange_in[item_ct1.get_local_id(2)];

      for (auto d = 1; d < 32; d *= 2) {
        if constexpr (OneapiUseExperimental) {
          /* DPCT1108 */
          T n = dpct::experimental::shift_sub_group_right(
              0xffffffff, item_ct1.get_sub_group(), addend, d);
          if (item_ct1.get_local_id(2) >= d) addend += n;
        }
        else {
          /* DPCT1023 */
          /* DPCT1096 */
          T n =
              dpct::shift_sub_group_right(item_ct1.get_sub_group(), addend, d);
          if (item_ct1.get_local_id(2) >= d) addend += n;
        }
      }
      // exclusive scan
      if constexpr (OneapiUseExperimental) {
        /* DPCT1108 */
        T prev_addend = dpct::experimental::shift_sub_group_right(
            0xffffffff, item_ct1.get_sub_group(), addend, 1);
        exchange_out[warp_id] = (warp_id > 0) * prev_addend;
      }
      else {
        /* DPCT1023 */
        /* DPCT1096 */
        T prev_addend =
            dpct::shift_sub_group_right(item_ct1.get_sub_group(), addend, 1);
        exchange_out[warp_id] = (warp_id > 0) * prev_addend;
      }
    }
  }
  // else-case handled by static_assert
  item_ct1.barrier(sycl::access::fence_space::local_space);

  // propagate
  auto addend = exchange_out[warp_id];
  for (auto i = 0; i < SEQ; i++) private_buffer[i] += addend;
  item_ct1.barrier(sycl::access::fence_space::local_space);
}

}  // namespace psz::dpcpp::wave32

namespace psz::dpcpp {

//////// 1D

// compression load
template <typename T, typename FP, int NTHREAD, int SEQ>
__dpct_inline__ void load_prequant_1d(
    T* data, uint32_t dimx, uint32_t id_base, volatile T* shmem,
    T private_buffer[SEQ], T& prev, FP ebx2_r,
    const sycl::nd_item<3>& item_ct1);

// decompression load
template <typename T, typename EQ, int NTHREAD, int SEQ>
__dpct_inline__ void load_fuse_1d(
    EQ* quant, T* outlier, uint32_t dimx, uint32_t id_base, int radius,
    volatile T* shmem, T private_buffer[SEQ],
    const sycl::nd_item<3>& item_ct1);

// compression and decompression store
template <typename T1, typename T2, int NTHREAD, int SEQ, bool NO_OUTLIER>
__dpct_inline__ void write_1d(  //
    volatile T1* shmem_a1, volatile T2* shmem_a2, uint32_t dimx,
    uint32_t id_base, T1* a1, T2* a2, const sycl::nd_item<3>& item_ct1);

// compression pred-quant, method 1
template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__dpct_inline__ void predict_quantize__no_outlier_1d(  //
    T private_buffer[SEQ], volatile EQ* shmem_quant,
    const sycl::nd_item<3>& item_ct1, T prev = 0);

// compression pred-quant, method 2
template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__dpct_inline__ void predict_quantize_1d(  //
    T private_buffer[SEQ], volatile EQ* shmem_quant, volatile T* shmem_outlier,
    int radius, const sycl::nd_item<3>& item_ct1, T prev = 0);

// decompression pred-quant
template <typename T, int SEQ, int NTHREAD>
__dpct_inline__ void block_scan_1d(
    T private_buffer[SEQ], T ebx2, volatile T* exchange_in,
    volatile T* exchange_out, volatile T* shmem_buffer,
    const sycl::nd_item<3>& item_ct1);

//////// 2D

template <
    typename T, typename FP, int YSEQ, bool OneapiUseExperimental = false>
__dpct_inline__ void load_prequant_2d(
    T* data, uint32_t dimx, uint32_t gix, uint32_t dimy, uint32_t giy_base,
    uint32_t stridey, FP ebx2_r, T center[YSEQ + 1],
    const sycl::nd_item<3>& item_ct1);

template <
    typename T, typename FP, int YSEQ, bool OneapiUseExperimental = false>
__dpct_inline__ void predict_2d(
    T center[YSEQ + 1], const sycl::nd_item<3>& item_ct1);

template <typename T, typename EQ, int YSEQ>
__dpct_inline__ void quantize_write_2d(
    T delta[YSEQ + 1], uint32_t dimx, uint32_t gix, uint32_t dimy,
    uint32_t giy_base, uint32_t stridey, int radius, EQ* quant, T* outlier);

// decompression load
template <typename T, typename EQ, int YSEQ>
__dpct_inline__ void load_fuse_2d(
    EQ* quant, T* outlier, uint32_t dimx, uint32_t gix, uint32_t dimy,
    uint32_t giy_base, uint32_t stridey, int radius, T private_buffer[YSEQ]);

template <
    typename T, typename EQ, typename FP, int YSEQ,
    bool OneapiUseExperimental = false>
__dpct_inline__ void block_scan_2d(  //
    T thread_private[YSEQ], volatile T* intermediate, FP ebx2,
    const sycl::nd_item<3>& item_ct1);

template <typename T, int YSEQ>
__dpct_inline__ void decomp_write_2d(
    T thread_private[YSEQ], uint32_t dimx, uint32_t gix, uint32_t dimy,
    uint32_t giy_base, uint32_t stridey, T* xdata);

//////// 3D

}  // namespace psz::dpcpp

////////////////////////////////////////////////////////////////////////////////

//////// 1D

template <typename T, typename FP, int NTHREAD, int SEQ>
__dpct_inline__ void psz::dpcpp::load_prequant_1d(
    T* data, uint32_t dimx, uint32_t id_base, volatile T* shmem,
    T private_buffer[SEQ],
    T& prev,  // TODO use pointer?
    FP ebx2_r, const sycl::nd_item<3>& item_ct1)
{
#pragma unroll
  for (auto i = 0; i < SEQ; i++) {
    auto id = id_base + item_ct1.get_local_id(2) + i * NTHREAD;
    if (id < dimx)
      shmem[item_ct1.get_local_id(2) + i * NTHREAD] =
          sycl::round(data[id] * ebx2_r);
  }
  /*
  DPCT1065:37: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

#pragma unroll
  for (auto i = 0; i < SEQ; i++)
    private_buffer[i] = shmem[item_ct1.get_local_id(2) * SEQ + i];
  if (item_ct1.get_local_id(2) > 0)
    prev = shmem[item_ct1.get_local_id(2) * SEQ - 1];
  item_ct1.barrier(sycl::access::fence_space::local_space);
}

template <typename T, typename EQ, int NTHREAD, int SEQ>
__dpct_inline__ void psz::dpcpp::load_fuse_1d(
    EQ* quant, T* outlier, uint32_t dimx, uint32_t id_base, int radius,
    volatile T* shmem, T private_buffer[SEQ], const sycl::nd_item<3>& item_ct1)
{
#pragma unroll
  for (auto i = 0; i < SEQ; i++) {
    auto local_id = item_ct1.get_local_id(2) + i * NTHREAD;
    auto id = id_base + local_id;
    if (id < dimx)
      shmem[local_id] = outlier[id] + static_cast<T>(quant[id]) - radius;
  }
  /*
  DPCT1065:39: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

#pragma unroll
  for (auto i = 0; i < SEQ; i++)
    private_buffer[i] = shmem[item_ct1.get_local_id(2) * SEQ + i];
  item_ct1.barrier(sycl::access::fence_space::local_space);
}

template <
    typename T1, typename T2, int NTHREAD, int SEQ,
    bool NO_OUTLIER>  // TODO remove NO_OUTLIER, use nullable
__dpct_inline__ void psz::dpcpp::write_1d(
    volatile T1* shmem_a1, volatile T2* shmem_a2, uint32_t dimx,
    uint32_t id_base, T1* a1, T2* a2, const sycl::nd_item<3>& item_ct1)
{
#pragma unroll
  for (auto i = 0; i < SEQ; i++) {
    auto id = id_base + item_ct1.get_local_id(2) + i * NTHREAD;
    if (id < dimx) {
      if (NO_OUTLIER) {  //
        a1[id] = shmem_a1[item_ct1.get_local_id(2) + i * NTHREAD];
      }
      else {
        a1[id] = shmem_a1[item_ct1.get_local_id(2) + i * NTHREAD];
        a2[id] = shmem_a2[item_ct1.get_local_id(2) + i * NTHREAD];
      }
    }
  }
}

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__dpct_inline__ void psz::dpcpp::predict_quantize__no_outlier_1d(  //
    T private_buffer[SEQ], volatile EQ* shmem_quant,
    const sycl::nd_item<3>& item_ct1, T prev)
{
  auto quantize_1d = [&](T& cur, T& prev, uint32_t idx,
                         const sycl::nd_item<3>& item_ct1) {
    shmem_quant[idx + item_ct1.get_local_id(2) * SEQ] =
        static_cast<EQ>(cur - prev);
  };

  if (FIRST_POINT) {  // i == 0
    quantize_1d(private_buffer[0], prev, 0);
  }
  else {
#pragma unroll
    for (auto i = 1; i < SEQ; i++)
      quantize_1d(private_buffer[i], private_buffer[i - 1], i);
    /*
    DPCT1065:50: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
}

template <typename T, typename EQ, int SEQ, bool FIRST_POINT>
__dpct_inline__ void psz::dpcpp::predict_quantize_1d(
    T private_buffer[SEQ], volatile EQ* shmem_quant, volatile T* shmem_outlier,
    int radius, const sycl::nd_item<3>& item_ct1, T prev)
{
  auto quantize_1d = [&](T& cur, T& prev, uint32_t idx,
                         const sycl::nd_item<3>& item_ct1) {
    T delta = cur - prev;
    bool quantizable = sycl::fabs(delta) < radius;
    T candidate = delta + radius;

    // otherwise, need to reset shared memory (to 0)
    shmem_quant[idx + item_ct1.get_local_id(2) * SEQ] =
        quantizable * static_cast<EQ>(candidate);
    shmem_outlier[idx + item_ct1.get_local_id(2) * SEQ] =
        (not quantizable) * candidate;
  };

  if (FIRST_POINT) {  // i == 0
    quantize_1d(private_buffer[0], prev, 0, item_ct1);
  }
  else {
#pragma unroll
    for (auto i = 1; i < SEQ; i++)
      quantize_1d(private_buffer[i], private_buffer[i - 1], i, item_ct1);
    /*
    DPCT1065:41: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
}

// decompression pred-quant
template <typename T, int SEQ, int NTHREAD>
__dpct_inline__ void psz::dpcpp::block_scan_1d(
    T private_buffer[SEQ], T ebx2, volatile T* exchange_in,
    volatile T* exchange_out, volatile T* shmem_buffer,
    const sycl::nd_item<3>& item_ct1)
{
  namespace wave32 = psz::dpcpp::wave32;
  wave32::intrawarp_inclusivescan_1d<T, SEQ>(private_buffer, item_ct1);
  wave32::intrablock_exclusivescan_1d<T, SEQ, NTHREAD>(
      private_buffer, exchange_in, exchange_out, item_ct1);

  // put back to shmem
#pragma unroll
  for (auto i = 0; i < SEQ; i++)
    shmem_buffer[item_ct1.get_local_id(2) * SEQ + i] =
        private_buffer[i] * ebx2;
  item_ct1.barrier(sycl::access::fence_space::local_space);
}

////////////////////////////////////////////////////////////////////////////////

//////// 2D

template <typename T, typename FP, int YSEQ, bool OneapiUseExperimental>
__dpct_inline__ void psz::dpcpp::load_prequant_2d(
    // clang-format off
    T*       data,
    uint32_t dimx, uint32_t gix,
    uint32_t dimy, uint32_t giy_base, uint32_t stridey,
    FP ebx2_r,
    T  center[YSEQ + 1]
    , const sycl::nd_item<3> &item_ct1  // clang-format on
)
{
  auto g_id = [&](auto iy) { return (giy_base + iy) * stridey + gix; };

  // use a warp as two half-warps
  // block_dim = (16, 2, 1) makes a full warp internally

#pragma unroll
  for (auto iy = 0; iy < YSEQ; iy++) {
    if (gix < dimx and giy_base + iy < dimy)
      center[iy + 1] = sycl::round(data[g_id(iy)] * ebx2_r);
  }
  if constexpr (OneapiUseExperimental) {
    /* DPCT1108 */
    auto tmp = dpct::experimental::shift_sub_group_right(
        0xffffffff, item_ct1.get_sub_group(), center[YSEQ],
        16);  // same-warp, next-16
    if (item_ct1.get_local_id(1) == 1) center[0] = tmp;
  }
  else {
    /* DPCT1023 */
    /* DPCT1096 */
    auto tmp = dpct::shift_sub_group_right(
        item_ct1.get_sub_group(), center[YSEQ], 16);  // same-warp, next-16
    if (item_ct1.get_local_id(1) == 1) center[0] = tmp;
  }
}

template <typename T, typename FP, int YSEQ, bool OneapiUseExperimental>
__dpct_inline__ void psz::dpcpp::predict_2d(
    T center[YSEQ + 1], const sycl::nd_item<3>& item_ct1)
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

    if constexpr (OneapiUseExperimental) {
      /* DPCT1108 */
      auto west = dpct::experimental::shift_sub_group_right(
          0xffffffff, item_ct1.get_sub_group(), center[i], 1, 16);
      if (item_ct1.get_local_id(2) > 0) center[i] -= west;  // delta
    }
    else {
      /* DPCT1023 */
      /* DPCT1096 */
      auto west = dpct::shift_sub_group_right(
          item_ct1.get_sub_group(), center[i], 1, 16);
      if (item_ct1.get_local_id(2) > 0) center[i] -= west;  // delta
    }
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);
}

template <typename T, typename EQ, int YSEQ>
__dpct_inline__ void psz::dpcpp::quantize_write_2d(
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
      bool quantizable = sycl::fabs(delta[i]) < radius;
      T candidate = delta[i] + radius;

      // outlier array is not in sparse form in this version
      quant[gid] = quantizable * static_cast<EQ>(candidate);
      outlier[gid] = (not quantizable) * candidate;
    }
  }
}

// load to thread-private array (fuse at the same time)
template <typename T, typename EQ, int YSEQ>
__dpct_inline__ void psz::dpcpp::load_fuse_2d(
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

// partial-sum along y-axis, sequantially
// then, in-warp partial-sum along x-axis
template <
    typename T, typename EQ, typename FP, int YSEQ, bool OneapiUseExperimental>
__dpct_inline__ void psz::dpcpp::block_scan_2d(
    T thread_private[YSEQ], volatile T* intermediate, FP ebx2,
    const sycl::nd_item<3>& item_ct1)
{
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

  constexpr auto BLOCK = 16;

  for (auto i = 1; i < YSEQ; i++) thread_private[i] += thread_private[i - 1];
  // two-pass: store for cross-thread-private update
  // TODO shuffle up by 16 in the same warp
  if (item_ct1.get_local_id(1) == 0)
    intermediate[item_ct1.get_local_id(2)] = thread_private[YSEQ - 1];
  item_ct1.barrier(sycl::access::fence_space::local_space);
  // broadcast the partial-sum result from a previous segment
  if (item_ct1.get_local_id(1) == 1) {
    auto tmp = intermediate[item_ct1.get_local_id(2)];
#pragma unroll
    for (auto i = 0; i < YSEQ; i++)
      thread_private[i] += tmp;  // regression as pointer
  }
  // implicit sync as there is half-warp divergence

#pragma unroll
  for (auto i = 0; i < YSEQ; i++) {
    for (auto d = 1; d < BLOCK; d *= 2) {
      if constexpr (OneapiUseExperimental) {
        /* DPCT1108 */
        T n = dpct::experimental::shift_sub_group_right(
            0xffffffff, item_ct1.get_sub_group(), thread_private[i], d,
            16);  // half-warp shuffle
        if (item_ct1.get_local_id(2) >= d) thread_private[i] += n;
      }
      else {
        /* DPCT1023 */
        /* DPCT1096 */
        T n = dpct::shift_sub_group_right(
            item_ct1.get_sub_group(), thread_private[i], d,
            16);  // half-warp shuffle
        if (item_ct1.get_local_id(2) >= d) thread_private[i] += n;
      }
    }
    thread_private[i] *= ebx2;  // scale accordingly
  }
}

// write to DRAM
template <typename T, int YSEQ>
__dpct_inline__ void psz::dpcpp::decomp_write_2d(
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
