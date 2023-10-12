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

#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "cusz/suint.hh"
#include "port.hh"
#include "wave32.dp.inl"

namespace psz {
namespace dpcpp {
namespace __kernel {

template <typename T, typename Eq, typename FP = T, int BLOCK, int SEQ>
/* DPCT1110: high register pressure */
void x_lorenzo_1d1l(  //
    Eq *eq, T *outlier, sycl::range<3> len3, sycl::range<3> stride3,
    int radius, FP ebx2, T *xdata, const sycl::nd_item<3> &item_ct1,
    T *scratch, Eq *s_eq, T *exch_in, T *exch_out)
{
  constexpr auto NTHREAD = BLOCK / SEQ;  // equiv. to blockDim.x

  T thp_data[SEQ];

  auto id_base = item_ct1.get_group(2) * BLOCK;

  auto load_fuse_1d = [&](const sycl::nd_item<3> &item_ct1) {
#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
      auto local_id = item_ct1.get_local_id(2) + i * NTHREAD;
      auto id = id_base + local_id;
      if (id < len3[2])
        scratch[local_id] = outlier[id] + static_cast<T>(eq[id]) - radius;
    }
    item_ct1.barrier();

#pragma unroll
    for (auto i = 0; i < SEQ; i++)
      thp_data[i] = scratch[item_ct1.get_local_id(2) * SEQ + i];
    item_ct1.barrier(sycl::access::fence_space::local_space);
  };

  auto block_scan_1d = [&](const sycl::nd_item<3> &item_ct1) {
    namespace wave32 = psz::dpcpp::wave32;
    wave32::intrawarp_inclscan_1d<T, SEQ>(thp_data, item_ct1);
    wave32::intrablock_exclscan_1d<T, SEQ, NTHREAD>(
        thp_data, exch_in, exch_out, item_ct1);

    // put back to shmem
#pragma unroll
    for (auto i = 0; i < SEQ; i++)
      scratch[item_ct1.get_local_id(2) * SEQ + i] = thp_data[i] * ebx2;
    item_ct1.barrier(sycl::access::fence_space::local_space);
  };

  auto write_1d = [&](const sycl::nd_item<3> &item_ct1) {
#pragma unroll
    for (auto i = 0; i < SEQ; i++) {
      auto local_id = item_ct1.get_local_id(2) + i * NTHREAD;
      auto id = id_base + local_id;
      if (id < len3[2]) xdata[id] = scratch[local_id];
    }
  };

  /*-----------*/

  load_fuse_1d(item_ct1);
  block_scan_1d(item_ct1);
  write_1d(item_ct1);
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

template <typename T, typename Eq, typename FP = T>
/* DPCT1110: high register pressure */
void x_lorenzo_2d1l(  //
    Eq *eq, T *outlier, sycl::range<3> len3, sycl::range<3> stride3,
    int radius, FP ebx2, T *xdata, const sycl::nd_item<3> &item_ct1,
    T *scratch)
{
  constexpr auto BLOCK = 16;
  constexpr auto YSEQ = BLOCK / 2;  // sequentiality in y direction
  static_assert(BLOCK == 16, "In one case, we need BLOCK for 2D == 16");

  // TODO use warp shuffle to eliminate this
  T thp_data[YSEQ] = {0};

  auto gix = item_ct1.get_group(2) * BLOCK + item_ct1.get_local_id(2);
  auto giy_base = item_ct1.get_group(1) * BLOCK +
                  item_ct1.get_local_id(1) * YSEQ;  // BDY * YSEQ = BLOCK == 16

  auto get_gid = [&](auto i) { return (giy_base + i) * stride3[1] + gix; };

  auto load_fuse_2d = [&](const sycl::nd_item<3> &item_ct1) {

#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      auto gid = get_gid(i);
      if (gix < len3[2] and (giy_base + i) < len3[1])
        thp_data[i] = outlier[gid] + static_cast<T>(eq[gid]) - radius;  // fuse
    }
  };

  // partial-sum along y-axis, sequantially
  // then, in-warp partial-sum along x-axis
  auto block_scan_2d = [&](const sycl::nd_item<3> &item_ct1) {
    for (auto i = 1; i < YSEQ; i++) thp_data[i] += thp_data[i - 1];
    // two-pass: store for cross-thread-private update
    // TODO shuffle up by 16 in the same warp
    if (item_ct1.get_local_id(1) == 0)
      scratch[item_ct1.get_local_id(2)] = thp_data[YSEQ - 1];
    item_ct1.barrier(sycl::access::fence_space::local_space);
    // broadcast the partial-sum result from a previous segment
    if (item_ct1.get_local_id(1) == 1) {
      auto tmp = scratch[item_ct1.get_local_id(2)];
#pragma unroll
      for (auto i = 0; i < YSEQ; i++)
        thp_data[i] += tmp;  // regression as pointer
    }
    // implicit sync as there is half-warp divergence

#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      for (auto d = 1; d < BLOCK; d *= 2) {
        T n = psz::dpcpp::compat::shift_sub_group_right(
            0xffffffff, item_ct1.get_sub_group(), thp_data[i], d,
            16);  // half-warp shuffle
        if (item_ct1.get_local_id(2) >= d) thp_data[i] += n;
      }
      thp_data[i] *= ebx2;  // scale accordingly
    }
  };

  auto decomp_write_2d = [&](const sycl::nd_item<3> &item_ct1) {
#pragma unroll
    for (auto i = 0; i < YSEQ; i++) {
      auto gid = get_gid(i);
      if (gix < len3[2] and (giy_base + i) < len3[1]) xdata[gid] = thp_data[i];
    }
  };

  /*-----------*/

  load_fuse_2d(item_ct1);
  block_scan_2d(item_ct1);
  decomp_write_2d(item_ct1);
}

// 32x8x8 data block maps to 32x1x8 thread block
template <typename T, typename Eq, typename FP = T>
/* DPCT1110: high register pressure */
void x_lorenzo_3d1l(  //
    Eq *eq, T *outlier, sycl::range<3> len3, sycl::range<3> stride3,
    int radius, FP ebx2, T *xdata, const sycl::nd_item<3> &item_ct1,
    sycl::local_accessor<T, 3> scratch)
{
  constexpr auto BLOCK = 8;
  constexpr auto YSEQ = BLOCK;
  static_assert(BLOCK == 8, "In one case, we need BLOCK for 3D == 8");

  T thread_private[YSEQ] = {0};

  auto seg_id = item_ct1.get_local_id(2) / 8;
  auto seg_tix = item_ct1.get_local_id(2) % 8;

  auto gix = item_ct1.get_group(2) * (4 * BLOCK) + item_ct1.get_local_id(2);
  auto giy_base = item_ct1.get_group(1) * BLOCK;
  auto giy = [&](auto y) { return giy_base + y; };
  auto giz = item_ct1.get_group(0) * BLOCK + item_ct1.get_local_id(0);
  auto gid = [&](auto y) {
    return giz * stride3[0] + (giy_base + y) * stride3[1] + gix;
  };

  auto load_fuse_3d = [&](const sycl::nd_item<3> &item_ct1) {
  // load to thread-private array (fuse at the same time)
#pragma unroll
    for (auto y = 0; y < YSEQ; y++) {
      if (gix < len3[2] and giy_base + y < len3[1] and giz < len3[0])
        thread_private[y] =
            outlier[gid(y)] + static_cast<T>(eq[gid(y)]) - radius;  // fuse
    }
  };

  auto block_scan_3d = [&](const sycl::nd_item<3> &item_ct1) {
    // partial-sum along y-axis, sequentially
    for (auto y = 1; y < YSEQ; y++) thread_private[y] += thread_private[y - 1];

#pragma unroll
    for (auto i = 0; i < BLOCK; i++) {
      // ND partial-sums along x- and z-axis
      // in-warp shuffle used: in order to perform, it's transposed after
      // X-partial sum
      T val = thread_private[i];

      for (auto dist = 1; dist < BLOCK; dist *= 2) {
        auto addend = psz::dpcpp::compat::shift_sub_group_right(
            0xffffffff, item_ct1.get_sub_group(), val, dist, 8);
        if (seg_tix >= dist) val += addend;
      }

      // x-z transpose
      scratch[item_ct1.get_local_id(0)][seg_id][seg_tix] = val;
      item_ct1.barrier(sycl::access::fence_space::local_space);
      val = scratch[seg_tix][seg_id][item_ct1.get_local_id(0)];
      item_ct1.barrier(sycl::access::fence_space::local_space);

      for (auto dist = 1; dist < BLOCK; dist *= 2) {
        auto addend = psz::dpcpp::compat::shift_sub_group_right(
            0xffffffff, item_ct1.get_sub_group(), val, dist, 8);
        if (seg_tix >= dist) val += addend;
      }

      scratch[item_ct1.get_local_id(0)][seg_id][seg_tix] = val;
      item_ct1.barrier(sycl::access::fence_space::local_space);
      val = scratch[seg_tix][seg_id][item_ct1.get_local_id(0)];
      item_ct1.barrier(sycl::access::fence_space::local_space);

      thread_private[i] = val;
    }
  };

  auto decomp_write_3d = [&](const sycl::nd_item<3> &item_ct1) {
#pragma unroll
    for (auto y = 0; y < YSEQ; y++)
      if (gix < len3[2] and giy(y) < len3[1] and giz < len3[0])
        xdata[gid(y)] = thread_private[y] * ebx2;
  };

  ////////////////////////////////////////////////////////////////////////////
  load_fuse_3d(item_ct1);
  block_scan_3d(item_ct1);
  decomp_write_3d(item_ct1);
}

}  // namespace __kernel
}  // namespace dpcpp
}  // namespace psz