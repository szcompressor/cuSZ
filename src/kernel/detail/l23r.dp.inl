#ifndef F0EAE3AE_EDAC_46DC_B1C2_080FA2725740
#define F0EAE3AE_EDAC_46DC_B1C2_080FA2725740

#include <cmath>
#include <cstdint>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "kernel/lrz.hh"
#include "mem/compact.hh"
#include "port.hh"

#define SETUP_ZIGZAG                                                         \
  using EqUint = typename psz::typing::UInt<sizeof(Eq)>::T;                  \
  using EqInt = typename psz::typing::Int<sizeof(Eq)>::T;                    \
  static_assert(                                                             \
      std::is_same<Eq, EqUint>::value, "Eq must be unsigned integer type."); \
  auto posneg_encode = [](EqInt x) -> EqUint {                               \
    return (2 * (x)) ^ ((x) >> (sizeof(Eq) * 8 - 1));                        \
  };

namespace psz {
namespace rolling_dp {

template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    int TileDim = 256, int Seq = 8, typename CompactVal = T,
    typename CompactIdx = uint32_t, typename CompactNum = uint32_t>
void c_lorenzo_1d1l(
    T* data, sycl::range<3> len3, sycl::range<3> stride3, int radius,
    Fp ebx2_r, Eq* eq, CompactVal* cval, CompactIdx* cidx, CompactNum* cn,
    const sycl::nd_item<3>& item_ct1, T* s_data,
    typename psz::typing::UInt<sizeof(Eq)>::T* s_eq_uint)
{
  constexpr auto NumThreads = TileDim / Seq;

  SETUP_ZIGZAG;

  T _thp_data[Seq + 1] = {0};
  auto prev = [&]() -> T& { return _thp_data[0]; };
  auto thp_data = [&](auto i) -> T& { return _thp_data[i + 1]; };

  auto id_base = item_ct1.get_group(2) * TileDim;

// dram.data to shmem.data
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + item_ct1.get_local_id(2) + ix * NumThreads;
    if (id < len3[2])
      s_data[item_ct1.get_local_id(2) + ix * NumThreads] =
          sycl::round(data[id] * ebx2_r);
  }
  /* DPCT1065 */
  item_ct1.barrier();
  item_ct1.barrier(sycl::access::fence_space::local_space);

// shmem.data to private.data
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++)
    thp_data(ix) = s_data[item_ct1.get_local_id(2) * Seq + ix];
  if (item_ct1.get_local_id(2) > 0)
    prev() = s_data[item_ct1.get_local_id(2) * Seq - 1];  // from last thread
  item_ct1.barrier(sycl::access::fence_space::local_space);

// quantize & write back to shmem.eq
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    T delta = thp_data(ix) - thp_data(ix - 1);
    bool quantizable = sycl::fabs(delta) < radius;
    T candidate = ZigZag ? delta : delta + radius;
    // otherwise, need to reset shared memory (to 0)
    if constexpr (ZigZag)
      s_eq_uint[ix + item_ct1.get_local_id(2) * Seq] =
          posneg_encode(quantizable * static_cast<EqInt>(candidate));
    else
      s_eq_uint[ix + item_ct1.get_local_id(2) * Seq] =
          quantizable * static_cast<EqUint>(candidate);
    if (not quantizable) {
      auto cur_idx =
          dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
              cn, 1);
      cidx[cur_idx] = id_base + item_ct1.get_local_id(2) * Seq + ix;
      cval[cur_idx] = candidate;
    }
  }
  item_ct1.barrier(sycl::access::fence_space::local_space);

// write from shmem.eq to dram.eq
#pragma unroll
  for (auto ix = 0; ix < Seq; ix++) {
    auto id = id_base + item_ct1.get_local_id(2) + ix * NumThreads;
    if (id < len3[2])
      eq[id] = s_eq_uint[item_ct1.get_local_id(2) + ix * NumThreads];
  }

  // end of kernel
}

template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, bool OneapiUseExperimental = true>
/* DPCT1110 */
void c_lorenzo_2d1l(
    T* data, sycl::range<3> len3, sycl::range<3> stride3, int radius,
    Fp ebx2_r, Eq* eq, CompactVal* cval, CompactIdx* cidx, CompactNum* cn,
    const sycl::nd_item<3>& item_ct1)
{
  constexpr auto TileDim = 16;
  constexpr auto Yseq = 8;

  SETUP_ZIGZAG;

  // NW  N       first el <- 0
  //  W  center
  T center[Yseq + 1] = {0};
  // auto prev = [&]() -> T& { return _center[0]; };
  // auto center = [&](auto i) -> T& { return _center[i + 1]; };
  // auto last = [&]() -> T& { return _center[Yseq]; };

  // BDX == TileDim == 16, BDY * Yseq = TileDim == 16
  auto gix = item_ct1.get_group(2) * TileDim + item_ct1.get_local_id(2);
  auto giy_base =
      item_ct1.get_group(1) * TileDim + item_ct1.get_local_id(1) * Yseq;
  auto g_id = [&](auto i) { return (giy_base + i) * stride3[1] + gix; };

  // use a warp as two half-warps
  // block_dim = (16, 2, 1) makes a full warp internally

// read to private.data (center)
#pragma unroll
  for (auto iy = 0; iy < Yseq; iy++) {
    if (gix < len3[2] and giy_base + iy < len3[1])
      center[iy + 1] = sycl::round(data[g_id(iy)] * ebx2_r);
  }
  // same-warp, next-16

  if constexpr (OneapiUseExperimental) {
    /* DPCT1108 */
    auto tmp = dpct::experimental::shift_sub_group_right(
        0xffffffff, item_ct1.get_sub_group(), center[Yseq], 16);
    if (item_ct1.get_local_id(1) == 1) center[0] = tmp;
  }
  else {
    /* DPCT1023 */
    /* DPCT1096 */
    auto tmp = dpct::shift_sub_group_right(
        item_ct1.get_sub_group(), center[Yseq], 16);
    if (item_ct1.get_local_id(1) == 1) center[0] = tmp;
  }

// prediction (apply Lorenzo filter)
#pragma unroll
  for (auto i = Yseq; i > 0; i--) {
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
      /* DPCT1023 */ /* DPCT1096 */
      auto west = dpct::shift_sub_group_right(
          item_ct1.get_sub_group(), center[i], 1, 16);
      if (item_ct1.get_local_id(2) > 0) center[i] -= west;  // delta
    }
  }
  /* DPCT1065 */
  item_ct1.barrier(sycl::access::fence_space::local_space);

#pragma unroll
  for (auto i = 1; i < Yseq + 1; i++) {
    auto gid = g_id(i - 1);

    if (gix < len3[2] and giy_base + (i - 1) < len3[1]) {
      bool quantizable = sycl::fabs(center[i]) < radius;
      T candidate = ZigZag ? center[i] : center[i] + radius;
      if constexpr (ZigZag)
        eq[gid] = posneg_encode(quantizable * static_cast<EqInt>(candidate));
      else
        eq[gid] = quantizable * static_cast<EqUint>(candidate);
      if (not quantizable) {
        auto cur_idx =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                cn, 1);
        cidx[cur_idx] = gid;
        cval[cur_idx] = candidate;
      }
    }
  }

  // end of kernel
}

template <
    typename T, bool ZigZag = false, typename Eq = uint32_t, typename Fp = T,
    typename CompactVal = T, typename CompactIdx = uint32_t,
    typename CompactNum = uint32_t, bool OneapiUseExperimental = true>
/* DPCT1110 register pressure */
void c_lorenzo_3d1l(
    T* data, sycl::range<3> len3, sycl::range<3> stride3, int radius,
    Fp ebx2_r, Eq* eq, CompactVal* cval, CompactIdx* cidx, CompactNum* cn,
    const sycl::nd_item<3>& item_ct1, sycl::local_accessor<T, 2> s)
{
  SETUP_ZIGZAG;

  constexpr auto TileDim = 8;

  T delta[TileDim + 1] = {0};  // first el = 0

  auto gix = [&]() {
    return item_ct1.get_group(2) * (TileDim * 4) + item_ct1.get_local_id(2);
  };
  auto giy = [&]() {
    return item_ct1.get_group(1) * TileDim + item_ct1.get_local_id(1);
  };
  auto giz_base = [&]() { return item_ct1.get_group(0) * TileDim; };
  auto base_id = gix() + giy() * stride3[1] + giz_base() * stride3[0];

  auto giz = [&](auto z) { return giz_base() + z; };
  auto gid = [&](auto z) { return base_id + z * stride3[0]; };

  auto load_prequant_3d = [&](const sycl::nd_item<3>& item_ct1) {
    if (gix() < len3[2] and giy() < len3[1]) {
      for (auto z = 0; z < TileDim; z++)
        if (giz(z) < len3[0])
          delta[z + 1] = sycl::round(data[gid(z)] * ebx2_r);
    }
    /* DPCT1065 */
    item_ct1.barrier();
    // item_ct1.barrier(sycl::access::fence_space::local_space);
  };

  auto quantize_compact_write = [&](T delta, auto x, auto y, auto z,
                                    auto gid) {
    bool quantizable = sycl::fabs(delta) < radius;
    T candidate;
    if constexpr (ZigZag) { candidate = delta; }
    else {
      candidate = delta + radius;
    }

    if (x < len3[2] and y < len3[1] and z < len3[0]) {
      if constexpr (ZigZag) {
        eq[gid] = posneg_encode(quantizable * static_cast<EqInt>(candidate));
      }
      else {
        eq[gid] = quantizable * static_cast<EqUint>(candidate);
      }

      if (not quantizable) {
        auto cur_idx =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                cn, 1);
        cidx[cur_idx] = gid;
        cval[cur_idx] = candidate;
      }
    }
  };

  ////////////////////////////////////////////////////////////////////////////

  load_prequant_3d(item_ct1);

  for (auto z = TileDim; z > 0; z--) {
    // z-direction
    delta[z] -= delta[z - 1];

    // x-direction
    if constexpr (OneapiUseExperimental) {
      /* DPCT1108 */
      auto prev_x = dpct::experimental::shift_sub_group_right(
          0xffffffff, item_ct1.get_sub_group(), delta[z], 1, 8);
      if (item_ct1.get_local_id(2) % TileDim > 0) delta[z] -= prev_x;
    }
    else {
      /* DPCT1023 */
      /* DPCT1096 */
      auto prev_x = dpct::shift_sub_group_right(
          item_ct1.get_sub_group(), delta[z], 1, 8);
      if (item_ct1.get_local_id(2) % TileDim > 0) delta[z] -= prev_x;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // y-direction, exchange via shmem
    // ghost padding along y
    s[item_ct1.get_local_id(1) + 1][item_ct1.get_local_id(2)] = delta[z];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    delta[z] -= (item_ct1.get_local_id(1) > 0) *
                s[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)];

    // now delta[z] is delta
    quantize_compact_write(delta[z], gix(), giy(), giz(z - 1), gid(z - 1));
    item_ct1.barrier();
  }
}

}  // namespace rolling_dp
}  // namespace psz

#endif /* F0EAE3AE_EDAC_46DC_B1C2_080FA2725740 */
