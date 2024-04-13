#include <cmath>
#include <dpct/dpct.hpp>
#include <sycl/sycl.hpp>

#include "cusz/suint.hh"
#include "port.hh"
#include "subr.dp.inl"

namespace psz {
namespace dpcpp {
namespace __kernel {

/**
 * @deprecated
 */
template <typename T, typename Eq, typename FP, int BLOCK, int SEQ>
void c_lorenzo_1d1l(
    T* data, sycl::range<3> len3, sycl::range<3> stride3, int radius,
    FP ebx2_r, Eq* eq, T* outlier, const sycl::nd_item<3>& item_ct1,
    T* scratch, Eq* s_eq)
{
  namespace subr_v0 = psz::dpcpp;

  constexpr auto NTHREAD = BLOCK / SEQ;

  // for data and outlier

  T prev{0};
  T thp_data[SEQ];

  auto id_base = item_ct1.get_group(2) * BLOCK;

  subr_v0::load_prequant_1d<T, FP, NTHREAD, SEQ>(
      data, len3[2], id_base, scratch, thp_data, prev, ebx2_r, item_ct1);
  subr_v0::predict_quantize_1d<T, Eq, SEQ, true>(
      thp_data, s_eq, scratch, radius, item_ct1, prev);
  subr_v0::predict_quantize_1d<T, Eq, SEQ, false>(
      thp_data, s_eq, scratch, radius, item_ct1);
  subr_v0::write_1d<Eq, T, NTHREAD, SEQ, false>(
      s_eq, scratch, len3[2], id_base, eq, outlier, item_ct1);
}

/**
 * @deprecated
 */
template <typename T, typename Eq, typename FP>
void c_lorenzo_2d1l(
    T* data, sycl::range<3> len3, sycl::range<3> stride3, int radius,
    FP ebx2_r, Eq* eq, T* outlier, const sycl::nd_item<3>& item_ct1)
{
  namespace subr_v0 = psz::dpcpp;

  constexpr auto BLOCK = 16;
  constexpr auto YSEQ = 8;

  T center[YSEQ + 1] = {0};  // NW  N       first element <- 0
                             //  W  center

  auto gix = item_ct1.get_group(2) * BLOCK +
             item_ct1.get_local_id(2);  // BDX == BLOCK == 16
  auto giy_base = item_ct1.get_group(1) * BLOCK +
                  item_ct1.get_local_id(1) * YSEQ;  // BDY * YSEQ = BLOCK == 16

  subr_v0::load_prequant_2d<T, FP, YSEQ>(
      data, len3[2], gix, len3[1], giy_base, stride3[1], ebx2_r, center,
      item_ct1);
  subr_v0::predict_2d<T, Eq, YSEQ>(center, item_ct1);
  subr_v0::quantize_write_2d<T, Eq, YSEQ>(
      center, len3[2], gix, len3[1], giy_base, stride3[1], radius, eq,
      outlier);
}

// 16x16 data block maps to 16x2 (one warp) thread block

/**
 * @deprecated
 */
template <typename T, typename Eq, typename FP>
void x_lorenzo_2d1l(  //
    Eq* eq, T* outlier, sycl::range<3> len3, sycl::range<3> stride3,
    int radius, FP ebx2, T* xdata, const sycl::nd_item<3>& item_ct1,
    T* scratch)
{
  namespace subr_v0 = psz::dpcpp;

  constexpr auto BLOCK = 16;
  constexpr auto YSEQ = BLOCK / 2;  // sequentiality in y direction
  static_assert(BLOCK == 16, "In one case, we need BLOCK for 2D == 16");

  // TODO use warp shuffle to eliminate this
  T thread_private[YSEQ];

  auto gix = item_ct1.get_group(2) * BLOCK + item_ct1.get_local_id(2);
  auto giy_base = item_ct1.get_group(1) * BLOCK +
                  item_ct1.get_local_id(1) * YSEQ;  // BDY * YSEQ = BLOCK == 16

  auto get_gid = [&](auto i) { return (giy_base + i) * stride3[1] + gix; };

  subr_v0::load_fuse_2d<T, Eq, YSEQ>(
      eq, outlier, len3[2], gix, len3[1], giy_base, stride3[1], radius,
      thread_private);
  subr_v0::block_scan_2d<T, Eq, FP, YSEQ>(
      thread_private, scratch, ebx2, item_ct1);
  subr_v0::decomp_write_2d<T, YSEQ>(
      thread_private, len3[2], gix, len3[1], giy_base, stride3[1], xdata);
}

template <typename T, typename Eq, typename FP>
void c_lorenzo_3d1l_legacy(
    T* data, sycl::range<3> len3, sycl::range<3> stride3, int radius,
    FP ebx2_r, Eq* eq, T* outlier, const sycl::nd_item<3>& item_ct1,
    sycl::local_accessor<T, 3> s)
{
  constexpr auto BLOCK = 8;

  auto z = item_ct1.get_local_id(0);

  auto gix = item_ct1.get_group(2) * (BLOCK * 4) + item_ct1.get_local_id(2);
  auto giy_base = item_ct1.get_group(1) * BLOCK;
  auto giz = item_ct1.get_group(0) * BLOCK + z;
  auto base_id = gix + giy_base * stride3[1] + giz * stride3[0];

  auto giy = [&](auto y) { return giy_base + y; };
  auto gid = [&](auto y) { return base_id + y * stride3[1]; };

  auto load_prequant_3d = [&](const sycl::nd_item<3>& item_ct1) {
    if (gix < len3[2] and giz < len3[0]) {
      for (auto y = 0; y < BLOCK; y++)
        if (giy(y) < len3[1])
          s[z][y][item_ct1.get_local_id(2)] =
              round(data[gid(y)] * ebx2_r);  // prequant (fp presence)
    }
    /*
    DPCT1065:16: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  };

  auto quantize_write = [&](T delta, auto x, auto y, auto z, auto gid) {
    bool quantizable = fabs(delta) < radius;
    T candidate = delta + radius;
    if (x < len3[2] and y < len3[1] and z < len3[0]) {
      eq[gid] = quantizable * static_cast<Eq>(candidate);
      outlier[gid] = (not quantizable) * candidate;
    }
  };

  auto x = item_ct1.get_local_id(2) % 8;

  auto predict_3d = [&](auto y, const sycl::nd_item<3>& item_ct1) {
    T delta =
        s[z][y][item_ct1.get_local_id(2)] -  //
        ((z > 0 and y > 0 and x > 0
              ? s[z - 1][y - 1][item_ct1.get_local_id(2) - 1]
              : 0)  // dist=3
         - (y > 0 and x > 0 ? s[z][y - 1][item_ct1.get_local_id(2) - 1]
                            : 0)  // dist=2
         -
         (z > 0 and x > 0 ? s[z - 1][y][item_ct1.get_local_id(2) - 1] : 0)  //
         -
         (z > 0 and y > 0 ? s[z - 1][y - 1][item_ct1.get_local_id(2)] : 0)  //
         + (x > 0 ? s[z][y][item_ct1.get_local_id(2) - 1] : 0)    // dist=1
         + (y > 0 ? s[z][y - 1][item_ct1.get_local_id(2)] : 0)    //
         + (z > 0 ? s[z - 1][y][item_ct1.get_local_id(2)] : 0));  //
    return delta;
  };

  ////////////////////////////////////////////////////////////////////////////

  load_prequant_3d();
  for (auto y = 0; y < BLOCK; y++) {
    auto delta = predict_3d(y);
    quantize_write(delta, gix, giy(y), giz, gid(y));
  }
}

/**
 * @deprecated
 */
template <typename T, typename Eq, typename FP>
/*
DPCT1110:20: The total declared local variable size in device function
c_lorenzo_3d1l exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void c_lorenzo_3d1l(
    T* data, sycl::range<3> len3, sycl::range<3> stride3, int radius,
    FP ebx2_r, Eq* eq, T* outlier, const sycl::nd_item<3>& item_ct1,
    sycl::local_accessor<T, 2> s)
{
  constexpr auto BLOCK = 8;

  T delta[BLOCK + 1] = {0};  // first el = 0

  const auto gix =
      item_ct1.get_group(2) * (BLOCK * 4) + item_ct1.get_local_id(2);
  const auto giy = item_ct1.get_group(1) * BLOCK + item_ct1.get_local_id(1);
  const auto giz_base = item_ct1.get_group(0) * BLOCK;
  const auto base_id = gix + giy * stride3[1] + giz_base * stride3[0];

  auto giz = [&](auto z) { return giz_base + z; };
  auto gid = [&](auto z) { return base_id + z * stride3[0]; };

  auto load_prequant_3d = [&](const sycl::nd_item<3>& item_ct1) {
    if (gix < len3[2] and giy < len3[1]) {
      for (auto z = 0; z < BLOCK; z++)
        if (giz(z) < len3[0])
          delta[z + 1] =
              sycl::round(data[gid(z)] * ebx2_r);  // prequant (fp presence)
    }
    /*
    DPCT1065:17: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  };

  auto quantize_write = [&](T delta, auto x, auto y, auto z, auto gid) {
    bool quantizable = sycl::fabs(delta) < radius;
    T candidate = delta + radius;
    if (x < len3[2] and y < len3[1] and z < len3[0]) {
      eq[gid] = quantizable * static_cast<Eq>(candidate);
      outlier[gid] = (not quantizable) * candidate;
    }
  };

  ////////////////////////////////////////////////////////////////////////////

  /* z-direction, sequential in private buffer
     delta = + (s[z][y][x] - s[z-1][y][x])
             - (s[z][y][x-1] - s[z-1][y][x-1])
             + (s[z][y-1][x-1] - s[z-1][y-1][x-1])
             - (s[z][y-1][x] - s[z-1][y-1][x])

     x-direction, shuffle
     delta = + (s[z][y][x] - s[z][y][x-1])
             - (s[z][y-1][x] - s[z][y-1][x-1])

     y-direction, shmem
     delta = s[z][y][x] - s[z][y-1][x]
   */

  load_prequant_3d(item_ct1);

  for (auto z = BLOCK; z > 0; z--) {
    // z-direction
    delta[z] -= delta[z - 1];

    // x-direction
    /*
    DPCT1023:21: The SYCL sub-group does not support mask options for
    dpct::shift_sub_group_right. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_up_sync.
    */
    /*
    DPCT1096:29: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::shift_sub_group_right" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    auto prev_x =
        dpct::shift_sub_group_right(item_ct1.get_sub_group(), delta[z], 1, 8);
    if (item_ct1.get_local_id(2) % BLOCK > 0) delta[z] -= prev_x;

    // y-direction, exchange via shmem
    // ghost padding along y
    s[item_ct1.get_local_id(1) + 1][item_ct1.get_local_id(2)] = delta[z];
    /*
    DPCT1065:18: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    delta[z] -= (item_ct1.get_local_id(1) > 0) *
                s[item_ct1.get_local_id(1)][item_ct1.get_local_id(2)];

    // now delta[z] is delta
    quantize_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    /*
    DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }
}

}  // namespace __kernel
}  // namespace dpcpp
}  // namespace psz