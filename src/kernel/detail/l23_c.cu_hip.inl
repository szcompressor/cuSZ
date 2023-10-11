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

#include "cusz/suint.hh"
#include "port.hh"
#include "subr.cu_hip.inl"

namespace psz {
namespace cuda_hip {
namespace __kernel {

/**
 * @deprecated
 */
template <typename T, typename Eq, typename FP, int BLOCK, int SEQ>
__global__ void c_lorenzo_1d1l(
    T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, Eq* eq,
    T* outlier)
{
  namespace subr_v0 = psz::cuda_hip;

  constexpr auto NTHREAD = BLOCK / SEQ;

  __shared__ T scratch[BLOCK];  // for data and outlier
  __shared__ Eq s_eq[BLOCK];

  T prev{0};
  T thp_data[SEQ];

  auto id_base = blockIdx.x * BLOCK;

  subr_v0::load_prequant_1d<T, FP, NTHREAD, SEQ>(
      data, len3.x, id_base, scratch, thp_data, prev, ebx2_r);
  subr_v0::predict_quantize_1d<T, Eq, SEQ, true>(
      thp_data, s_eq, scratch, radius, prev);
  subr_v0::predict_quantize_1d<T, Eq, SEQ, false>(
      thp_data, s_eq, scratch, radius);
  subr_v0::write_1d<Eq, T, NTHREAD, SEQ, false>(
      s_eq, scratch, len3.x, id_base, eq, outlier);
}

/**
 * @deprecated
 */
template <typename T, typename Eq, typename FP>
__global__ void c_lorenzo_2d1l(
    T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, Eq* eq,
    T* outlier)
{
  namespace subr_v0 = psz::cuda_hip;

  constexpr auto BLOCK = 16;
  constexpr auto YSEQ = 8;

  T center[YSEQ + 1] = {0};  // NW  N       first element <- 0
                             //  W  center

  auto gix = blockIdx.x * BLOCK + threadIdx.x;  // BDX == BLOCK == 16
  auto giy_base =
      blockIdx.y * BLOCK + threadIdx.y * YSEQ;  // BDY * YSEQ = BLOCK == 16

  subr_v0::load_prequant_2d<T, FP, YSEQ>(
      data, len3.x, gix, len3.y, giy_base, stride3.y, ebx2_r, center);
  subr_v0::predict_2d<T, Eq, YSEQ>(center);
  subr_v0::quantize_write_2d<T, Eq, YSEQ>(
      center, len3.x, gix, len3.y, giy_base, stride3.y, radius, eq, outlier);
}

// 16x16 data block maps to 16x2 (one warp) thread block

/**
 * @deprecated
 */
template <typename T, typename Eq, typename FP>
__global__ void x_lorenzo_2d1l(  //
    Eq* eq, T* outlier, dim3 len3, dim3 stride3, int radius, FP ebx2, T* xdata)
{
  namespace subr_v0 = psz::cuda_hip;

  constexpr auto BLOCK = 16;
  constexpr auto YSEQ = BLOCK / 2;  // sequentiality in y direction
  static_assert(BLOCK == 16, "In one case, we need BLOCK for 2D == 16");

  __shared__ T scratch[BLOCK];  // TODO use warp shuffle to eliminate this
  T thread_private[YSEQ];

  auto gix = blockIdx.x * BLOCK + threadIdx.x;
  auto giy_base =
      blockIdx.y * BLOCK + threadIdx.y * YSEQ;  // BDY * YSEQ = BLOCK == 16

  auto get_gid = [&](auto i) { return (giy_base + i) * stride3.y + gix; };

  subr_v0::load_fuse_2d<T, Eq, YSEQ>(
      eq, outlier, len3.x, gix, len3.y, giy_base, stride3.y, radius,
      thread_private);
  subr_v0::block_scan_2d<T, Eq, FP, YSEQ>(thread_private, scratch, ebx2);
  subr_v0::decomp_write_2d<T, YSEQ>(
      thread_private, len3.x, gix, len3.y, giy_base, stride3.y, xdata);
}

template <typename T, typename Eq, typename FP>
__global__ void c_lorenzo_3d1l_legacy(
    T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, Eq* eq,
    T* outlier)
{
  constexpr auto BLOCK = 8;
  __shared__ T s[8][8][32];

  auto z = threadIdx.z;

  auto gix = blockIdx.x * (BLOCK * 4) + threadIdx.x;
  auto giy_base = blockIdx.y * BLOCK;
  auto giz = blockIdx.z * BLOCK + z;
  auto base_id = gix + giy_base * stride3.y + giz * stride3.z;

  auto giy = [&](auto y) { return giy_base + y; };
  auto gid = [&](auto y) { return base_id + y * stride3.y; };

  auto load_prequant_3d = [&]() {
    if (gix < len3.x and giz < len3.z) {
      for (auto y = 0; y < BLOCK; y++)
        if (giy(y) < len3.y)
          s[z][y][threadIdx.x] =
              round(data[gid(y)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_write = [&](T delta, auto x, auto y, auto z, auto gid) {
    bool quantizable = fabs(delta) < radius;
    T candidate = delta + radius;
    if (x < len3.x and y < len3.y and z < len3.z) {
      eq[gid] = quantizable * static_cast<Eq>(candidate);
      outlier[gid] = (not quantizable) * candidate;
    }
  };

  auto x = threadIdx.x % 8;

  auto predict_3d = [&](auto y) {
    T delta =
        s[z][y][threadIdx.x] -  //
        ((z > 0 and y > 0 and x > 0 ? s[z - 1][y - 1][threadIdx.x - 1]
                                    : 0)                         // dist=3
         - (y > 0 and x > 0 ? s[z][y - 1][threadIdx.x - 1] : 0)  // dist=2
         - (z > 0 and x > 0 ? s[z - 1][y][threadIdx.x - 1] : 0)  //
         - (z > 0 and y > 0 ? s[z - 1][y - 1][threadIdx.x] : 0)  //
         + (x > 0 ? s[z][y][threadIdx.x - 1] : 0)                // dist=1
         + (y > 0 ? s[z][y - 1][threadIdx.x] : 0)                //
         + (z > 0 ? s[z - 1][y][threadIdx.x] : 0));              //
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
__global__ void c_lorenzo_3d1l(
    T* data, dim3 len3, dim3 stride3, int radius, FP ebx2_r, Eq* eq,
    T* outlier)
{
  constexpr auto BLOCK = 8;
  __shared__ T s[9][33];
  T delta[BLOCK + 1] = {0};  // first el = 0

  const auto gix = blockIdx.x * (BLOCK * 4) + threadIdx.x;
  const auto giy = blockIdx.y * BLOCK + threadIdx.y;
  const auto giz_base = blockIdx.z * BLOCK;
  const auto base_id = gix + giy * stride3.y + giz_base * stride3.z;

  auto giz = [&](auto z) { return giz_base + z; };
  auto gid = [&](auto z) { return base_id + z * stride3.z; };

  auto load_prequant_3d = [&]() {
    if (gix < len3.x and giy < len3.y) {
      for (auto z = 0; z < BLOCK; z++)
        if (giz(z) < len3.z)
          delta[z + 1] =
              round(data[gid(z)] * ebx2_r);  // prequant (fp presence)
    }
    __syncthreads();
  };

  auto quantize_write = [&](T delta, auto x, auto y, auto z, auto gid) {
    bool quantizable = fabs(delta) < radius;
    T candidate = delta + radius;
    if (x < len3.x and y < len3.y and z < len3.z) {
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

  load_prequant_3d();

  for (auto z = BLOCK; z > 0; z--) {
    // z-direction
    delta[z] -= delta[z - 1];

    // x-direction
    auto prev_x = __shfl_up_sync(0xffffffff, delta[z], 1, 8);
    if (threadIdx.x % BLOCK > 0) delta[z] -= prev_x;

    // y-direction, exchange via shmem
    // ghost padding along y
    s[threadIdx.y + 1][threadIdx.x] = delta[z];
    __syncthreads();

    delta[z] -= (threadIdx.y > 0) * s[threadIdx.y][threadIdx.x];

    // now delta[z] is delta
    quantize_write(delta[z], gix, giy, giz(z - 1), gid(z - 1));
    __syncthreads();
  }
}

}  // namespace __kernel
}  // namespace cuda_hip
}  // namespace psz