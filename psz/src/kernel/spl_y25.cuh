// Authors: Jinyang Liu, Shixun Wu, Jiannan Tian

#ifndef CUSZ_KERNEL_SPLINE_Y25_CUH
#define CUSZ_KERNEL_SPLINE_Y25_CUH

#include <cstdint>
#include <cstdio>
#include <tuple>

#include "cusz/type.h"
#include "utils/err.hh"

constexpr auto Spl3_Comp = true;
constexpr auto Spl3_Decomp = false;
constexpr auto Spl3_PredAtt = true;
constexpr auto Spl3_AbAtt = false;

#define TIX threadIdx.x
#define TIY threadIdx.y
#define TIZ threadIdx.z
#define BIX blockIdx.x
#define BIY blockIdx.y
#define BIZ blockIdx.z
#define BDX blockDim.x
#define BDY blockDim.y
#define BDZ blockDim.z
#define GDX gridDim.x
#define GDY gridDim.y
#define GDZ gridDim.z

constexpr int Blk16 = 16;
constexpr int Blk17 = 17;
constexpr int BlkDimLin = 384;
constexpr int DefaultLinBlkSz = BlkDimLin;

namespace psz {

template <
    typename T, int SplDim, int ProfBlkSzX, int ProfBlkSzY, int ProfBlkSzZ, int ProfNBlkX,
    int ProfNBlkY, int ProfNBlkZ, int LinBlkSz>
__global__ void KCU_c_spl_prof_data(T* data, dim3 data_size, dim3 data_leap, T* errors);

template <typename T, int SplDim, int ProfNBlkX, int ProfNBlkY, int ProfNBlkZ, int LinBlkSz>
__global__ void KCU_c_spl_prof_data_2(T* data, dim3 data_size, dim3 data_leap, T* errors);

template <
    typename T, typename E, typename FP = float, int LEVEL = 4, int SplDim = 2, int AncBlkSzX = 8,
    int AncBlkSzY = 8, int AncBlkSzZ = 1, int NAncBlkX = 4, int NAncBlkY = 1, int NAncBlkZ = 1,
    int LinBlkSz = DefaultLinBlkSz, typename CompactValIdx = void*,
    typename CompactNum = uint32_t*>
__global__ void KCU_c_spl_infprecis_data(
    T*, dim3, dim3, E*, dim3, dim3, T*, dim3, CompactValIdx, CompactNum, FP, FP, int,
    INTERP_PARAMS);

template <
    typename E, typename T, typename FP = float, int LEVEL = 4, int SplDim = 2, int AncBlkSzX = 8,
    int AncBlkSzY = 8, int AncBlkSzZ = 1, int NAncBlkX = 4, int NAncBlkY = 1, int NAncBlkZ = 1,
    int LinBlkSz = DefaultLinBlkSz>
__global__ void KCU_x_spl_infprecis_data(
    E* eq, dim3 eq_size, dim3 eq_leap, T* anchor, dim3 anchor_size, dim3 anchor_leap, T* data,
    dim3 data_size, dim3 data_leap, T* outlier_tmp, FP eb_r, FP ebx2, int radius,
    INTERP_PARAMS intp_param);

template <typename T>
__global__ void reset_errors(T* errors);

template <
    typename T, typename FP, int LEVEL = 4, int SplDim = 2, int AncBlkSzX = 8, int AncBlkSzY = 8,
    int AncBlkSzZ = 1, int NAncBlkX = 4, int NAncBlkY = 1, int NAncBlkZ = 1,
    int LinBlkSz = DefaultLinBlkSz>
__global__ void KCU_pa_spl_infprecis_data(
    T* data, dim3 data_size, dim3 data_leap, dim3 sample_starts, dim3 sample_block_grid_sizes,
    dim3 sample_strides, FP eb_r, FP ebx2, INTERP_PARAMS intp_param, T* errors,
    bool workflow = Spl3_PredAtt);

template <
    typename T, int SplDim = 3, int ProfBlkSzX = 4, int ProfBlkSzY = 4, int ProfBlkSzZ = 4,
    int ProfNBlkX = 4, int ProfNBlkY = 4, int ProfNBlkZ = 4, int LinBlkSz = DefaultLinBlkSz>
__device__ void auto_tuning(
    T s_data[ProfBlkSzZ * ProfNBlkZ][ProfBlkSzY * ProfNBlkY][ProfBlkSzX * ProfNBlkX],
    T local_errs[6], dim3 data_size, T* count);

template <
    typename T, int SplDim = 3, int ProfNBlkX = 4, int ProfNBlkY = 4, int ProfNBlkZ = 4,
    int LinBlkSz = DefaultLinBlkSz>
__device__ void auto_tuning_2(
    T s_data[ProfNBlkX * ProfNBlkY * ProfNBlkZ], T s_nx[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4],
    T s_ny[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4], T s_nz[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4],
    T local_errs[6], dim3 data_size, T* count);

template <
    typename T1, typename T2, typename FP, int LEVEL, int SplDim = 2, int AncBlkSzX = 8,
    int AncBlkSzY = 8, int AncBlkSzZ = 1, int NAncBlkX = 4, int NAncBlkY = 1, int NAncBlkZ = 1,
    int LinBlkSz = DefaultLinBlkSz, bool Workflow = Spl3_Comp, bool PROBE_PRED_ERROR = false>
__device__ void spline_layout_interpolate(
    T1 s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
             [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    T2 s_eq[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
           [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    dim3 data_size, FP eb_r, FP ebx2, int radius, INTERP_PARAMS intp_param);

template <
    typename T, typename FP, int LEVEL, int SplDim, int AncBlkSzX, int AncBlkSzY, int AncBlkSzZ,
    int NAncBlkX, int NAncBlkY, int NAncBlkZ, int LinBlkSz, bool Workflow>
__device__ void spline_layout_interpolate_att(
    T s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
            [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    dim3 data_size, dim3 g_starts, FP eb_r, FP ebx2, uint8_t level, INTERP_PARAMS intp_param,
    T* error);

}  // namespace psz

namespace {

template <
    int SplDim, int AncBlkSzX, int AncBlkSzY, int AncBlkSzZ, int NAncBlkX, int NAncBlkY,
    int NAncBlkZ, bool INCLUSIVE = true>
__forceinline__ __device__ bool xyz_predicate(
    unsigned int x, unsigned int y, unsigned int z, const dim3& data_size)
{
  if constexpr (INCLUSIVE) {  //

    return (x <= (AncBlkSzX * NAncBlkX) and y <= (AncBlkSzY * NAncBlkY) and
            z <= (AncBlkSzZ * NAncBlkZ)) and
           BIX * (AncBlkSzX * NAncBlkX) + x < data_size.x and
           BIY * (AncBlkSzY * NAncBlkY) + y < data_size.y and
           BIZ * (AncBlkSzZ * NAncBlkZ) + z < data_size.z;
  }
  else {
    return x < (AncBlkSzX * NAncBlkX) + (BIX == GDX - 1) * (SplDim <= 1) and
           y < (AncBlkSzY * NAncBlkY) + (BIY == GDY - 1) * (SplDim <= 2) and
           z < (AncBlkSzZ * NAncBlkZ) + (BIZ == GDZ - 1) * (SplDim <= 3) and
           BIX * (AncBlkSzX * NAncBlkX) + x < data_size.x and
           BIY * (AncBlkSzY * NAncBlkY) + y < data_size.y and
           BIZ * (AncBlkSzZ * NAncBlkZ) + z < data_size.z;
  }
}

template <
    int SplDim, int AncBlkSzX, int AncBlkSzY, int AncBlkSzZ, int NAncBlkX, int NAncBlkY,
    int NAncBlkZ, bool INCLUSIVE = true>
__forceinline__ __device__ bool xyz_predicate_att(
    unsigned int x, unsigned int y, unsigned int z, const dim3& data_size, const dim3& g_starts)
{
  if constexpr (INCLUSIVE) {
    return (x <= (AncBlkSzX * NAncBlkX) and y <= (AncBlkSzY * NAncBlkY) and
            z <= (AncBlkSzZ * NAncBlkZ)) and
           g_starts.x + x < data_size.x and g_starts.y + y < data_size.y and
           g_starts.z + z < data_size.z;
  }
  else {
    return x < (AncBlkSzX * NAncBlkX) + (BIX == GDX - 1) and
           y < (AncBlkSzY * NAncBlkY) + (BIY == GDY - 1) and
           z < (AncBlkSzZ * NAncBlkZ) + (BIZ == GDZ - 1) and g_starts.x + x < data_size.x and
           g_starts.y + y < data_size.y and g_starts.z + z < data_size.z;
  }
}

template <
    typename T1, typename T2, int SplDim, int AncBlkSzX, int AncBlkSzY, int AncBlkSzZ,
    int NAncBlkX, int NAncBlkY, int NAncBlkZ, int LinBlkSz = DefaultLinBlkSz>
__device__ void c_reset_scratch_data(
    T1 s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
             [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    T2 s_eq[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
           [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    int radius)
{
  for (auto _tix = TIX;
       _tix < (AncBlkSzX * NAncBlkX + (SplDim >= 1)) * (AncBlkSzY * NAncBlkY + (SplDim >= 2)) *
                  (AncBlkSzZ * NAncBlkZ + (SplDim >= 3));
       _tix += LinBlkSz) {
    auto x = (_tix % (AncBlkSzX * NAncBlkX + (SplDim >= 1)));
    auto y =
        (_tix / (AncBlkSzX * NAncBlkX + (SplDim >= 1))) % (AncBlkSzY * NAncBlkY + (SplDim >= 2));
    auto z =
        (_tix / (AncBlkSzX * NAncBlkX + (SplDim >= 1))) / (AncBlkSzY * NAncBlkY + (SplDim >= 2));

    s_data[z][y][x] = 0;
    if (x % AncBlkSzX == 0 and y % AncBlkSzY == 0 and z % AncBlkSzZ == 0) s_eq[z][y][x] = radius;
  }
  __syncthreads();
}

template <
    typename T, int SplDim = 3, int ProfBlkSzX = 4, int ProfBlkSzY = 4, int ProfBlkSzZ = 4,
    int ProfNBlkX = 4, int ProfNBlkY = 4, int ProfNBlkZ = 4, int LinBlkSz = DefaultLinBlkSz>
__device__ void c_reset_scratch_profiling_data(
    T s_data[ProfBlkSzZ * ProfNBlkZ][ProfBlkSzY * ProfNBlkY][ProfBlkSzX * ProfNBlkX],
    T default_value)
{
  auto x_size = ProfBlkSzX * ProfNBlkX;
  auto y_size = ProfBlkSzY * ProfNBlkY;
  auto z_size = ProfBlkSzZ * ProfNBlkZ;
  for (auto _tix = TIX; _tix < x_size * y_size * z_size; _tix += LinBlkSz) {
    auto x = (_tix % x_size);
    auto y = (_tix / x_size) % y_size;
    auto z = (_tix / x_size) / y_size;
    s_data[z][y][x] = default_value;
  }
}

template <
    typename T, int SplDim = 3, int ProfNBlkX = 4, int ProfNBlkY = 4, int ProfNBlkZ = 4,
    int LinBlkSz = DefaultLinBlkSz>
__device__ void c_reset_scratch_profiling_data_2(
    T s_data[ProfNBlkX * ProfNBlkY * ProfNBlkZ], T nx[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4],
    T ny[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4], T nz[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4],
    T default_value)
{
  for (auto _tix = TIX; _tix < ProfNBlkX * ProfNBlkY * ProfNBlkZ * 4; _tix += LinBlkSz) {
    auto offset = (_tix % 4);
    auto idx = _tix / 4;
    nx[idx][offset] = ny[idx][offset] = nz[idx][offset] = default_value;
    // s_data[TIX] = default_value;
    s_data[idx] = default_value;
  }
}

template <
    typename T1, int AncBlkSzX, int AncBlkSzY, int AncBlkSzZ, int NAncBlkX, int NAncBlkY,
    int NAncBlkZ, int LinBlkSz = DefaultLinBlkSz>
__device__ void c_gather_anchor(
    T1* data, dim3 data_size, dim3 data_leap, T1* anchor, dim3 anchor_leap, dim3 begin)
{
  auto ax = begin.x / (AncBlkSzX * NAncBlkX) + BIX;  // global anchor index
  auto ay = begin.y / (AncBlkSzY * NAncBlkY) + BIY;
  auto az = begin.z / (AncBlkSzZ * NAncBlkZ) + BIZ;
  // 2d bug may be here!
  auto x = (AncBlkSzX * NAncBlkX) * ax;
  auto y = (AncBlkSzY * NAncBlkY) * ay;
  auto z = (AncBlkSzZ * NAncBlkZ) * az;

  bool pred1 = TIX < 1;  // 1 is num of anchor
  bool pred2 = x < data_size.x and y < data_size.y and z < data_size.z;

  if (pred1 and pred2) {
    auto data_id = x + y * data_leap.y + z * data_leap.z;
    auto anchor_id = ax + ay * anchor_leap.y + az * anchor_leap.z;
    anchor[anchor_id] = data[data_id];
  }
  __syncthreads();
}

template <
    typename T1, typename T2 = T1, int SplDim = 2, int AncBlkSzX = 8, int AncBlkSzY = 8,
    int AncBlkSzZ = 8, int NAncBlkX = 4, int NAncBlkY = 1, int NAncBlkZ = 1,
    int LinBlkSz = DefaultLinBlkSz>
__device__ void x_reset_scratch_data(
    T1 s_xdata[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
              [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    T2 s_eq[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
           [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    T1* anchor, dim3 anchor_size, dim3 anchor_leap, dim3 begin)
{
  for (auto _tix = TIX;
       _tix < (AncBlkSzX * NAncBlkX + (SplDim >= 1)) * (AncBlkSzY * NAncBlkY + (SplDim >= 2)) *
                  (AncBlkSzZ * NAncBlkZ + (SplDim >= 3));
       _tix += LinBlkSz) {
    auto x = (_tix % (AncBlkSzX * NAncBlkX + (SplDim >= 1)));
    auto y =
        (_tix / (AncBlkSzX * NAncBlkX + (SplDim >= 1))) % (AncBlkSzY * NAncBlkY + (SplDim >= 2));
    auto z =
        (_tix / (AncBlkSzX * NAncBlkX + (SplDim >= 1))) / (AncBlkSzY * NAncBlkY + (SplDim >= 2));

    s_eq[z][y][x] = 0;  // TODO explicitly handle zero-padding
    /*****************************************************************************
     okay to use
     ******************************************************************************/
    if (x % AncBlkSzX == 0 and y % AncBlkSzY == 0 and z % AncBlkSzZ == 0) {
      s_xdata[z][y][x] = 0;

      auto ax = (begin.x / AncBlkSzX + (x / AncBlkSzX) + BIX * NAncBlkX);
      auto ay = (begin.y / AncBlkSzY + (y / AncBlkSzY) + BIY * NAncBlkY);
      auto az = (begin.z / AncBlkSzZ + (z / AncBlkSzZ) + BIZ * NAncBlkZ);

      if (ax < anchor_size.x and ay < anchor_size.y and az < anchor_size.z)
        s_xdata[z][y][x] = anchor[ax + ay * anchor_leap.y + az * anchor_leap.z];
    }
  }

  __syncthreads();
}

template <
    typename T1, typename T2, int SplDim = 2, int AncBlkSzX = 8, int AncBlkSzY = 8,
    int AncBlkSzZ = 8, int NAncBlkX = 4, int NAncBlkY = 1, int NAncBlkZ = 1,
    int LinBlkSz = DefaultLinBlkSz>
__device__ void global2shmem_data(
    T1* data, dim3 data_size, dim3 data_leap, dim3 begin,
    T2 s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
             [AncBlkSzX * NAncBlkX + (SplDim >= 1)])
{
  constexpr auto TOTAL = (AncBlkSzX * NAncBlkX + (SplDim >= 1)) *
                         (AncBlkSzY * NAncBlkY + (SplDim >= 2)) *
                         (AncBlkSzZ * NAncBlkZ + (SplDim >= 3));

  for (auto _tix = TIX; _tix < TOTAL; _tix += LinBlkSz) {
    auto x = (_tix % (AncBlkSzX * NAncBlkX + (SplDim >= 1)));
    auto y =
        (_tix / (AncBlkSzX * NAncBlkX + (SplDim >= 1))) % (AncBlkSzY * NAncBlkY + (SplDim >= 2));
    auto z =
        (_tix / (AncBlkSzX * NAncBlkX + (SplDim >= 1))) / (AncBlkSzY * NAncBlkY + (SplDim >= 2));
    auto gx = (begin.x + x + BIX * (AncBlkSzX * NAncBlkX));
    auto gy = (begin.y + y + BIY * (AncBlkSzY * NAncBlkY));
    auto gz = (begin.z + z + BIZ * (AncBlkSzZ * NAncBlkZ));
    auto gid = gx + gy * data_leap.y + gz * data_leap.z;

    if (gx < data_size.x and gy < data_size.y and gz < data_size.z) s_data[z][y][x] = data[gid];
  }
  __syncthreads();
}

template <
    typename T1, typename T2, int SplDim = 3, int ProfBlkSzX = 4, int ProfBlkSzY = 4,
    int ProfBlkSzZ = 4, int ProfNBlkX = 4, int ProfNBlkY = 4, int ProfNBlkZ = 4,
    int LinBlkSz = DefaultLinBlkSz>
__device__ void global2shmem_profiling_data(
    T1* data, dim3 data_size, dim3 data_leap,
    T2 s_data[ProfBlkSzZ * ProfNBlkZ][ProfBlkSzY * ProfNBlkY][ProfBlkSzX * ProfNBlkX])
{
  constexpr auto x_size = ProfBlkSzX * ProfNBlkX;
  constexpr auto y_size = ProfBlkSzY * ProfNBlkY;
  constexpr auto z_size = ProfBlkSzZ * ProfNBlkZ;
  constexpr auto TOTAL = x_size * y_size * z_size;

  for (auto _tix = TIX; _tix < TOTAL; _tix += LinBlkSz) {
    auto x = (_tix % x_size);
    auto y = (_tix / x_size) % y_size;
    auto z = (_tix / x_size) / y_size;
    auto gx_1 = x / ProfBlkSzX;
    auto gx_2 = x % ProfBlkSzX;
    auto gy_1 = y / ProfBlkSzY;
    auto gy_2 = y % ProfBlkSzY;
    auto gz_1 = z / ProfBlkSzZ;
    auto gz_2 = z % ProfBlkSzZ;
    auto gx = (data_size.x / ProfNBlkX) * gx_1 + gx_2;
    auto gy = (data_size.y / ProfNBlkY) * gy_1 + gy_2;
    auto gz = (data_size.z / ProfNBlkZ) * gz_1 + gz_2;

    auto gid = gx + gy * data_leap.y + gz * data_leap.z;

    if (gx < data_size.x and gy < data_size.y and gz < data_size.z) s_data[z][y][x] = data[gid];
  }
  __syncthreads();
}

template <
    typename T1, typename T2, int SplDim = 3, int ProfNBlkX = 4, int ProfNBlkY = 4,
    int ProfNBlkZ = 4, int LinBlkSz = DefaultLinBlkSz>
__device__ void global2shmem_profiling_data_2(
    T1* data, dim3 data_size, dim3 data_leap, T2 s_data[ProfNBlkX * ProfNBlkY * ProfNBlkZ],
    T2 s_nx[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4], T2 s_ny[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4],
    T2 s_nz[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4])
{
  constexpr auto TOTAL = ProfNBlkX * ProfNBlkY * ProfNBlkZ * 4;
  int factors[4] = {-3, -1, 1, 3};
  for (auto _tix = TIX; _tix < TOTAL; _tix += LinBlkSz) {
    auto offset = (_tix % 4);
    auto idx = _tix / 4;
    auto x = idx % ProfNBlkX;
    auto y = (idx / ProfNBlkX) % ProfNBlkY;
    auto z = (idx / ProfNBlkX) / ProfNBlkY;
    auto gx = (data_size.x / ProfNBlkX) * x + data_size.x / (ProfNBlkX * 2);
    auto gy = (data_size.y / ProfNBlkY) * y + data_size.y / (ProfNBlkY * 2);
    auto gz = (data_size.z / ProfNBlkZ) * z + data_size.z / (ProfNBlkZ * 2);

    auto gid = gx + gy * data_leap.y + gz * data_leap.z;

    if constexpr (SplDim == 3) {
      if (gx >= 3 and gy >= 3 and gz >= 3 and gx + 3 < data_size.x and gy + 3 < data_size.y and
          gz + 3 < data_size.z) {
        s_data[idx] = data[gid];
        auto factor = factors[offset];
        s_nx[idx][offset] = data[gid + factor];
        s_ny[idx][offset] = data[gid + factor * data_leap.y];
        s_nz[idx][offset] = data[gid + factor * data_leap.z];
      }
    }

    if constexpr (SplDim == 2) {
      if (gx >= 3 and gy >= 3 and gx + 3 < data_size.x and gy + 3 < data_size.y) {
        s_data[idx] = data[gid];
        auto factor = factors[offset];
        s_nx[idx][offset] = data[gid + factor];
        s_ny[idx][offset] = data[gid + factor * data_leap.y];
      }
    }
  }
  __syncthreads();
}

template <
    typename T = float, typename E = u4, int LEVEL = 4, int SplDim = 2, int AncBlkSzX = 8,
    int AncBlkSzY = 8, int AncBlkSzZ = 8, int NAncBlkX = 4, int NAncBlkY = 1, int NAncBlkZ = 1,
    int LinBlkSz = DefaultLinBlkSz>
__device__ void global2shmem_fuse(
    E* eq, dim3 eq_size, dim3 eq_leap, T* scattered_outlier, dim3 begin,
    T s_eq[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
          [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    size_t grid_leaps[LEVEL + 1][2], size_t prefix_nums[LEVEL + 1])
{
  constexpr auto TOTAL = (AncBlkSzX * NAncBlkX + (SplDim >= 1)) *
                         (AncBlkSzY * NAncBlkY + (SplDim >= 2)) *
                         (AncBlkSzZ * NAncBlkZ + (SplDim >= 3));

  for (auto _tix = TIX; _tix < TOTAL; _tix += LinBlkSz) {
    auto x = (_tix % (AncBlkSzX * NAncBlkX + (SplDim >= 1)));
    auto y =
        (_tix / (AncBlkSzX * NAncBlkX + (SplDim >= 1))) % (AncBlkSzY * NAncBlkY + (SplDim >= 2));
    auto z =
        (_tix / (AncBlkSzX * NAncBlkX + (SplDim >= 1))) / (AncBlkSzY * NAncBlkY + (SplDim >= 2));
    auto gx = (begin.x + x + BIX * (AncBlkSzX * NAncBlkX));
    auto gy = (begin.y + y + BIY * (AncBlkSzY * NAncBlkY));
    auto gz = (begin.z + z + BIZ * (AncBlkSzZ * NAncBlkZ));
    if (gx < eq_size.x and gy < eq_size.y and gz < eq_size.z) {
      // todo: pre-compute the leaps and their halves

      int level = 0;
      auto data_gid = gx + gy * eq_leap.y + gz * eq_leap.z;
      while (gx % 2 == 0 and gy % 2 == 0 and gz % 2 == 0 and level < LEVEL) {
        gx = gx >> 1;
        gy = gy >> 1;
        gz = gz >> 1;
        level++;
      }
      auto gid = gx + gy * grid_leaps[level][0] + gz * grid_leaps[level][1];

      if (level < LEVEL) {  // non-anchor
        gid += prefix_nums[level] - ((gz + 1) >> 1) * grid_leaps[level + 1][1] -
               (gz % 2 == 0) * ((gy + 1) >> 1) * grid_leaps[level + 1][0] -
               (gz % 2 == 0 && gy % 2 == 0) * ((gx + 1) >> 1);
      }

      s_eq[z][y][x] = static_cast<T>(eq[gid]) + scattered_outlier[data_gid];
    }
  }
  __syncthreads();
}

// dram_outlier should be the same in type with shared memory buf
template <
    typename T1, typename T2, int SplDim, int AncBlkSzX, int AncBlkSzY, int AncBlkSzZ,
    int NAncBlkX, int NAncBlkY, int NAncBlkZ, int LinBlkSz = DefaultLinBlkSz>
__device__ void shmem2global_data(
    T1 s_buf[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
            [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    T2* dram_buf, dim3 buf_size, dim3 buf_leap, dim3 begin)
{
  auto x_size = AncBlkSzX * NAncBlkX + (BIX == GDX - 1) * (SplDim >= 1);
  auto y_size = AncBlkSzY * NAncBlkY + (BIY == GDY - 1) * (SplDim >= 2);
  auto z_size = AncBlkSzZ * NAncBlkZ + (BIZ == GDZ - 1) * (SplDim >= 3);
  auto TOTAL = x_size * y_size * z_size;

  for (auto _tix = TIX; _tix < TOTAL; _tix += LinBlkSz) {
    auto x = (_tix % x_size);
    auto y = (_tix / x_size) % y_size;
    auto z = (_tix / x_size) / y_size;
    auto gx = (begin.x + x + BIX * AncBlkSzX * NAncBlkX);
    auto gy = (begin.y + y + BIY * AncBlkSzY * NAncBlkY);
    auto gz = (begin.z + z + BIZ * AncBlkSzZ * NAncBlkZ);
    auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;

    if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z) dram_buf[gid] = s_buf[z][y][x];
  }
  __syncthreads();
}

template <
    typename T1, typename T2, int LEVEL = 4, int SplDim = 2, int AncBlkSzX = 8, int AncBlkSzY = 8,
    int AncBlkSzZ = 8, int NAncBlkX = 4, int NAncBlkY = 1, int NAncBlkZ = 1,
    int LinBlkSz = DefaultLinBlkSz, typename CompactValIdx>
__device__ void shmem2global_data_with_compaction(
    T1 s_buf[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
            [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    T2* dram_buf, dim3 buf_size, dim3 buf_leap, dim3 begin, int radius,
    size_t grid_leaps[LEVEL + 1][2], size_t prefix_nums[LEVEL + 1],
    CompactValIdx* dram_compact = nullptr, uint32_t* dram_compactnum = nullptr)
{
  using Val = typename CompactValIdx::OutlierValT;

  auto x_size = AncBlkSzX * NAncBlkX + (BIX == GDX - 1) * (SplDim >= 1);
  auto y_size = AncBlkSzY * NAncBlkY + (BIY == GDY - 1) * (SplDim >= 2);
  auto z_size = AncBlkSzZ * NAncBlkZ + (BIZ == GDZ - 1) * (SplDim >= 3);
  auto TOTAL = x_size * y_size * z_size;

  for (auto _tix = TIX; _tix < TOTAL; _tix += LinBlkSz) {
    auto x = (_tix % x_size);
    auto y = (_tix / x_size) % y_size;
    auto z = (_tix / x_size) / y_size;
    auto gx = (begin.x + x + BIX * AncBlkSzX * NAncBlkX);
    auto gy = (begin.y + y + BIY * AncBlkSzY * NAncBlkY);
    auto gz = (begin.z + z + BIZ * AncBlkSzZ * NAncBlkZ);
    // auto gid = gx + gy * buf_leap.y + gz * buf_leap.z;

    auto candidate = s_buf[z][y][x];
    bool quantizable = (candidate >= 0) and (candidate < 2 * radius);

    if (gx < buf_size.x and gy < buf_size.y and gz < buf_size.z) {
      if (not quantizable) {
        auto data_gid = [&]() { return gx + gy * buf_leap.y + gz * buf_leap.z; };
        auto cur_idx = atomicAdd(dram_compactnum, 1);
        dram_compact[cur_idx] = {(Val)candidate, data_gid()};
      }
      int level = 0;
      // todo: pre-compute the leaps and their halves
      while (gx % 2 == 0 and gy % 2 == 0 and gz % 2 == 0 and level < LEVEL) {
        gx = gx >> 1;
        gy = gy >> 1;
        gz = gz >> 1;
        level++;
      }
      auto gid = gx + gy * grid_leaps[level][0] + gz * grid_leaps[level][1];

      if (level < LEVEL) {  // non-anchor
        gid += prefix_nums[level] - ((gz + 1) >> 1) * grid_leaps[level + 1][1] -
               (gz % 2 == 0) * ((gy + 1) >> 1) * grid_leaps[level + 1][0] -
               (gz % 2 == 0 && gy % 2 == 0) * ((gx + 1) >> 1);
      }

      // TODO this is for algorithmic demo by reading from shmem
      // For performance purpose, it can be inlined in quantization
      dram_buf[gid] = quantizable * static_cast<T2>(candidate);
    }
  }
}

template <
    typename T1, typename T2, typename FP, int SplDim, int AncBlkSzX, int AncBlkSzY, int AncBlkSzZ,
    int NAncBlkX, int NAncBlkY, int NAncBlkZ, typename LAMBDAX, typename LAMBDAY, typename LAMBDAZ,
    bool BLUE, bool YELLOW, bool HOLLOW, bool Coarsen, int LinBlkSz, bool BorderIncl,
    bool Workflow>
__forceinline__ __device__ void interpolate_stage(
    T1 s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
             [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    T2 s_eq[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
           [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    dim3 data_size, LAMBDAX xmap, LAMBDAY ymap, LAMBDAZ zmap, int unit, FP eb_r, FP ebx2,
    int radius, bool interpolator, int BLOCK_DIMX, int BLOCK_DIMY, int BLOCK_DIMZ)
{
  // static_assert(BLOCK_DIMX * BLOCK_DIMY * (Coarsen ? 1 : BLOCK_DIMZ) <= BlkDimLin, "block
  // oversized");
  static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
  static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
  static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
  static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");

  auto run = [&](auto x, auto y, auto z) {
    if (xyz_predicate<
            SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ, BorderIncl>(
            x, y, z, data_size)) {
      auto global_x = BIX * AncBlkSzX * NAncBlkX + x;
      auto global_y = BIY * AncBlkSzY * NAncBlkY + y;
      auto global_z = BIZ * AncBlkSzZ * NAncBlkZ + z;

      T1 pred = 0;
      auto input_x = x;
      auto input_BI = BIX;
      auto input_GD = GDX;
      auto input_gx = global_x;
      auto input_gs = data_size.x;
      auto right_bound = AncBlkSzX * NAncBlkX + (SplDim >= 1);
      auto x_size = AncBlkSzX * NAncBlkX + (SplDim >= 1);
      auto y_size = AncBlkSzY * NAncBlkY + (SplDim >= 2);
      // auto z_size = AncBlkSzZ * NAncBlkZ + (SplDim >= 3);
      int p1 = -1, p2 = 9, p3 = 9, p4 = -1, p5 = 16;
      if (interpolator == 0) { p1 = -3, p2 = 23, p3 = 23, p4 = -3, p5 = 40; }
      if constexpr (BLUE) {
        input_x = z;
        input_BI = BIZ;
        input_GD = GDZ;
        input_gx = global_z;
        input_gs = data_size.z;
        right_bound = AncBlkSzZ * NAncBlkZ + (SplDim >= 3);
      }
      if constexpr (YELLOW) {
        input_x = y;
        input_BI = BIY;
        input_GD = GDY;
        input_gx = global_y;
        input_gs = data_size.y;
        right_bound = AncBlkSzY * NAncBlkY + (SplDim >= 2);
      }

      int id_[4], s_id[4];
      id_[0] = input_x - 3 * unit;
      id_[0] = id_[0] >= 0 ? id_[0] : 0;

      id_[1] = input_x - unit;
      id_[1] = id_[1] >= 0 ? id_[1] : 0;

      id_[2] = input_x + unit;
      id_[2] = id_[2] < right_bound ? id_[2] : 0;

      id_[3] = input_x + 3 * unit;
      id_[3] = id_[3] < right_bound ? id_[3] : 0;

      s_id[0] = x_size * y_size * z + x_size * y + id_[0];
      s_id[1] = x_size * y_size * z + x_size * y + id_[1];
      s_id[2] = x_size * y_size * z + x_size * y + id_[2];
      s_id[3] = x_size * y_size * z + x_size * y + id_[3];
      if constexpr (BLUE) {
        s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
        s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
        s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
        s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
      }
      if constexpr (YELLOW) {
        s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
        s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
        s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
        s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
      }

      bool case1 = (input_BI != input_GD - 1);
      bool case2 = (input_x >= 3 * unit);
      bool case3 = (input_x + 3 * unit <= AncBlkSzX * NAncBlkX);
      bool case4 = (input_gx + 3 * unit < input_gs);
      bool case5 = (input_gx + unit < input_gs);

      // 预加载 shared memory 数据到寄存器
      T1 tmp0 = *((T1*)s_data + s_id[0]);
      T1 tmp1 = *((T1*)s_data + s_id[1]);
      T1 tmp2 = *((T1*)s_data + s_id[2]);
      T1 tmp3 = *((T1*)s_data + s_id[3]);

      // 初始预测值
      pred = tmp1;

      // 计算不同 case 对应的 pred
      if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
        pred = (tmp1 + tmp2) / 2;
      }
      else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4)) {
        pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;
      }
      else if ((case1 && case2 && !case3) || (!case1 && case2 && !(case3 && case4) && case5)) {
        pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
      }
      else if ((case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
        pred = (p1 * tmp0 + p2 * tmp1 + p3 * tmp2 + p4 * tmp3) / p5;
      }

      if constexpr (Workflow == Spl3_Comp) {
        auto err = s_data[z][y][x] - pred;
        decltype(err) code;
        // TODO unsafe, did not deal with the out-of-cap case
        {
          code = fabs(err) * eb_r + 1;
          code = err < 0 ? -code : code;
          code = int(code / 2) + radius;
        }
        s_eq[z][y][x] = code;  // TODO double check if unsigned type works
        s_data[z][y][x] = pred + (code - radius) * ebx2;
      }
      else {  // TODO == DECOMPRESSS and static_assert
        auto code = s_eq[z][y][x];
        s_data[z][y][x] = pred + (code - radius) * ebx2;
      }
    }
  };
  // -------------------------------------------------------------------------------- //
  auto TOTAL = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
  if constexpr (Coarsen) {
    // if( BLOCK_DIMX *BLOCK_DIMY<= LinBlkSz){
    for (auto _tix = TIX; _tix < TOTAL; _tix += LinBlkSz) {
      auto itix = (_tix % BLOCK_DIMX);
      auto itiy = (_tix / BLOCK_DIMX) % BLOCK_DIMY;
      auto itiz = (_tix / BLOCK_DIMX) / BLOCK_DIMY;
      auto x = xmap(itix, unit);
      auto y = ymap(itiy, unit);
      auto z = zmap(itiz, unit);

      run(x, y, z);
    }
  }
  else {
    if (TIX < TOTAL) {
      auto itix = (TIX % BLOCK_DIMX);
      auto itiy = (TIX / BLOCK_DIMX) % BLOCK_DIMY;
      auto itiz = (TIX / BLOCK_DIMX) / BLOCK_DIMY;
      auto x = xmap(itix, unit);
      auto y = ymap(itiy, unit);
      auto z = zmap(itiz, unit);

      run(x, y, z);
    }
  }
  __syncthreads();
}

template <
    typename T1, typename T2, typename FP, int SplDim, int AncBlkSzX, int AncBlkSzY, int AncBlkSzZ,
    int NAncBlkX, int NAncBlkY, int NAncBlkZ, typename LAMBDA, bool LINE, bool FACE, bool CUBE,
    int LinBlkSz, bool Coarsen, bool BorderIncl, bool Workflow, typename INTERP>
__forceinline__ __device__ void interpolate_stage_md(
    T1 s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
             [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    T2 s_eq[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
           [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    dim3 data_size, LAMBDA xyzmap, int unit, FP eb_r, FP ebx2, int radius,
    INTERP cubic_interpolator, int NUM_ELE)
{
  // static_assert(Coarsen or (NUM_ELE <= BlkDimLin), "block oversized");
  static_assert((LINE or FACE or CUBE) == true, "must be one hot");
  static_assert((LINE and FACE) == false, "must be only one hot (1)");
  static_assert((LINE and CUBE) == false, "must be only one hot (2)");
  static_assert((FACE and CUBE) == false, "must be only one hot (3)");

  auto run = [&](auto x, auto y, auto z) {
    if (xyz_predicate<
            SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ, BorderIncl>(
            x, y, z, data_size)) {
      T1 pred = 0;
      auto global_x = BIX * AncBlkSzX * NAncBlkX + x;
      auto global_y = BIY * AncBlkSzY * NAncBlkY + y;
      auto global_z = BIZ * AncBlkSzZ * NAncBlkZ + z;

      int id_z[4], id_y[4], id_x[4];
      id_z[0] = (z - 3 * unit >= 0) ? z - 3 * unit : 0;
      id_z[1] = (z - unit >= 0) ? z - unit : 0;
      id_z[2] = (z + unit <= AncBlkSzZ * NAncBlkZ) ? z + unit : 0;
      id_z[3] = (z + 3 * unit <= AncBlkSzZ * NAncBlkZ) ? z + 3 * unit : 0;

      id_y[0] = (y - 3 * unit >= 0) ? y - 3 * unit : 0;
      id_y[1] = (y - unit >= 0) ? y - unit : 0;
      id_y[2] = (y + unit <= AncBlkSzY * NAncBlkY) ? y + unit : 0;
      id_y[3] = (y + 3 * unit <= AncBlkSzY * NAncBlkY) ? y + 3 * unit : 0;

      id_x[0] = (x - 3 * unit >= 0) ? x - 3 * unit : 0;
      id_x[1] = (x - unit >= 0) ? x - unit : 0;
      id_x[2] = (x + unit <= AncBlkSzX * NAncBlkX) ? x + unit : 0;
      id_x[3] = (x + 3 * unit <= AncBlkSzX * NAncBlkX) ? x + 3 * unit : 0;

      if constexpr (LINE) {
        bool I_Y = (y % (2 * unit)) > 0;
        bool I_Z = (z % (2 * unit)) > 0;

        pred = 0;
        auto input_x = x;
        auto input_BI = BIX;
        auto input_GD = GDX;
        auto input_gx = global_x;
        auto input_gs = data_size.x;

        auto right_bound = AncBlkSzX * NAncBlkX + (SplDim >= 1);
        auto x_size = AncBlkSzX * NAncBlkX + (SplDim >= 1);
        auto y_size = AncBlkSzY * NAncBlkY + (SplDim >= 2);
        // auto z_size = AncBlkSzZ * NAncBlkZ + (SplDim >= 3);

        if (I_Z) {
          input_x = z;
          input_BI = BIZ;
          input_GD = GDZ;
          input_gx = global_z;
          input_gs = data_size.z;
          right_bound = AncBlkSzZ * NAncBlkZ + (SplDim >= 3);
        }
        else if (I_Y) {
          input_x = y;
          input_BI = BIY;
          input_GD = GDY;
          input_gx = global_y;
          input_gs = data_size.y;
          right_bound = AncBlkSzY * NAncBlkY + (SplDim >= 2);
        }

        int id_[4], s_id[4];
        id_[0] = input_x - 3 * unit;
        id_[0] = id_[0] >= 0 ? id_[0] : 0;

        id_[1] = input_x - unit;
        id_[1] = id_[1] >= 0 ? id_[1] : 0;

        id_[2] = input_x + unit;
        id_[2] = id_[2] < right_bound ? id_[2] : 0;

        id_[3] = input_x + 3 * unit;
        id_[3] = id_[3] < right_bound ? id_[3] : 0;

        s_id[0] = x_size * y_size * z + x_size * y + id_[0];
        s_id[1] = x_size * y_size * z + x_size * y + id_[1];
        s_id[2] = x_size * y_size * z + x_size * y + id_[2];
        s_id[3] = x_size * y_size * z + x_size * y + id_[3];
        if (I_Z) {
          s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
          s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
          s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
          s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
        }
        else if (I_Y) {
          s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
          s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
          s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
          s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
        }

        bool case1 = (input_BI != input_GD - 1);
        bool case2 = (input_x >= 3 * unit);
        bool case3 = (input_x + 3 * unit <= AncBlkSzX * NAncBlkX);
        bool case4 = (input_gx + 3 * unit < input_gs);
        bool case5 = (input_gx + unit < input_gs);

        // 预加载 shared memory 数据到寄存器
        T1 tmp0 = *((T1*)s_data + s_id[0]);
        T1 tmp1 = *((T1*)s_data + s_id[1]);
        T1 tmp2 = *((T1*)s_data + s_id[2]);
        T1 tmp3 = *((T1*)s_data + s_id[3]);

        // 初始预测值
        pred = tmp1;

        // 计算不同 case 对应的 pred
        if ((case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
          pred = cubic_interpolator(tmp0, tmp1, tmp2, tmp3);
        }
        else if ((case1 && case2 && !case3) || (!case1 && case2 && !(case3 && case4) && case5)) {
          pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
        }
        else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4)) {
          pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;
        }
        else if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
          pred = (tmp1 + tmp2) / 2;
        }
      }
      auto get_interp_order = [&](auto x, auto BI, auto GD, auto gx, auto gs) {
        int b = (x >= 3 * unit) ? 3 : 1;
        int f =
            ((x + 3 * unit <= AncBlkSzX * NAncBlkX) && ((BI != GD - 1) || (gx + 3 * unit < gs)))
                ? 3
                : (((BI != GD - 1) || (gx + unit < gs)) ? 1 : 0);

        return (b == 3) ? ((f == 3) ? 4 : ((f == 1) ? 3 : 0))
                        : ((f == 3) ? 2 : ((f == 1) ? 1 : 0));
      };
      if constexpr (FACE) {  //

        bool I_YZ = (x % (2 * unit)) == 0;
        bool I_XZ = (y % (2 * unit)) == 0;

        int x_1, BI_1, GD_1, gx_1, gs_1;
        int x_2, BI_2, GD_2, gx_2, gs_2;
        int s_id_1[4], s_id_2[4];
        auto x_size = AncBlkSzX * NAncBlkX + (SplDim >= 1);
        auto y_size = AncBlkSzY * NAncBlkY + (SplDim >= 2);
        // auto z_size = AncBlkSzZ * NAncBlkZ + (SplDim >= 3);
        if (I_YZ) {
          x_1 = z, BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z, gs_1 = data_size.z;
          x_2 = y, BI_2 = BIY, GD_2 = GDY, gx_2 = global_y, gs_2 = data_size.y;
          s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
          s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
          s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
          s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;
          s_id_2[0] = x_size * y_size * z + x_size * id_y[0] + x;
          s_id_2[1] = x_size * y_size * z + x_size * id_y[1] + x;
          s_id_2[2] = x_size * y_size * z + x_size * id_y[2] + x;
          s_id_2[3] = x_size * y_size * z + x_size * id_y[3] + x;
          pred = s_data[id_z[1]][id_y[1]][x];
        }
        else if (I_XZ) {
          x_1 = z, BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z, gs_1 = data_size.z;
          x_2 = x, BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
          s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
          s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
          s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
          s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;

          s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
          s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
          s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
          s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
          pred = s_data[id_z[1]][y][id_x[1]];
        }
        else {
          x_1 = y, BI_1 = BIY, GD_1 = GDY, gx_1 = global_y, gs_1 = data_size.y;
          x_2 = x, BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
          s_id_1[0] = x_size * y_size * z + x_size * id_y[0] + x;
          s_id_1[1] = x_size * y_size * z + x_size * id_y[1] + x;
          s_id_1[2] = x_size * y_size * z + x_size * id_y[2] + x;
          s_id_1[3] = x_size * y_size * z + x_size * id_y[3] + x;
          s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
          s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
          s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
          s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
          pred = s_data[z][id_y[1]][id_x[1]];
        }

        auto interp_1 = get_interp_order(x_1, BI_1, GD_1, gx_1, gs_1);
        auto interp_2 = get_interp_order(x_2, BI_2, GD_2, gx_2, gs_2);

        int case_num = interp_1 + interp_2 * 5;

        // clang-format off
        if (interp_1 == 4 && interp_2 == 4) {
          pred  = ( cubic_interpolator( *((T1*)s_data + s_id_1[0]), *((T1*)s_data + s_id_1[1]), *((T1*)s_data + s_id_1[2]), *((T1*)s_data + s_id_1[3])) +
                    cubic_interpolator( *((T1*)s_data + s_id_2[0]), *((T1*)s_data + s_id_2[1]), *((T1*)s_data + s_id_2[2]), *((T1*)s_data + s_id_2[3]))) / 2; }
        else if (interp_1 != 4 && interp_2 == 4) {
          pred  =   cubic_interpolator( *((T1*)s_data + s_id_2[0]), *((T1*)s_data + s_id_2[1]), *((T1*)s_data + s_id_2[2]), *((T1*)s_data + s_id_2[3])); }
        else if (interp_1 == 4 && interp_2 != 4) {
          pred  =   cubic_interpolator( *((T1*)s_data + s_id_1[0]), *((T1*)s_data + s_id_1[1]), *((T1*)s_data + s_id_1[2]), *((T1*)s_data + s_id_1[3])); }
        else if (interp_1 == 3 && interp_2 == 3) {
          pred  = (-   (*((T1*)s_data + s_id_2[0])) + 6 * (*((T1*)s_data + s_id_2[1])) + 3 * (*((T1*)s_data + s_id_2[2]))) / 8;
          pred += (-   (*((T1*)s_data + s_id_1[0])) + 6 * (*((T1*)s_data + s_id_1[1])) + 3 * (*((T1*)s_data + s_id_1[2]))) / 8;
          pred /= 2; }
        else if (interp_1 == 3 && interp_2 == 2) {
          pred  = (3 * (*((T1*)s_data + s_id_2[1])) + 6 * (*((T1*)s_data + s_id_2[2])) -     (*((T1*)s_data + s_id_2[3]))) / 8;
          pred += (-   (*((T1*)s_data + s_id_1[0])) + 6 * (*((T1*)s_data + s_id_1[1])) + 3 * (*((T1*)s_data + s_id_1[2]))) / 8;
          pred /= 2; }
        else if (interp_1 == 3 && interp_2 < 2) {
          pred  = (-   (*((T1*)s_data + s_id_1[0])) + 6 * (*((T1*)s_data + s_id_1[1])) + 3 * (*((T1*)s_data + s_id_1[2]))) / 8; }
        else if (interp_1 == 2 && interp_2 == 3) {
          pred  = (3 * (*((T1*)s_data + s_id_1[1])) + 6 * (*((T1*)s_data + s_id_1[2])) -     (*((T1*)s_data + s_id_1[3]))) / 8;
          pred += (-   (*((T1*)s_data + s_id_2[0])) + 6 * (*((T1*)s_data + s_id_2[1])) + 3 * (*((T1*)s_data + s_id_2[2]))) / 8;
          pred /= 2; }
        else if (interp_1 == 2 && interp_2 == 2) {
          pred  = (3 * (*((T1*)s_data + s_id_1[1])) + 6 * (*((T1*)s_data + s_id_1[2])) -     (*((T1*)s_data + s_id_1[3]))) / 8;
          pred += (3 * (*((T1*)s_data + s_id_2[1])) + 6 * (*((T1*)s_data + s_id_2[2])) -     (*((T1*)s_data + s_id_2[3]))) / 8;
          pred /= 2; }
        else if (interp_1 == 2 && interp_2 < 2) {
          pred  = (3 * (*((T1*)s_data + s_id_1[1])) + 6 * (*((T1*)s_data + s_id_1[2])) -     (*((T1*)s_data + s_id_1[3]))) / 8; }
        else if (interp_1 <= 1 && interp_2 == 3) {
          pred  = (-   (*((T1*)s_data + s_id_2[0])) + 6 * (*((T1*)s_data + s_id_2[1])) + 3 * (*((T1*)s_data + s_id_2[2]))) / 8; }
        else if (interp_1 <= 1 && interp_2 == 2) {
          pred  = (3 * (*((T1*)s_data + s_id_2[1])) + 6 * (*((T1*)s_data + s_id_2[2])) -     (*((T1*)s_data + s_id_2[3]))) / 8; }
        else if (interp_1 == 1 && interp_2 == 1) {
          pred  = (    (*((T1*)s_data + s_id_2[1])) +     (*((T1*)s_data + s_id_2[2]))) / 2;
          pred += (    (*((T1*)s_data + s_id_1[1])) +     (*((T1*)s_data + s_id_1[2]))) / 2;
          pred /= 2; }
        else if (interp_1 == 1 && interp_2 < 1) {
          pred  = (    (*((T1*)s_data + s_id_1[1])) + (*((T1*)s_data + s_id_1[2]))) / 2; }
        else if (interp_1 == 0 && interp_2 == 1) {
          pred  = (     (*((T1*)s_data + s_id_2[1])) + (*((T1*)s_data + s_id_2[2]))) / 2; }
        else {
          pred  = (*((T1*)s_data + s_id_1[1])) + (*((T1*)s_data + s_id_2[1])) - pred; }
      }
      // clang-format on

      if constexpr (CUBE) {  //
        T1 tmp_z[4], tmp_y[4], tmp_x[4];
        auto interp_z = get_interp_order(z, BIZ, GDZ, global_z, data_size.z);
        auto interp_y = get_interp_order(y, BIY, GDY, global_y, data_size.y);
        auto interp_x = get_interp_order(x, BIX, GDX, global_x, data_size.x);

#pragma unroll
        for (int id_itr = 0; id_itr < 4; ++id_itr) { tmp_x[id_itr] = s_data[z][y][id_x[id_itr]]; }
        if (interp_z == 4) {
#pragma unroll
          for (int id_itr = 0; id_itr < 4; ++id_itr) {
            tmp_z[id_itr] = s_data[id_z[id_itr]][y][x];
          }
        }
        if (interp_y == 4) {
#pragma unroll
          for (int id_itr = 0; id_itr < 4; ++id_itr) {
            tmp_y[id_itr] = s_data[z][id_y[id_itr]][x];
          }
        }

        T1 pred_z[5], pred_y[5], pred_x[5];
        pred_x[0] = tmp_x[1];
        pred_x[1] = cubic_interpolator(tmp_x[0], tmp_x[1], tmp_x[2], tmp_x[3]);
        pred_x[2] = (-tmp_x[0] + 6 * tmp_x[1] + 3 * tmp_x[2]) / 8;
        pred_x[3] = (3 * tmp_x[1] + 6 * tmp_x[2] - tmp_x[3]) / 8;
        pred_x[4] = (tmp_x[1] + tmp_x[2]) / 2;

        pred_y[1] = cubic_interpolator(tmp_y[0], tmp_y[1], tmp_y[2], tmp_y[3]);
        pred_z[1] = cubic_interpolator(tmp_z[0], tmp_z[1], tmp_z[2], tmp_z[3]);

        // clang-format off
        pred = pred_x[0];
        pred = (interp_z == 4 && interp_y == 4 && interp_x == 4) ? (pred_x[1] + pred_y[1] + pred_z[1]) / 3 : pred;
        pred = (interp_z == 4 && interp_y == 4 && interp_x != 4) ? (pred_z[1] + pred_y[1]) / 2             : pred;
        pred = (interp_z == 4 && interp_y != 4 && interp_x == 4) ? (pred_z[1] + pred_x[1]) / 2             : pred;
        pred = (interp_z != 4 && interp_y == 4 && interp_x == 4) ? (pred_y[1] + pred_x[1]) / 2             : pred;
        pred = (interp_z == 4 && interp_y != 4 && interp_x != 4) ? pred_z[1]                               : pred;
        pred = (interp_z != 4 && interp_y == 4 && interp_x != 4) ? pred_y[1]                               : pred;
        pred = (interp_z != 4 && interp_y != 4 && interp_x == 4) ? pred_x[1]                               : pred;
        pred = (interp_z != 4 && interp_y != 4 && interp_x == 3) ? pred_x[2]                               : pred;
        pred = (interp_z != 4 && interp_y != 4 && interp_x == 2) ? pred_x[3]                               : pred;
        pred = (interp_z != 4 && interp_y != 4 && interp_x == 1) ? pred_x[4]                               : pred;
        // pred = (interp_z != 4 && interp_y != 4 && interp_x == 0) ? pred_x[0]: pred;
        // clang-format on
      }

      if constexpr (Workflow == Spl3_Comp) {
        auto err = s_data[z][y][x] - pred;
        decltype(err) code;
        // TODO unsafe, did not deal with the out-of-cap case
        {
          code = fabs(err) * eb_r + 1;
          code = err < 0 ? -code : code;
          code = int(code / 2) + radius;
        }
        s_eq[z][y][x] = code;  // TODO double check if unsigned type works

        s_data[z][y][x] = pred + (code - radius) * ebx2;
      }
      else {  // TODO == DECOMPRESSS and static_assert

        auto code = s_eq[z][y][x];
        s_data[z][y][x] = pred + (code - radius) * ebx2;
      }
    }
  };
  // -------------------------------------------------------------------------------- //

  if constexpr (Coarsen) {
    auto TOTAL = NUM_ELE;
    for (auto _tix = TIX; _tix < TOTAL; _tix += LinBlkSz) {
      auto [x, y, z] = xyzmap(_tix, unit);
      run(x, y, z);
    }
  }
  else {
    if (TIX < NUM_ELE) {
      auto [x, y, z] = xyzmap(TIX, unit);
      run(x, y, z);
    }
  }
  __syncthreads();
}

}  // namespace

template <
    typename T, int SplDim, int ProfBlkSzX, int ProfBlkSzY, int ProfBlkSzZ, int ProfNBlkX,
    int ProfNBlkY, int ProfNBlkZ, int LinBlkSz>
__device__ void psz::auto_tuning(
    T s_data[ProfBlkSzZ * ProfNBlkZ][ProfBlkSzY * ProfNBlkY][ProfBlkSzX * ProfNBlkX],
    T local_errs[2], dim3 data_size, T* errs)
{
  if (TIX < 2) local_errs[TIX] = 0;
  __syncthreads();

  auto local_idx = TIX % 2;
  auto temp = TIX / 2;

  auto block_idx_x = temp % ProfNBlkX;
  auto block_idx_y = (temp / ProfNBlkX) % ProfNBlkY;
  auto block_idx_z = ((temp / ProfNBlkX) / ProfNBlkY) % ProfNBlkZ;
  auto dir = ((temp / ProfNBlkX) / ProfNBlkY) / ProfNBlkZ;

  bool predicate = dir < 2;

  if (predicate) {
    auto x = ProfBlkSzX * block_idx_x + 1 + local_idx;
    auto y = ProfBlkSzY * block_idx_y + 1 + local_idx;
    auto z = ProfBlkSzZ * block_idx_z + 1 + local_idx;
    T pred = 0;
    switch (dir) {
      case 0: pred = (s_data[z - 1][y][x] + s_data[z + 1][y][x]) / 2; break;
      case 1: pred = (s_data[z][y][x - 1] + s_data[z][y][x + 1]) / 2; break;
      default: break;
    }

    T abs_error = fabs(pred - s_data[z][y][x]);
    atomicAdd(const_cast<T*>(local_errs) + dir, abs_error);
  }
  __syncthreads();
  if (TIX < 2) errs[TIX] = local_errs[TIX];
  __syncthreads();
}

template <typename T, int SplDim, int ProfNBlkX, int ProfNBlkY, int ProfNBlkZ, int LinBlkSz>
__device__ void psz::auto_tuning_2(
    T s_data[ProfNBlkX * ProfNBlkY * ProfNBlkZ], T s_nx[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4],
    T s_ny[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4], T s_nz[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4],
    T local_errs[6], dim3 data_size, T* errs)
{
  if constexpr (SplDim == 3) {
    if (TIX < 6) local_errs[TIX] = 0;
    __syncthreads();

    auto pt_idx = TIX % (ProfNBlkX * ProfNBlkY * ProfNBlkZ);
    auto c = TIX / (ProfNBlkX * ProfNBlkY * ProfNBlkZ);

    bool predicate = c < 6;
    if (predicate) {
      T pred = 0;

      // auto unit = 1;
      switch (c) {
          // clang-format off
        case 0: pred = (-    s_nz[pt_idx][0] +  9 * s_nz[pt_idx][1] +  9 * s_nz[pt_idx][2] -     s_nz[pt_idx][3]) / 16; break;
        case 1: pred = (-3 * s_nz[pt_idx][0] + 23 * s_nz[pt_idx][1] + 23 * s_nz[pt_idx][2] - 3 * s_nz[pt_idx][3]) / 40; break;
        case 2: pred = (-    s_ny[pt_idx][0] +  9 * s_ny[pt_idx][1] +  9 * s_ny[pt_idx][2] -     s_ny[pt_idx][3]) / 16; break;
        case 3: pred = (-3 * s_ny[pt_idx][0] + 23 * s_ny[pt_idx][1] + 23 * s_ny[pt_idx][2] - 3 * s_ny[pt_idx][3]) / 40; break;
        case 4: pred = (-    s_nx[pt_idx][0] +  9 * s_nx[pt_idx][1] +  9 * s_nx[pt_idx][2] -     s_nx[pt_idx][3]) / 16; break;
        case 5: pred = (-3 * s_nx[pt_idx][0] + 23 * s_nx[pt_idx][1] + 23 * s_nx[pt_idx][2] - 3 * s_nx[pt_idx][3]) / 40; break;
        default: break;
          // clang-format on
      }
      T abs_error = fabs(pred - s_data[pt_idx]);
      atomicAdd(const_cast<T*>(local_errs) + c, abs_error);
    }
    __syncthreads();
    if (TIX < 6) errs[TIX] = local_errs[TIX];
    __syncthreads();
  }

  if constexpr (SplDim == 3) {
    if (TIX < 4) local_errs[TIX] = 0;
    __syncthreads();
    auto pt_idx = TIX % (ProfNBlkX * ProfNBlkY * ProfNBlkZ);
    auto c = TIX / (ProfNBlkX * ProfNBlkY * ProfNBlkZ);
    bool predicate = c < 4;
    if (predicate) {
      T pred = 0;
      switch (c) {
          // clang-format off
        case 0: pred = (-    s_ny[pt_idx][0] +  9 * s_ny[pt_idx][1] +  9 * s_ny[pt_idx][2] -     s_ny[pt_idx][3]) / 16; break;
        case 1: pred = (-3 * s_ny[pt_idx][0] + 23 * s_ny[pt_idx][1] + 23 * s_ny[pt_idx][2] - 3 * s_ny[pt_idx][3]) / 40; break;
        case 2: pred = (-    s_nx[pt_idx][0] +  9 * s_nx[pt_idx][1] +  9 * s_nx[pt_idx][2] -     s_nx[pt_idx][3]) / 16; break;
        case 3: pred = (-3 * s_nx[pt_idx][0] + 23 * s_nx[pt_idx][1] + 23 * s_nx[pt_idx][2] - 3 * s_nx[pt_idx][3]) / 40; break;
        default: break;
          // clang-format on
      }
      T abs_error = fabs(pred - s_data[pt_idx]);
      atomicAdd(const_cast<T*>(local_errs) + c, abs_error);
    }
    __syncthreads();
    if (TIX < 4) errs[TIX] = local_errs[TIX];
    __syncthreads();
  }
}

template <int SplDim, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_line(int _tix, const int UNIT);
template <int SplDim, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_face(int _tix, const int UNIT);
template <int SplDim, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_cube(int _tix, const int UNIT);

template <int SplDim, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_line(int _tix, const int UNIT)
{
  if constexpr (SplDim == 3) {
    auto N = BLOCKSIZE / (UNIT * 2);
    auto L = N * (N + 1) * (N + 1);
    auto Q = (N + 1) * (N + 1);
    auto group = _tix / L;
    auto m = _tix % L;
    auto i = m / Q;
    auto j = (m % Q) / (N + 1);
    auto k = (m % Q) % (N + 1);
    if (group == 0)
      return std::make_tuple(2 * UNIT * i + UNIT, 2 * UNIT * j, 2 * UNIT * k);
    else if (group == 1)
      return std::make_tuple(2 * UNIT * k, 2 * UNIT * i + UNIT, 2 * UNIT * j);
    else
      return std::make_tuple(2 * UNIT * j, 2 * UNIT * k, 2 * UNIT * i + UNIT);
  }
  if constexpr (SplDim == 2) {
    auto N = BLOCKSIZE / (UNIT * 2);
    auto L = N * (N + 1);
    auto Q = (N + 1);
    auto group = _tix / L;
    auto m = _tix % L;
    auto i = m / Q;
    auto j = (m % Q);
    if (group == 0)
      return std::make_tuple(2 * UNIT * i + UNIT, 2 * UNIT * j, 0);
    else
      return std::make_tuple(2 * UNIT * j, 2 * UNIT * i + UNIT, 0);
  }
}

template <int SplDim, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_face(int _tix, const int UNIT)
{
  if constexpr (SplDim == 3) {
    auto N = BLOCKSIZE / (UNIT * 2);
    auto L = N * N * (N + 1);
    auto Q = N * N;
    auto group = _tix / L;
    auto m = _tix % L;
    auto i = m / Q;
    auto j = (m % Q) / N;
    auto k = (m % Q) % N;
    if (group == 0)
      return std::make_tuple(2 * UNIT * i, 2 * UNIT * j + UNIT, 2 * UNIT * k + UNIT);
    else if (group == 1)
      return std::make_tuple(2 * UNIT * k + UNIT, 2 * UNIT * i, 2 * UNIT * j + UNIT);
    else
      return std::make_tuple(2 * UNIT * j + UNIT, 2 * UNIT * k + UNIT, 2 * UNIT * i);
  }
  if constexpr (SplDim == 2) {
    auto N = BLOCKSIZE / (UNIT * 2);
    auto L = N * N;
    auto Q = N * N;
    // auto group = _tix / L ;
    auto m = _tix % L;

    auto i = (m % Q) / N;
    auto j = (m % Q) % N;
    return std::make_tuple(2 * UNIT * i + UNIT, 2 * UNIT * j + UNIT, 0);
  }
}

template <int SplDim, int BLOCKSIZE>
__device__ std::tuple<int, int, int> xyzmap_cube(int _tix, const int UNIT)
{
  if constexpr (SplDim == 3) {
    auto N = BLOCKSIZE / (UNIT * 2);
    auto Q = N * N;
    auto i = _tix / Q;
    auto j = (_tix % Q) / N;
    auto k = (_tix % Q) % N;
    return std::make_tuple(2 * UNIT * i + UNIT, 2 * UNIT * j + UNIT, 2 * UNIT * k + UNIT);
  }
}

template <
    typename T1, typename T2, typename FP, int LEVEL, int SplDim, int AncBlkSzX, int AncBlkSzY,
    int AncBlkSzZ, int NAncBlkX, int NAncBlkY, int NAncBlkZ, int LinBlkSz, bool Workflow,
    bool PROBE_PRED_ERROR>
__device__ void psz::spline_layout_interpolate(
    T1 s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
             [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    T2 s_eq[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
           [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    dim3 data_size, FP eb_r, FP ebx2, int radius, INTERP_PARAMS intp_param)
{
  auto xblue = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
  auto yblue = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
  auto zblue = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

  auto xblue_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix); };
  auto yblue_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy); };
  auto zblue_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

  auto xyellow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
  auto yyellow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2 + 1); };
  auto zyellow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

  auto xyellow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix); };
  auto yyellow_reverse = [] __device__(int _tiy, int unit) -> int {
    return unit * (_tiy * 2 + 1);
  };
  auto zyellow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2); };

  auto xhollow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 + 1); };
  auto yhollow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy); };
  auto zhollow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

  auto xhollow_reverse = [] __device__(int _tix, int unit) -> int {
    return unit * (_tix * 2 + 1);
  };
  auto yhollow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
  auto zhollow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2); };

  auto nan_cubic_interp = [] __device__(T1 a, T1 b, T1 c, T1 d) -> T1 {
    return (-a + 9 * b + 9 * c - d) / 16;
  };

  auto nat_cubic_interp = [] __device__(T1 a, T1 b, T1 c, T1 d) -> T1 {
    return (-3 * a + 23 * b + 23 * c - 3 * d) / 40;
  };

  constexpr auto Coarsen = true;
  // constexpr auto NO_COARSEN       = false;
  constexpr auto BorderIncl = true;
  constexpr auto BORDER_EXCLUSIVE = false;

  FP cur_ebx2 = ebx2, cur_eb_r = eb_r;

  auto calc_eb = [&](auto unit) {
    cur_ebx2 = ebx2, cur_eb_r = eb_r;
    int temp = 1;
    while (temp < unit) {
      temp *= 2;
      cur_eb_r *= intp_param.alpha;
      cur_ebx2 /= intp_param.alpha;
    }
    if (cur_ebx2 < ebx2 / intp_param.beta) {
      cur_ebx2 = ebx2 / intp_param.beta;
      cur_eb_r = eb_r * intp_param.beta;
    }
  };

  int max_unit = ((AncBlkSzX >= AncBlkSzY) ? AncBlkSzX : AncBlkSzY);
  max_unit = ((max_unit >= AncBlkSzZ) ? max_unit : AncBlkSzZ);
  max_unit /= 2;
  int unit_x = AncBlkSzX, unit_y = AncBlkSzY, unit_z = AncBlkSzZ;
  int level_id = LEVEL;
  level_id -= 1;
#pragma unroll
  for (int unit = max_unit; unit >= 1; unit /= 2, level_id--) {
    calc_eb(unit);
    unit_x = (SplDim >= 1) ? unit * 2 : 1;
    unit_y = (SplDim >= 2) ? unit * 2 : 1;
    unit_z = (SplDim >= 3) ? unit * 2 : 1;
    if (level_id != 0) {
      if (intp_param.use_md[level_id]) {
        int N_x = AncBlkSzX / (unit * 2);
        int N_y = AncBlkSzY / (unit * 2);
        int N_z = AncBlkSzZ / (unit * 2);
        int N_line = N_x * (N_y + 1) * (N_z + 1) + (N_x + 1) * N_y * (N_z + 1) +
                     (N_x + 1) * (N_y + 1) * N_z;
        int N_face = N_x * N_y * (N_z + 1) + N_x * (N_y + 1) * N_z + (N_x + 1) * N_y * N_z;
        int N_cube = N_x * N_y * N_z;
        if (intp_param.use_natural[level_id] == 0) {
          if constexpr (SplDim >= 1)
            interpolate_stage_md<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_line<SplDim, AncBlkSzX>), true, false, false, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyzmap_line<SplDim, AncBlkSzX>, unit, cur_eb_r, cur_ebx2,
                radius, nan_cubic_interp, N_line);
          if constexpr (SplDim >= 2)
            interpolate_stage_md<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_face<SplDim, AncBlkSzX>), false, true, false, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyzmap_face<SplDim, AncBlkSzX>, unit, cur_eb_r, cur_ebx2,
                radius, nan_cubic_interp, N_face);
          if constexpr (SplDim >= 3)
            interpolate_stage_md<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_cube<SplDim, AncBlkSzX>), false, false, true, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyzmap_cube<SplDim, AncBlkSzX>, unit, cur_eb_r, cur_ebx2,
                radius, nan_cubic_interp, N_cube);
        }
        else {
          if constexpr (SplDim >= 1)
            interpolate_stage_md<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_line<SplDim, AncBlkSzX>), true, false, false, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyzmap_line<SplDim, AncBlkSzX>, unit, cur_eb_r, cur_ebx2,
                radius, nat_cubic_interp, N_line);
          if constexpr (SplDim >= 2)
            interpolate_stage_md<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_face<SplDim, AncBlkSzX>), false, true, false, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyzmap_face<SplDim, AncBlkSzX>, unit, cur_eb_r, cur_ebx2,
                radius, nat_cubic_interp, N_face);
          if constexpr (SplDim >= 3)
            interpolate_stage_md<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_cube<SplDim, AncBlkSzX>), false, false, true, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyzmap_cube<SplDim, AncBlkSzX>, unit, cur_eb_r, cur_ebx2,
                radius, nat_cubic_interp, N_cube);
        }
      }
      else {
        if (intp_param.reverse[level_id]) {
          if constexpr (SplDim >= 1) {
            interpolate_stage<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),
                false, false, true, Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, s_eq, data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit,
                cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id],
                NAncBlkX * AncBlkSzX / unit_x, NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2),
                NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
            unit_x /= 2;
          }
          if constexpr (SplDim >= 2) {
            interpolate_stage<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),
                false, true, false, Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit,
                cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id],
                NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1), NAncBlkY * AncBlkSzY / unit_y,
                NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
            unit_y /= 2;
          }
          if constexpr (SplDim >= 3) {
            interpolate_stage<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse), true,
                false, false, Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, s_eq, data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit,
                cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id],
                NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1),
                NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2), NAncBlkZ * AncBlkSzZ / unit_z);
            unit_z /= 2;
          }
        }
        else {
          if constexpr (SplDim >= 3) {
            interpolate_stage<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xblue), decltype(yblue), decltype(zblue), true, false, false, Coarsen,
                LinBlkSz, BorderIncl, Workflow>(
                s_data, s_eq, data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius,
                intp_param.use_natural[level_id], NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1),
                NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2), NAncBlkZ * AncBlkSzZ / unit_z);
            unit_z /= 2;
          }
          if constexpr (SplDim >= 2) {
            interpolate_stage<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyellow), decltype(yyellow), decltype(zyellow), false, true, false,
                Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2,
                radius, intp_param.use_natural[level_id],
                NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1), NAncBlkY * AncBlkSzY / unit_y,
                NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
            unit_y /= 2;
          }
          if constexpr (SplDim >= 1) {
            interpolate_stage<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xhollow), decltype(yhollow), decltype(zhollow), false, false, true,
                Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, s_eq, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2,
                radius, intp_param.use_natural[level_id], NAncBlkX * AncBlkSzX / unit_x,
                NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2),
                NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
            unit_x /= 2;
          }
        }
      }
    }
    else {
      if (intp_param.use_md[level_id]) {
        int N_x = AncBlkSzX / (unit * 2);
        int N_y = AncBlkSzY / (unit * 2);
        int N_z = AncBlkSzZ / (unit * 2);
        int N_line = N_x * (N_y + 1) * (N_z + 1) + (N_x + 1) * N_y * (N_z + 1) +
                     (N_x + 1) * (N_y + 1) * N_z;
        int N_face = N_x * N_y * (N_z + 1) + N_x * (N_y + 1) * N_z + (N_x + 1) * N_y * N_z;
        int N_cube = N_x * N_y * N_z;
        if (intp_param.use_natural[level_id] == 0) {
          if constexpr (SplDim >= 1)
            interpolate_stage_md<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_line<SplDim, AncBlkSzX>), true, false, false, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyzmap_line<SplDim, AncBlkSzX>, unit, cur_eb_r, cur_ebx2,
                radius, nan_cubic_interp, N_line);
          if constexpr (SplDim >= 2)
            interpolate_stage_md<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_face<SplDim, AncBlkSzX>), false, true, false, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyzmap_face<SplDim, AncBlkSzX>, unit, cur_eb_r, cur_ebx2,
                radius, nan_cubic_interp, N_face);
          if constexpr (SplDim >= 3)
            interpolate_stage_md<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_cube<SplDim, AncBlkSzX>), false, false, true, LinBlkSz, Coarsen,
                BORDER_EXCLUSIVE, Workflow>(
                s_data, s_eq, data_size, xyzmap_cube<SplDim, AncBlkSzX>, unit, cur_eb_r, cur_ebx2,
                radius, nan_cubic_interp, N_cube);
        }
        else {
          if constexpr (SplDim >= 1)
            interpolate_stage_md<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_line<SplDim, AncBlkSzX>), true, false, false, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyzmap_line<SplDim, AncBlkSzX>, unit, cur_eb_r, cur_ebx2,
                radius, nat_cubic_interp, N_line);
          if constexpr (SplDim >= 2)
            interpolate_stage_md<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_face<SplDim, AncBlkSzX>), false, true, false, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyzmap_face<SplDim, AncBlkSzX>, unit, cur_eb_r, cur_ebx2,
                radius, nat_cubic_interp, N_face);
          if constexpr (SplDim >= 3)
            interpolate_stage_md<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_cube<SplDim, AncBlkSzX>), false, false, true, LinBlkSz, Coarsen,
                BORDER_EXCLUSIVE, Workflow>(
                s_data, s_eq, data_size, xyzmap_cube<SplDim, AncBlkSzX>, unit, cur_eb_r, cur_ebx2,
                radius, nat_cubic_interp, N_cube);
        }
      }
      else {
        if (intp_param.reverse[level_id]) {
          if constexpr (SplDim >= 1) {
            interpolate_stage<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),
                false, false, true, Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, s_eq, data_size, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit,
                cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id],
                NAncBlkX * AncBlkSzX / unit_x, NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2),
                NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
            unit_x /= 2;
          }
          if constexpr (SplDim >= 2) {
            interpolate_stage<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),
                false, true, false, Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit,
                cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id],
                NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1), NAncBlkY * AncBlkSzY / unit_y,
                NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
            unit_y /= 2;
          }
          if constexpr (SplDim >= 3) {
            interpolate_stage<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse), true,
                false, false, Coarsen, LinBlkSz, BORDER_EXCLUSIVE, Workflow>(
                s_data, s_eq, data_size, xblue_reverse, yblue_reverse, zblue_reverse, unit,
                cur_eb_r, cur_ebx2, radius, intp_param.use_natural[level_id],
                NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1),
                NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2), NAncBlkZ * AncBlkSzZ / unit_z);
            unit_z /= 2;
          }
        }
        else {
          if constexpr (SplDim >= 3) {
            interpolate_stage<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xblue), decltype(yblue), decltype(zblue), true, false, false, Coarsen,
                LinBlkSz, BorderIncl, Workflow>(
                s_data, s_eq, data_size, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2, radius,
                intp_param.use_natural[level_id], NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1),
                NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2), NAncBlkZ * AncBlkSzZ / unit_z);
            unit_z /= 2;
          }
          if constexpr (SplDim >= 2) {
            interpolate_stage<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyellow), decltype(yyellow), decltype(zyellow), false, true, false,
                Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, s_eq, data_size, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2,
                radius, intp_param.use_natural[level_id],
                NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1), NAncBlkY * AncBlkSzY / unit_y,
                NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
            unit_y /= 2;
          }
          if constexpr (SplDim >= 1) {
            interpolate_stage<
                T1, T2, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xhollow), decltype(yhollow), decltype(zhollow), false, false, true,
                Coarsen, LinBlkSz, BORDER_EXCLUSIVE, Workflow>(
                s_data, s_eq, data_size, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2,
                radius, intp_param.use_natural[level_id], NAncBlkX * AncBlkSzX / unit_x,
                NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2),
                NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
            unit_x /= 2;
          }
        }
      }
    }
  }
}

/********************************************************************************
 * host API/kernel
 ********************************************************************************/
template <
    typename T, int SplDim = 3, int ProfBlkSzX = 4, int ProfBlkSzY = 4, int ProfBlkSzZ = 4,
    int ProfNBlkX = 4, int ProfNBlkY = 4, int ProfNBlkZ = 4, int LinBlkSz = DefaultLinBlkSz>
__global__ void psz::KCU_c_spl_prof_data(T* data, dim3 data_size, dim3 data_leap, T* errors)
{
  __shared__ T s_data[ProfBlkSzZ * ProfNBlkZ][ProfBlkSzY * ProfNBlkY][ProfBlkSzX * ProfNBlkX];
  __shared__ T s_local_errs[2];

  c_reset_scratch_profiling_data<
      T, SplDim, ProfBlkSzX, ProfBlkSzY, ProfBlkSzZ, ProfNBlkX, ProfNBlkY, ProfNBlkZ, LinBlkSz>(
      s_data, 0.0);

  global2shmem_profiling_data<
      T, T, ProfBlkSzX, ProfBlkSzY, ProfBlkSzZ, ProfNBlkX, ProfNBlkY, ProfNBlkZ, LinBlkSz>(
      data, data_size, data_leap, s_data);

  psz::auto_tuning<
      T, SplDim, ProfBlkSzX, ProfBlkSzY, ProfBlkSzZ, ProfNBlkX, ProfNBlkY, ProfNBlkZ, LinBlkSz>(
      s_data, s_local_errs, data_size, errors);
}

template <
    typename T, int SplDim = 3, int ProfNBlkX = 4, int ProfNBlkY = 4, int ProfNBlkZ = 4,
    int LinBlkSz = DefaultLinBlkSz>
__global__ void psz::KCU_c_spl_prof_data_2(T* data, dim3 data_size, dim3 data_leap, T* errors)
{
  __shared__ T s_data[ProfNBlkX * ProfNBlkY * ProfNBlkZ];
  __shared__ T s_neighbor_x[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4];
  __shared__ T s_neighbor_y[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4];
  __shared__ T s_neighbor_z[ProfNBlkX * ProfNBlkY * ProfNBlkZ][4];
  __shared__ T s_local_errs[6];

  c_reset_scratch_profiling_data_2<T, SplDim, ProfNBlkX, ProfNBlkY, ProfNBlkZ, LinBlkSz>(
      s_data, s_neighbor_x, s_neighbor_y, s_neighbor_z, 0.0);
  global2shmem_profiling_data_2<T, T, SplDim, ProfNBlkX, ProfNBlkY, ProfNBlkZ, LinBlkSz>(
      data, data_size, data_leap, s_data, s_neighbor_x, s_neighbor_y, s_neighbor_z);

  if (TIX < 6 and BIX == 0 and BIY == 0 and BIZ == 0) errors[TIX] = 0.0;  // risky

  psz::auto_tuning_2<T, SplDim, ProfNBlkX, ProfNBlkY, ProfNBlkZ, LinBlkSz>(
      s_data, s_neighbor_x, s_neighbor_y, s_neighbor_z, s_local_errs, data_size, errors);
}

template <int LEVEL>
__forceinline__ __device__ void pre_compute(
    dim3 data_size, size_t grid_leaps[LEVEL + 1][2], size_t prefix_nums[LEVEL + 1])
{
  if (TIX == 0) {
    auto d_size = data_size;

    int level = 0;
    while (level <= LEVEL) {
      // grid_leaps[level][0] = 1;
      grid_leaps[level][0] = d_size.x;
      grid_leaps[level][1] = d_size.x * d_size.y;
      if (level < LEVEL) {
        d_size.x = (d_size.x + 1) >> 1;
        d_size.y = (d_size.y + 1) >> 1;
        d_size.z = (d_size.z + 1) >> 1;
        prefix_nums[level] = d_size.x * d_size.y * d_size.z;
      }
      level++;
    }
    prefix_nums[LEVEL] = 0;
  }
  __syncthreads();
}

template <
    typename T, typename E, typename FP, int LEVEL, int SplDim, int AncBlkSzX, int AncBlkSzY,
    int AncBlkSzZ, int NAncBlkX, int NAncBlkY, int NAncBlkZ, int LinBlkSz, typename CompactValIdx,
    typename CompactNum>
__global__ void psz::KCU_c_spl_infprecis_data(
    T* data, dim3 data_size, dim3 data_leap, E* eq, dim3 eq_size, dim3 eq_leap, T* anchor,
    dim3 anchor_leap, CompactValIdx cvi, CompactNum cn, FP eb_r, FP ebx2, int radius,
    INTERP_PARAMS intp_param)
{
  __shared__ T s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
                     [AncBlkSzX * NAncBlkX + (SplDim >= 1)];
  __shared__ T s_eq[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
                   [AncBlkSzX * NAncBlkX + (SplDim >= 1)];
  __shared__ size_t s_grid_leaps[LEVEL + 1][2];
  __shared__ size_t s_prefix_nums[LEVEL + 1];

  dim3 begin{0, 0, 0};  // local frame; the offset lives in the (pre-offset) pointers
  auto sub_extent = data_size;

  pre_compute<LEVEL>(eq_size, s_grid_leaps, s_prefix_nums);

  c_reset_scratch_data<
      T, T, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ, LinBlkSz>(
      s_data, s_eq, radius);

  global2shmem_data<
      T, T, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ, LinBlkSz>(
      data, data_size, data_leap, begin, s_data);

  c_gather_anchor<T, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ>(
      data, data_size, data_leap, anchor, anchor_leap, begin);
  psz::spline_layout_interpolate<
      T, T, FP, LEVEL, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
      LinBlkSz, Spl3_Comp, false>(s_data, s_eq, sub_extent, eb_r, ebx2, radius, intp_param);

  shmem2global_data_with_compaction<
      T, E, LEVEL, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
      LinBlkSz>(s_eq, eq, eq_size, eq_leap, begin, radius, s_grid_leaps, s_prefix_nums, cvi, cn);
}

template <
    typename E, typename T, typename FP, int LEVEL, int SplDim, int AncBlkSzX, int AncBlkSzY,
    int AncBlkSzZ, int NAncBlkX, int NAncBlkY, int NAncBlkZ,
    int LinBlkSz>
__global__ void psz::KCU_x_spl_infprecis_data(
    E* eq,             // input 1
    dim3 eq_size,      //
    dim3 eq_leap,      //
    T* anchor,         // input 2
    dim3 anchor_size,  //
    dim3 anchor_leap,  //
    T* data,           // output
    dim3 data_size,    //
    dim3 data_leap,    //
    T* outlier_tmp, FP eb_r, FP ebx2, int radius, INTERP_PARAMS intp_param)
{
  __shared__ T s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
                     [AncBlkSzX * NAncBlkX + (SplDim >= 1)];
  __shared__ T s_eq[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
                   [AncBlkSzX * NAncBlkX + (SplDim >= 1)];
  __shared__ size_t s_grid_leaps[LEVEL + 1][2];
  __shared__ size_t s_prefix_nums[LEVEL + 1];

  dim3 begin{0, 0, 0};  // local frame; the offset lives in the (pre-offset) pointers
  auto sub_extent = data_size;

  pre_compute<LEVEL>(eq_size, s_grid_leaps, s_prefix_nums);

  x_reset_scratch_data<
      T, T, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ, LinBlkSz>(
      s_data, s_eq, anchor, anchor_size, anchor_leap, begin);
  global2shmem_fuse<
      T, E, LEVEL, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
      LinBlkSz>(eq, eq_size, eq_leap, outlier_tmp, begin, s_eq, s_grid_leaps, s_prefix_nums);

  psz::spline_layout_interpolate<
      T, T, FP, LEVEL, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
      LinBlkSz, Spl3_Decomp, false>(s_data, s_eq, sub_extent, eb_r, ebx2, radius, intp_param);
  shmem2global_data<
      T, T, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ, LinBlkSz>(
      s_data, data, data_size, data_leap, begin);
}

template <typename T>
__global__ void psz::reset_errors(T* errors)
{
  if (TIX < 36) errors[TIX] = 0;
}

template <typename T, int SplDim>
__forceinline__ __device__ void pre_compute_att(
    dim3 sam_starts, dim3 sam_bgs, dim3 sam_strides, dim3& g_starts, INTERP_PARAMS& intp_param,
    uint8_t& level, uint8_t& unit, T err[6], bool workflow);

// template <typename T>
// __forceinline__ __device__ void pre_compute_att<T, 3>(dim3 sam_starts, dim3 sam_bgs, dim3
// sam_strides, dim3 &g_starts, INTERP_PARAMS &intp_param, uint8_t &level, uint8_t
// &unit, T err[6], bool workflow){
template <typename T, int SplDim, int LEVEL>
__forceinline__ __device__ void pre_compute_att(
    dim3 sam_starts, dim3 sam_bgs, dim3 sam_strides, dim3& g_starts, INTERP_PARAMS& intp_param,
    uint8_t& level, uint8_t& unit, T err[9], bool workflow)
{
  if (TIX < 9) err[TIX] = 0.0;

  auto grid_idx_x = BIX % sam_bgs.x;
  auto grid_idx_y = (BIX / sam_bgs.x) % sam_bgs.y;
  auto grid_idx_z = (BIX / sam_bgs.x) / sam_bgs.y;
  g_starts.x = sam_starts.x + grid_idx_x * sam_strides.x;
  g_starts.y = sam_starts.y + grid_idx_y * sam_strides.y;
  g_starts.z = sam_starts.z + grid_idx_z * sam_strides.z;

  if constexpr (SplDim == 3) {
    if (workflow == Spl3_PredAtt) {
      bool use_natural = false, use_md = false, reverse = false;
      if (BIY == 0) { level = 2; }
      else if (BIY < 3) {
        level = 1;
        use_natural = (BIY == 2);
      }
      else {
        level = 0;
        use_natural = BIY > 5;
        use_md = (BIY == 5 or BIY == 8);
        reverse = BIY % 3;
      }
      intp_param.use_natural[level] = use_natural;
      intp_param.use_md[level] = use_md;
      intp_param.reverse[level] = reverse;
    }
    else {
      level = 0;
      if (BIY == 0) {
        intp_param.alpha = 1.0;
        intp_param.beta = 2.0;
      }
      else if (BIY == 1) {
        intp_param.alpha = 1.25;
        intp_param.beta = 2.0;
      }
      else {
        intp_param.alpha = 1.5 + 0.25 * ((BIY - 2) / 3);
        intp_param.beta = 2.0 + ((BIY - 2) % 3);
      }
    }
    unit = 1 << level;
  }

  if constexpr (SplDim == 2) {
    if (workflow == Spl3_PredAtt) {
      // bool use_natural = false, use_md = false, reverse = false;
      // level = LEVEL - (BIY / 6) - 1;
      // use_natural = (BIY % 6) >= 3;
      // use_md = (BIY % 3) == 2;
      // reverse = BIY % 3;
      // intp_param.use_natural[level] = use_natural;
      // intp_param.use_md[level] = use_md;
      // intp_param.reverse[level] = reverse;
      bool use_natural = false, use_md = false, reverse = false;
      if (BIY == 0) { level = 3; }
      else if (BIY < 3) {
        level = 2;
        use_natural = (BIY == 2);
      }
      else if (BIY < 5) {
        level = 1;
        use_natural = (BIY == 2);
      }
      else {
        level = 0;
        use_natural = BIY > 7;
        use_md = (BIY == 7 or BIY == 10);
        reverse = (BIY + 1) % 3;
      }
      intp_param.use_natural[level] = use_natural;
      intp_param.use_md[level] = use_md;
      intp_param.reverse[level] = reverse;
    }
    else {
      level = 0;
      if (BIY == 0) {
        intp_param.alpha = 1.0;
        intp_param.beta = 2.0;
      }
      else if (BIY == 1) {
        intp_param.alpha = 1.25;
        intp_param.beta = 2.0;
      }
      else {
        intp_param.alpha = 1.5 + 0.25 * ((BIY - 2) / 3);
        intp_param.beta = 2.0 + ((BIY - 2) % 3);
      }
    }
    unit = 1 << level;
  }

  __syncthreads();
}

template <
    typename T1, typename T2, int SplDim = 2, int AncBlkSzX = 8, int AncBlkSzY = 8,
    int AncBlkSzZ = 8, int NAncBlkX = 4, int NAncBlkY = 1, int NAncBlkZ = 1,
    int LinBlkSz = DefaultLinBlkSz>
__device__ void global2shmem_data_att(
    T1* data, dim3 data_size, dim3 data_leap,
    T2 s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
             [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    dim3 g_starts, uint8_t unit)
{
  constexpr auto TOTAL = (AncBlkSzX * NAncBlkX + (SplDim >= 1)) *
                         (AncBlkSzY * NAncBlkY + (SplDim >= 2)) *
                         (AncBlkSzZ * NAncBlkZ + (SplDim >= 3));

  for (auto _tix = TIX; _tix < TOTAL; _tix += LinBlkSz) {
    auto x = (_tix % (AncBlkSzX * NAncBlkX + (SplDim >= 1)));
    auto y =
        (_tix / (AncBlkSzX * NAncBlkX + (SplDim >= 1))) % (AncBlkSzY * NAncBlkY + (SplDim >= 2));
    auto z =
        (_tix / (AncBlkSzX * NAncBlkX + (SplDim >= 1))) / (AncBlkSzY * NAncBlkY + (SplDim >= 2));
    auto gx = (x + g_starts.x);
    auto gy = (y + g_starts.y);
    auto gz = (z + g_starts.z);
    auto gid = gx + gy * data_leap.y + gz * data_leap.z;

    if (gx < data_size.x and gy < data_size.y and gz < data_size.z) s_data[z][y][x] = data[gid];
  }
  __syncthreads();
}

template <
    typename T, typename FP, int SplDim, int AncBlkSzX, int AncBlkSzY, int AncBlkSzZ, int NAncBlkX,
    int NAncBlkY, int NAncBlkZ, typename LAMBDAX, typename LAMBDAY, typename LAMBDAZ, bool BLUE,
    bool YELLOW, bool HOLLOW, bool Coarsen, int LinBlkSz, bool BorderIncl, bool Workflow>
__forceinline__ __device__ void interpolate_stage_att(
    T s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
            [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    dim3 data_size, dim3 g_starts, LAMBDAX xmap, LAMBDAY ymap, LAMBDAZ zmap, int unit, FP eb_r,
    FP ebx2, bool interpolator, T* error, int BLOCK_DIMX, int BLOCK_DIMY, int BLOCK_DIMZ)
{
  // static_assert(BLOCK_DIMX * BLOCK_DIMY * (Coarsen ? 1 : BLOCK_DIMZ) <= BlkDimLin, "block
  // oversized");
  static_assert((BLUE or YELLOW or HOLLOW) == true, "must be one hot");
  static_assert((BLUE and YELLOW) == false, "must be only one hot (1)");
  static_assert((BLUE and YELLOW) == false, "must be only one hot (2)");
  static_assert((YELLOW and HOLLOW) == false, "must be only one hot (3)");
  // dim3 g_starts (g_starts_v.x,g_starts_v.y, g_starts_v.z);
  auto run = [&](auto x, auto y, auto z) {
    if (xyz_predicate_att<
            SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ, BorderIncl>(
            x, y, z, data_size, g_starts)) {
      T pred = 0;

      auto global_x = g_starts.x + x, global_y = g_starts.y + y, global_z = g_starts.z + z;
      auto input_x = x;
      // auto input_BI = BIX;
      // auto input_GD = GDX;
      auto input_gx = global_x;
      auto input_gs = data_size.x;
      auto right_bound = AncBlkSzX * NAncBlkX + (SplDim >= 1);
      auto x_size = AncBlkSzX * NAncBlkX + (SplDim >= 1);
      auto y_size = AncBlkSzY * NAncBlkY + (SplDim >= 2);
      // auto z_size = AncBlkSzZ * NAncBlkZ + (SplDim >= 3);
      int global_start_ = g_starts.x;
      int p1 = -1, p2 = 9, p3 = 9, p4 = -1, p5 = 16;
      if (interpolator == 0) { p1 = -3, p2 = 23, p3 = 23, p4 = -3, p5 = 40; }
      if constexpr (BLUE) {
        input_x = z;
        //    input_BI = BIZ;
        //    input_GD = GDZ;
        input_gx = global_z;
        input_gs = data_size.z;
        global_start_ = g_starts.z;
        right_bound = AncBlkSzZ * NAncBlkZ + (SplDim >= 3);
      }
      if constexpr (YELLOW) {
        input_x = y;
        //    input_BI = BIY;
        //    input_GD = GDY;
        input_gx = global_y;
        input_gs = data_size.y;
        global_start_ = g_starts.y;
        right_bound = AncBlkSzY * NAncBlkY + (SplDim >= 2);
      }

      int id_[4], s_id[4];
      id_[0] = input_x - 3 * unit;
      id_[0] = id_[0] >= 0 ? id_[0] : 0;

      id_[1] = input_x - unit;
      id_[1] = id_[1] >= 0 ? id_[1] : 0;

      id_[2] = input_x + unit;
      id_[2] = id_[2] < right_bound ? id_[2] : 0;

      id_[3] = input_x + 3 * unit;
      id_[3] = id_[3] < right_bound ? id_[3] : 0;

      s_id[0] = x_size * y_size * z + x_size * y + id_[0];
      s_id[1] = x_size * y_size * z + x_size * y + id_[1];
      s_id[2] = x_size * y_size * z + x_size * y + id_[2];
      s_id[3] = x_size * y_size * z + x_size * y + id_[3];
      if constexpr (BLUE) {
        s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
        s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
        s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
        s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
      }
      if constexpr (YELLOW) {
        s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
        s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
        s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
        s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
      }

      bool case1 = (global_start_ + AncBlkSzX * NAncBlkX < input_gs);
      bool case2 = (input_x >= 3 * unit);
      bool case3 = (input_x + 3 * unit <= AncBlkSzX * NAncBlkX);
      bool case4 = (input_gx + 3 * unit < input_gs);
      bool case5 = (input_gx + unit < input_gs);

      // 预加载 shared memory 数据到寄存器
      T tmp0 = *((T*)s_data + s_id[0]);
      T tmp1 = *((T*)s_data + s_id[1]);
      T tmp2 = *((T*)s_data + s_id[2]);
      T tmp3 = *((T*)s_data + s_id[3]);

      // 初始预测值
      pred = tmp1;

      // 计算不同 case 对应的 pred
      if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
        pred = (tmp1 + tmp2) / 2;
      }
      else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4)) {
        pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;
      }
      else if ((case1 && case2 && !case3) || (!case1 && case2 && !(case3 && case4) && case5)) {
        pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
      }
      else if ((case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
        pred = (p1 * tmp0 + p2 * tmp1 + p3 * tmp2 + p4 * tmp3) / p5;
      }

      if constexpr (Workflow == Spl3_AbAtt) {
        auto err = s_data[z][y][x] - pred;
        decltype(err) code;
        // TODO unsafe, did not deal with the out-of-cap case
        {
          code = fabs(err) * eb_r + 1;
          code = err < 0 ? -code : code;
          code = int(code / 2);
        }

        s_data[z][y][x] = pred + code * ebx2;
        atomicAdd(const_cast<T*>(error), code != 0);
      }
      else {
        atomicAdd(const_cast<T*>(error), fabs(s_data[z][y][x] - pred));
      }
    }
  };
  // -------------------------------------------------------------------------------- //
  auto TOTAL = BLOCK_DIMX * BLOCK_DIMY * BLOCK_DIMZ;
  if constexpr (Coarsen) {
    for (auto _tix = TIX; _tix < TOTAL; _tix += LinBlkSz) {
      auto itix = (_tix % BLOCK_DIMX);
      auto itiy = (_tix / BLOCK_DIMX) % BLOCK_DIMY;
      auto itiz = (_tix / BLOCK_DIMX) / BLOCK_DIMY;
      auto x = xmap(itix, unit);
      auto y = ymap(itiy, unit);
      auto z = zmap(itiz, unit);

      run(x, y, z);
    }
  }
  else {
    if (TIX < TOTAL) {
      auto itix = (TIX % BLOCK_DIMX);
      auto itiy = (TIX / BLOCK_DIMX) % BLOCK_DIMY;
      auto itiz = (TIX / BLOCK_DIMX) / BLOCK_DIMY;
      auto x = xmap(itix, unit);
      auto y = ymap(itiy, unit);
      auto z = zmap(itiz, unit);
      run(x, y, z);
    }
  }
  __syncthreads();
}

template <
    typename T, typename FP, int SplDim, int AncBlkSzX, int AncBlkSzY, int AncBlkSzZ, int NAncBlkX,
    int NAncBlkY, int NAncBlkZ, typename LAMBDA, bool LINE, bool FACE, bool CUBE, int LinBlkSz,
    bool Coarsen, bool BorderIncl, bool Workflow, typename INTERP>
__forceinline__ __device__ void interpolate_stage_md_att(
    T s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
            [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    dim3 data_size, dim3 g_starts, LAMBDA xyzmap, int unit, FP eb_r, FP ebx2,
    INTERP cubic_interpolator, T* error, int NUM_ELE)
{
  // static_assert(Coarsen or (NUM_ELE <= BlkDimLin), "block oversized");
  static_assert((LINE or FACE or CUBE) == true, "must be one hot");
  static_assert((LINE and FACE) == false, "must be only one hot (1)");
  static_assert((LINE and CUBE) == false, "must be only one hot (2)");
  static_assert((FACE and CUBE) == false, "must be only one hot (3)");
  // dim3 g_starts (g_starts_v.x,g_starts_v.y, g_starts_v.z);
  auto run = [&](auto x, auto y, auto z) {
    if (xyz_predicate_att<
            SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ, BorderIncl>(
            x, y, z, data_size, g_starts)) {
      T pred = 0;

      auto global_x = g_starts.x + x, global_y = g_starts.y + y, global_z = g_starts.z + z;
      //    T tmp_z[4], tmp_y[4], tmp_x[4];
      int id_z[4], id_y[4], id_x[4];
      id_z[0] = (z - 3 * unit >= 0) ? z - 3 * unit : 0;
      id_z[1] = (z - unit >= 0) ? z - unit : 0;
      id_z[2] = (z + unit <= AncBlkSzZ * NAncBlkZ) ? z + unit : 0;
      id_z[3] = (z + 3 * unit <= AncBlkSzZ * NAncBlkZ) ? z + 3 * unit : 0;

      id_y[0] = (y - 3 * unit >= 0) ? y - 3 * unit : 0;
      id_y[1] = (y - unit >= 0) ? y - unit : 0;
      id_y[2] = (y + unit <= AncBlkSzY * NAncBlkY) ? y + unit : 0;
      id_y[3] = (y + 3 * unit <= AncBlkSzY * NAncBlkY) ? y + 3 * unit : 0;

      id_x[0] = (x - 3 * unit >= 0) ? x - 3 * unit : 0;
      id_x[1] = (x - unit >= 0) ? x - unit : 0;
      id_x[2] = (x + unit <= AncBlkSzX * NAncBlkX) ? x + unit : 0;
      id_x[3] = (x + 3 * unit <= AncBlkSzX * NAncBlkX) ? x + 3 * unit : 0;

      if constexpr (LINE) {
        bool I_Y = (y % (2 * unit)) > 0;
        bool I_Z = (z % (2 * unit)) > 0;

        pred = 0;
        auto input_x = x;
        // auto input_BI = BIX;
        // auto input_GD = GDX;
        auto input_gx = global_x;
        auto input_gs = data_size.x;
        auto right_bound = AncBlkSzX * NAncBlkX + (SplDim >= 1);
        auto x_size = AncBlkSzX * NAncBlkX + (SplDim >= 1);
        auto y_size = AncBlkSzY * NAncBlkY + (SplDim >= 2);
        // auto z_size = AncBlkSzZ * NAncBlkZ + (SplDim >= 3);
        int global_start_ = g_starts.x;
        if (I_Z) {
          input_x = z;
          // input_BI = BIZ;
          // input_GD = GDZ;
          input_gx = global_z;
          input_gs = data_size.z;
          global_start_ = g_starts.z;
          right_bound = AncBlkSzZ * NAncBlkZ + (SplDim >= 3);
        }
        else if (I_Y) {
          input_x = y;
          // input_BI = BIY;
          // input_GD = GDY;
          input_gx = global_y;
          input_gs = data_size.y;
          global_start_ = g_starts.y;
          right_bound = AncBlkSzY * NAncBlkY + (SplDim >= 2);
        }

        int id_[4], s_id[4];
        id_[0] = input_x - 3 * unit;
        id_[0] = id_[0] >= 0 ? id_[0] : 0;

        id_[1] = input_x - unit;
        id_[1] = id_[1] >= 0 ? id_[1] : 0;

        id_[2] = input_x + unit;
        id_[2] = id_[2] < right_bound ? id_[2] : 0;

        id_[3] = input_x + 3 * unit;
        id_[3] = id_[3] < right_bound ? id_[3] : 0;

        s_id[0] = x_size * y_size * z + x_size * y + id_[0];
        s_id[1] = x_size * y_size * z + x_size * y + id_[1];
        s_id[2] = x_size * y_size * z + x_size * y + id_[2];
        s_id[3] = x_size * y_size * z + x_size * y + id_[3];
        if (I_Z) {
          s_id[0] = x_size * y_size * id_[0] + x_size * y + x;
          s_id[1] = x_size * y_size * id_[1] + x_size * y + x;
          s_id[2] = x_size * y_size * id_[2] + x_size * y + x;
          s_id[3] = x_size * y_size * id_[3] + x_size * y + x;
        }
        else if (I_Y) {
          s_id[0] = x_size * y_size * z + x_size * id_[0] + x;
          s_id[1] = x_size * y_size * z + x_size * id_[1] + x;
          s_id[2] = x_size * y_size * z + x_size * id_[2] + x;
          s_id[3] = x_size * y_size * z + x_size * id_[3] + x;
        }

        bool case1 = (global_start_ + AncBlkSzX * NAncBlkX < input_gs);
        bool case2 = (input_x >= 3 * unit);
        bool case3 = (input_x + 3 * unit <= AncBlkSzX * NAncBlkX);
        bool case4 = (input_gx + 3 * unit < input_gs);
        bool case5 = (input_gx + unit < input_gs);

        // 预加载 shared memory 数据到寄存器
        T tmp0 = *((T*)s_data + s_id[0]);
        T tmp1 = *((T*)s_data + s_id[1]);
        T tmp2 = *((T*)s_data + s_id[2]);
        T tmp3 = *((T*)s_data + s_id[3]);

        // 初始预测值
        pred = tmp1;

        // 计算不同 case 对应的 pred
        if ((case1 && case2 && case3) || (!case1 && case2 && case3 && case4)) {
          pred = cubic_interpolator(tmp0, tmp1, tmp2, tmp3);
        }
        else if ((case1 && case2 && !case3) || (!case1 && case2 && !(case3 && case4) && case5)) {
          pred = (-tmp0 + 6 * tmp1 + 3 * tmp2) / 8;
        }
        else if ((case1 && !case2 && case3) || (!case1 && !case2 && case3 && case4)) {
          pred = (3 * tmp1 + 6 * tmp2 - tmp3) / 8;
        }
        else if ((case1 && !case2 && !case3) || (!case1 && !case2 && !(case3 && case4) && case5)) {
          pred = (tmp1 + tmp2) / 2;
        }
      }
      auto get_interp_order = [&](auto x, auto gx, auto gs) {
        int b = (x >= 3 * unit) ? 3 : 1;
        int f = ((x + 3 * unit <= AncBlkSzX * NAncBlkX) && ((gx + 3 * unit < gs)))
                    ? 3
                    : (((gx + unit < gs)) ? 1 : 0);

        return (b == 3) ? ((f == 3) ? 4 : ((f == 1) ? 3 : 0))
                        : ((f == 3) ? 2 : ((f == 1) ? 1 : 0));
      };
      if constexpr (FACE) {  //

        bool I_YZ = (x % (2 * unit)) == 0;
        bool I_XZ = (y % (2 * unit)) == 0;
        int x_1, BI_1, GD_1, gx_1, gs_1;
        int x_2, BI_2, GD_2, gx_2, gs_2;
        int s_id_1[4], s_id_2[4];
        auto x_size = AncBlkSzX * NAncBlkX + (SplDim >= 1);
        auto y_size = AncBlkSzY * NAncBlkY + (SplDim >= 2);
        // auto z_size = AncBlkSzZ * NAncBlkZ + (SplDim >= 3);
        if (I_YZ) {
          x_1 = z, BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z, gs_1 = data_size.z;
          x_2 = y, BI_2 = BIY, GD_2 = GDY, gx_2 = global_y, gs_2 = data_size.y;
          s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
          s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
          s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
          s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;
          s_id_2[0] = x_size * y_size * z + x_size * id_y[0] + x;
          s_id_2[1] = x_size * y_size * z + x_size * id_y[1] + x;
          s_id_2[2] = x_size * y_size * z + x_size * id_y[2] + x;
          s_id_2[3] = x_size * y_size * z + x_size * id_y[3] + x;
          pred = s_data[id_z[1]][id_y[1]][x];
        }
        else if (I_XZ) {
          x_1 = z, BI_1 = BIZ, GD_1 = GDZ, gx_1 = global_z, gs_1 = data_size.z;
          x_2 = x, BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
          s_id_1[0] = x_size * y_size * id_z[0] + x_size * y + x;
          s_id_1[1] = x_size * y_size * id_z[1] + x_size * y + x;
          s_id_1[2] = x_size * y_size * id_z[2] + x_size * y + x;
          s_id_1[3] = x_size * y_size * id_z[3] + x_size * y + x;

          s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
          s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
          s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
          s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
          pred = s_data[id_z[1]][y][id_x[1]];
        }
        else {
          x_1 = y, BI_1 = BIY, GD_1 = GDY, gx_1 = global_y, gs_1 = data_size.y;
          x_2 = x, BI_2 = BIX, GD_2 = GDX, gx_2 = global_x, gs_2 = data_size.x;
          s_id_1[0] = x_size * y_size * z + x_size * id_y[0] + x;
          s_id_1[1] = x_size * y_size * z + x_size * id_y[1] + x;
          s_id_1[2] = x_size * y_size * z + x_size * id_y[2] + x;
          s_id_1[3] = x_size * y_size * z + x_size * id_y[3] + x;
          s_id_2[0] = x_size * y_size * z + x_size * y + id_x[0];
          s_id_2[1] = x_size * y_size * z + x_size * y + id_x[1];
          s_id_2[2] = x_size * y_size * z + x_size * y + id_x[2];
          s_id_2[3] = x_size * y_size * z + x_size * y + id_x[3];
          pred = s_data[z][id_y[1]][id_x[1]];
        }

        auto interp_1 = get_interp_order(x_1, gx_1, gs_1);
        auto interp_2 = get_interp_order(x_2, gx_2, gs_2);

        int case_num = interp_1 + interp_2 * 5;

        // clang-format off
        if (interp_1 == 4 && interp_2 == 4) {
          pred  = ( cubic_interpolator( *((T*)s_data + s_id_1[0]), *((T*)s_data + s_id_1[1]), *((T*)s_data + s_id_1[2]), *((T*)s_data + s_id_1[3])) +
                    cubic_interpolator( *((T*)s_data + s_id_2[0]), *((T*)s_data + s_id_2[1]), *((T*)s_data + s_id_2[2]), *((T*)s_data + s_id_2[3]))   ) / 2; }
        else if (interp_1 != 4 && interp_2 == 4) {
          pred  =   cubic_interpolator( *((T*)s_data + s_id_2[0]), *((T*)s_data + s_id_2[1]), *((T*)s_data + s_id_2[2]), *((T*)s_data + s_id_2[3])); }
        else if (interp_1 == 4 && interp_2 != 4) {
          pred  =   cubic_interpolator( *((T*)s_data + s_id_1[0]), *((T*)s_data + s_id_1[1]), *((T*)s_data + s_id_1[2]), *((T*)s_data + s_id_1[3])); }
        else if (interp_1 == 3 && interp_2 == 3) {
          pred  = (-   (*((T*)s_data + s_id_2[0])) + 6 * (*((T*)s_data + s_id_2[1])) + 3 * (*((T*)s_data + s_id_2[2]))) / 8;
          pred += (-   (*((T*)s_data + s_id_1[0])) + 6 * (*((T*)s_data + s_id_1[1])) + 3 * (*((T*)s_data + s_id_1[2]))) / 8;
          pred /= 2; }
        else if (interp_1 == 3 && interp_2 == 2) {
          pred  = (3 * (*((T*)s_data + s_id_2[1])) + 6 * (*((T*)s_data + s_id_2[2])) -     (*((T*)s_data + s_id_2[3]))) / 8;
          pred += (-   (*((T*)s_data + s_id_1[0])) + 6 * (*((T*)s_data + s_id_1[1])) + 3 * (*((T*)s_data + s_id_1[2]))) / 8;
          pred /= 2; }
        else if (interp_1 == 3 && interp_2 < 2) {
          pred  = (-   (*((T*)s_data + s_id_1[0])) + 6 * (*((T*)s_data + s_id_1[1])) + 3 * (*((T*)s_data + s_id_1[2]))) / 8; }
        else if (interp_1 == 2 && interp_2 == 3) {
          pred  = (3 * (*((T*)s_data + s_id_1[1])) + 6 * (*((T*)s_data + s_id_1[2])) -     (*((T*)s_data + s_id_1[3]))) / 8;
          pred += (-   (*((T*)s_data + s_id_2[0])) + 6 * (*((T*)s_data + s_id_2[1])) + 3 * (*((T*)s_data + s_id_2[2]))) / 8;
          pred /= 2; }
        else if (interp_1 == 2 && interp_2 == 2) {
          pred  = (3 * (*((T*)s_data + s_id_1[1])) + 6 * (*((T*)s_data + s_id_1[2])) -     (*((T*)s_data + s_id_1[3]))) / 8;
          pred += (3 * (*((T*)s_data + s_id_2[1])) + 6 * (*((T*)s_data + s_id_2[2])) -     (*((T*)s_data + s_id_2[3]))) / 8;
          pred /= 2; }
        else if (interp_1 == 2 && interp_2 < 2) {
          pred  = (3 * (*((T*)s_data + s_id_1[1])) + 6 * (*((T*)s_data + s_id_1[2])) -     (*((T*)s_data + s_id_1[3]))) / 8; }
        else if (interp_1 <= 1 && interp_2 == 3) {
          pred  = (-   (*((T*)s_data + s_id_2[0])) + 6 * (*((T*)s_data + s_id_2[1])) + 3 * (*((T*)s_data + s_id_2[2]))) / 8; }
        else if (interp_1 <= 1 && interp_2 == 2) {
          pred  = (3 * (*((T*)s_data + s_id_2[1])) + 6 * (*((T*)s_data + s_id_2[2])) -     (*((T*)s_data + s_id_2[3]))) / 8; }
        else if (interp_1 == 1 && interp_2 == 1) {
          pred  = ((*((T*)s_data + s_id_2[1])) + (*((T*)s_data + s_id_2[2]))) / 2;
          pred += ((*((T*)s_data + s_id_1[1])) + (*((T*)s_data + s_id_1[2]))) / 2;
          pred /= 2; }
        else if (interp_1 == 1 && interp_2 < 1) {
          pred  = ((*((T*)s_data + s_id_1[1])) + (*((T*)s_data + s_id_1[2]))) / 2; }
        else if (interp_1 == 0 && interp_2 == 1) {
          pred  = ((*((T*)s_data + s_id_2[1])) + (*((T*)s_data + s_id_2[2]))) / 2; }
        else {
          pred  =  (*((T*)s_data + s_id_1[1])) + (*((T*)s_data + s_id_2[1])) - pred; }
      }
      // clang-format on

      if constexpr (CUBE) {  //
        T tmp_z[4], tmp_y[4], tmp_x[4];
        auto interp_z = get_interp_order(z, global_z, data_size.z);
        auto interp_y = get_interp_order(y, global_y, data_size.y);
        auto interp_x = get_interp_order(x, global_x, data_size.x);

#pragma unroll
        for (int id_itr = 0; id_itr < 4; ++id_itr) { tmp_x[id_itr] = s_data[z][y][id_x[id_itr]]; }
        if (interp_z == 4) {
#pragma unroll
          for (int id_itr = 0; id_itr < 4; ++id_itr) {
            tmp_z[id_itr] = s_data[id_z[id_itr]][y][x];
          }
        }
        if (interp_y == 4) {
#pragma unroll
          for (int id_itr = 0; id_itr < 4; ++id_itr) {
            tmp_y[id_itr] = s_data[z][id_y[id_itr]][x];
          }
        }

        T pred_z[5], pred_y[5], pred_x[5];
        pred_x[0] = tmp_x[1];
        pred_x[1] = cubic_interpolator(tmp_x[0], tmp_x[1], tmp_x[2], tmp_x[3]);
        pred_x[2] = (-tmp_x[0] + 6 * tmp_x[1] + 3 * tmp_x[2]) / 8;
        pred_x[3] = (3 * tmp_x[1] + 6 * tmp_x[2] - tmp_x[3]) / 8;
        pred_x[4] = (tmp_x[1] + tmp_x[2]) / 2;

        pred_y[1] = cubic_interpolator(tmp_y[0], tmp_y[1], tmp_y[2], tmp_y[3]);
        pred_z[1] = cubic_interpolator(tmp_z[0], tmp_z[1], tmp_z[2], tmp_z[3]);

        pred = pred_x[0];

        // clang-format off
        pred = (interp_z == 4 && interp_y == 4 && interp_x == 4) ? (pred_x[1] + pred_y[1] + pred_z[1]) / 3 : pred;
        pred = (interp_z == 4 && interp_y == 4 && interp_x != 4) ? (pred_z[1] + pred_y[1]) / 2             : pred;
        pred = (interp_z == 4 && interp_y != 4 && interp_x == 4) ? (pred_z[1] + pred_x[1]) / 2             : pred;
        pred = (interp_z != 4 && interp_y == 4 && interp_x == 4) ? (pred_y[1] + pred_x[1]) / 2             : pred;
        pred = (interp_z == 4 && interp_y != 4 && interp_x != 4) ?  pred_z[1]                              : pred;
        pred = (interp_z != 4 && interp_y == 4 && interp_x != 4) ?  pred_y[1]                              : pred;
        pred = (interp_z != 4 && interp_y != 4 && interp_x == 4) ?  pred_x[1]                              : pred;
        pred = (interp_z != 4 && interp_y != 4 && interp_x == 3) ?  pred_x[2]                              : pred;
        pred = (interp_z != 4 && interp_y != 4 && interp_x == 2) ?  pred_x[3]                              : pred;
        pred = (interp_z != 4 && interp_y != 4 && interp_x == 1) ?  pred_x[4]                              : pred;
        // pred = (interp_z != 4 && interp_y != 4 && interp_x == 0) ? pred_x[0]: pred;
        // clang-format on
      }

      if constexpr (Workflow == Spl3_AbAtt) {
        auto err = s_data[z][y][x] - pred;
        decltype(err) code;
        // TODO unsafe, did not deal with the out-of-cap case
        {
          code = fabs(err) * eb_r + 1;
          code = err < 0 ? -code : code;
          code = int(code / 2);
        }
        s_data[z][y][x] = pred + code * ebx2;
        atomicAdd(const_cast<T*>(error), code != 0);
      }
      else {
        atomicAdd(const_cast<T*>(error), fabs(s_data[z][y][x] - pred));
      }
    }
  };
  // -------------------------------------------------------------------------------- //

  if constexpr (Coarsen) {
    auto TOTAL = NUM_ELE;
    for (auto _tix = TIX; _tix < TOTAL; _tix += LinBlkSz) {
      auto [x, y, z] = xyzmap(_tix, unit);
      run(x, y, z);
    }
  }
  else {
    if (TIX < NUM_ELE) {
      auto [x, y, z] = xyzmap(TIX, unit);
      run(x, y, z);
    }
  }
  __syncthreads();
}

template <
    typename T, typename FP, int LEVEL, int SplDim, int AncBlkSzX, int AncBlkSzY, int AncBlkSzZ,
    int NAncBlkX, int NAncBlkY, int NAncBlkZ, int LinBlkSz, bool Workflow>
__device__ void psz::spline_layout_interpolate_att(
    T s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
            [AncBlkSzX * NAncBlkX + (SplDim >= 1)],
    dim3 data_size, dim3 g_starts, FP eb_r, FP ebx2, uint8_t level, INTERP_PARAMS intp_param,
    T* error)
{
  auto xblue = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
  auto yblue = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
  auto zblue = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

  auto xblue_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix); };
  auto yblue_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy); };
  auto zblue_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2 + 1); };

  auto xyellow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2); };
  auto yyellow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2 + 1); };
  auto zyellow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

  auto xyellow_reverse = [] __device__(int _tix, int unit) -> int { return unit * (_tix); };
  auto yyellow_reverse = [] __device__(int _tiy, int unit) -> int {
    return unit * (_tiy * 2 + 1);
  };
  auto zyellow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2); };

  auto xhollow = [] __device__(int _tix, int unit) -> int { return unit * (_tix * 2 + 1); };
  auto yhollow = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy); };
  auto zhollow = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz); };

  auto xhollow_reverse = [] __device__(int _tix, int unit) -> int {
    return unit * (_tix * 2 + 1);
  };
  auto yhollow_reverse = [] __device__(int _tiy, int unit) -> int { return unit * (_tiy * 2); };
  auto zhollow_reverse = [] __device__(int _tiz, int unit) -> int { return unit * (_tiz * 2); };

  auto nan_cubic_interp = [] __device__(T a, T b, T c, T d) -> T {
    return (-a + 9 * b + 9 * c - d) / 16;
  };

  auto nat_cubic_interp = [] __device__(T a, T b, T c, T d) -> T {
    return (-3 * a + 23 * b + 23 * c - 3 * d) / 40;
  };
  constexpr auto Coarsen = true;
  // constexpr auto NO_COARSEN       = false;
  constexpr auto BorderIncl = true;
  // constexpr auto BORDER_EXCLUSIVE = false;

  int unit;

  FP cur_ebx2 = ebx2, cur_eb_r = eb_r;

  auto calc_eb = [&](auto unit) {
    cur_ebx2 = ebx2, cur_eb_r = eb_r;
    int temp = 1;
    while (temp < unit) {
      temp *= 2;
      cur_eb_r *= intp_param.alpha;
      cur_ebx2 /= intp_param.alpha;
    }
    if (cur_ebx2 < ebx2 / intp_param.beta) {
      cur_ebx2 = ebx2 / intp_param.beta;
      cur_eb_r = eb_r * intp_param.beta;
    }
  };

  if constexpr (Workflow == Spl3_AbAtt) {
    int max_unit = ((AncBlkSzX >= AncBlkSzY) ? AncBlkSzX : AncBlkSzY);
    max_unit = ((max_unit >= AncBlkSzZ) ? max_unit : AncBlkSzZ);
    max_unit /= 2;
    int unit_x = AncBlkSzX, unit_y = AncBlkSzY, unit_z = AncBlkSzZ;
#pragma unroll
    for (int unit = max_unit; unit >= 1; unit /= 2) {
      calc_eb(unit);
      unit_x = (SplDim >= 1) ? unit * 2 : 1;
      unit_y = (SplDim >= 2) ? unit * 2 : 1;
      unit_z = (SplDim >= 3) ? unit * 2 : 1;
      if (intp_param.use_md[level]) {
        int N_x = AncBlkSzX / (unit * 2);
        int N_y = AncBlkSzY / (unit * 2);
        int N_z = AncBlkSzZ / (unit * 2);
        int N_line = N_x * (N_y + 1) * (N_z + 1) + (N_x + 1) * N_y * (N_z + 1) +
                     (N_x + 1) * (N_y + 1) * N_z;
        int N_face = N_x * N_y * (N_z + 1) + N_x * (N_y + 1) * N_z + (N_x + 1) * N_y * N_z;
        int N_cube = N_x * N_y * N_z;
        if (intp_param.use_natural[level] == 0) {
          if constexpr (SplDim >= 1)
            interpolate_stage_md_att<
                T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_line<SplDim, AncBlkSzX>), true, false, false, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, data_size, g_starts, xyzmap_line<SplDim, AncBlkSzX>, unit, cur_eb_r,
                cur_ebx2, nan_cubic_interp, error, N_line);

          if constexpr (SplDim >= 2)
            interpolate_stage_md_att<
                T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_face<SplDim, AncBlkSzX>), false, true, false, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, data_size, g_starts, xyzmap_face<SplDim, AncBlkSzX>, unit, cur_eb_r,
                cur_ebx2, nan_cubic_interp, error, N_face);

          if constexpr (SplDim >= 3)
            interpolate_stage_md_att<
                T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_cube<SplDim, AncBlkSzX>), false, false, true, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, data_size, g_starts, xyzmap_cube<SplDim, AncBlkSzX>, unit, cur_eb_r,
                cur_ebx2, nan_cubic_interp, error, N_cube);
        }
        else {
          if constexpr (SplDim >= 1)
            interpolate_stage_md_att<
                T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_line<SplDim, AncBlkSzX>), true, false, false, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, data_size, g_starts, xyzmap_line<SplDim, AncBlkSzX>, unit, cur_eb_r,
                cur_ebx2, nat_cubic_interp, error, N_line);

          if constexpr (SplDim >= 2)
            interpolate_stage_md_att<
                T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_face<SplDim, AncBlkSzX>), false, true, false, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, data_size, g_starts, xyzmap_face<SplDim, AncBlkSzX>, unit, cur_eb_r,
                cur_ebx2, nat_cubic_interp, error, N_face);

          if constexpr (SplDim >= 3)
            interpolate_stage_md_att<
                T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyzmap_cube<SplDim, AncBlkSzX>), false, false, true, LinBlkSz, Coarsen,
                BorderIncl, Workflow>(
                s_data, data_size, g_starts, xyzmap_cube<SplDim, AncBlkSzX>, unit, cur_eb_r,
                cur_ebx2, nat_cubic_interp, error, N_cube);
        }
      }
      else {
        if (intp_param.reverse[level]) {
          if constexpr (SplDim >= 1) {
            interpolate_stage_att<
                T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),
                false, false, true, Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, data_size, g_starts, xhollow_reverse, yhollow_reverse, zhollow_reverse,
                unit, cur_eb_r, cur_ebx2, intp_param.use_natural[level], error,
                NAncBlkX * AncBlkSzX / unit_x, NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2),
                NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
            unit_x /= 2;
          }
          if constexpr (SplDim >= 2) {
            interpolate_stage_att<
                T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),
                false, true, false, Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, data_size, g_starts, xyellow_reverse, yyellow_reverse, zyellow_reverse,
                unit, cur_eb_r, cur_ebx2, intp_param.use_natural[level], error,
                NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1), NAncBlkY * AncBlkSzY / unit_y,
                NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
            unit_y /= 2;
          }
          if constexpr (SplDim >= 3) {
            interpolate_stage_att<
                T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse), true,
                false, false, Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, data_size, g_starts, xblue_reverse, yblue_reverse, zblue_reverse, unit,
                cur_eb_r, cur_ebx2, intp_param.use_natural[level], error,
                NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1),
                NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2), NAncBlkZ * AncBlkSzZ / unit_z);
            unit_z /= 2;
          }
        }
        else {
          if constexpr (SplDim >= 3) {
            interpolate_stage_att<
                T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xblue), decltype(yblue), decltype(zblue), true, false, false, Coarsen,
                LinBlkSz, BorderIncl, Workflow>(
                s_data, data_size, g_starts, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2,
                intp_param.use_natural[level], error,
                NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1),
                NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2), NAncBlkZ * AncBlkSzZ / unit_z);
            unit_z /= 2;
          }
          if constexpr (SplDim >= 2) {
            interpolate_stage_att<
                T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xyellow), decltype(yyellow), decltype(zyellow), false, true, false,
                Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, data_size, g_starts, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2,
                intp_param.use_natural[level], error,
                NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1), NAncBlkY * AncBlkSzY / unit_y,
                NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
            unit_y /= 2;
          }
          if constexpr (SplDim >= 1) {
            interpolate_stage_att<
                T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
                decltype(xhollow), decltype(yhollow), decltype(zhollow), false, false, true,
                Coarsen, LinBlkSz, BorderIncl, Workflow>(
                s_data, data_size, g_starts, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2,
                intp_param.use_natural[level], error, NAncBlkX * AncBlkSzX / unit_x,
                NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2),
                NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
            unit_x /= 2;
          }
        }
      }
    }
  }

  if constexpr (Workflow != Spl3_AbAtt) {
    unit = 1 << level;
    int unit_x = (SplDim >= 1) ? unit * 2 : 1;
    int unit_y = (SplDim >= 2) ? unit * 2 : 1;
    int unit_z = (SplDim >= 3) ? unit * 2 : 1;
    if (intp_param.use_md[level]) {
      int N_x = AncBlkSzX / (unit * 2);
      int N_y = AncBlkSzY / (unit * 2);
      int N_z = AncBlkSzZ / (unit * 2);
      int N_line =
          N_x * (N_y + 1) * (N_z + 1) + (N_x + 1) * N_y * (N_z + 1) + (N_x + 1) * (N_y + 1) * N_z;
      int N_face = N_x * N_y * (N_z + 1) + N_x * (N_y + 1) * N_z + (N_x + 1) * N_y * N_z;
      int N_cube = N_x * N_y * N_z;

      // auto cubic_interp = (intp_param.use_natural[level] == 0) ? nan_cubic_interp :
      // nat_cubic_interp;

      if (intp_param.use_natural[level] == 0) {
        if constexpr (SplDim >= 1)
          interpolate_stage_md_att<
              T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
              decltype(xyzmap_line<SplDim, AncBlkSzX>), true, false, false, LinBlkSz, Coarsen,
              BorderIncl, Workflow>(
              s_data, data_size, g_starts, xyzmap_line<SplDim, AncBlkSzX>, unit, cur_eb_r,
              cur_ebx2, nan_cubic_interp, error, N_line);

        if constexpr (SplDim >= 2)
          interpolate_stage_md_att<
              T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
              decltype(xyzmap_face<SplDim, AncBlkSzX>), false, true, false, LinBlkSz, Coarsen,
              BorderIncl, Workflow>(
              s_data, data_size, g_starts, xyzmap_face<SplDim, AncBlkSzX>, unit, cur_eb_r,
              cur_ebx2, nan_cubic_interp, error, N_face);

        if constexpr (SplDim >= 3)
          interpolate_stage_md_att<
              T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
              decltype(xyzmap_cube<SplDim, AncBlkSzX>), false, false, true, LinBlkSz, Coarsen,
              BorderIncl, Workflow>(
              s_data, data_size, g_starts, xyzmap_cube<SplDim, AncBlkSzX>, unit, cur_eb_r,
              cur_ebx2, nan_cubic_interp, error, N_cube);
      }
      else {
        if constexpr (SplDim >= 1)
          interpolate_stage_md_att<
              T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
              decltype(xyzmap_line<SplDim, AncBlkSzX>), true, false, false, LinBlkSz, Coarsen,
              BorderIncl, Workflow>(
              s_data, data_size, g_starts, xyzmap_line<SplDim, AncBlkSzX>, unit, cur_eb_r,
              cur_ebx2, nat_cubic_interp, error, N_line);

        if constexpr (SplDim >= 2)
          interpolate_stage_md_att<
              T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
              decltype(xyzmap_face<SplDim, AncBlkSzX>), false, true, false, LinBlkSz, Coarsen,
              BorderIncl, Workflow>(
              s_data, data_size, g_starts, xyzmap_face<SplDim, AncBlkSzX>, unit, cur_eb_r,
              cur_ebx2, nat_cubic_interp, error, N_face);

        if constexpr (SplDim >= 3)
          interpolate_stage_md_att<
              T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
              decltype(xyzmap_cube<SplDim, AncBlkSzX>), false, false, true, LinBlkSz, Coarsen,
              BorderIncl, Workflow>(
              s_data, data_size, g_starts, xyzmap_cube<SplDim, AncBlkSzX>, unit, cur_eb_r,
              cur_ebx2, nat_cubic_interp, error, N_cube);
      }
    }
    else {
      if (intp_param.reverse[level]) {
        if constexpr (SplDim >= 1) {
          interpolate_stage_att<
              T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
              decltype(xhollow_reverse), decltype(yhollow_reverse), decltype(zhollow_reverse),
              false, false, true, Coarsen, LinBlkSz, BorderIncl, Workflow>(
              s_data, data_size, g_starts, xhollow_reverse, yhollow_reverse, zhollow_reverse, unit,
              cur_eb_r, cur_ebx2, intp_param.use_natural[level], error,
              NAncBlkX * AncBlkSzX / unit_x, NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2),
              NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
          unit_x /= 2;
        }
        if constexpr (SplDim >= 2) {
          interpolate_stage_att<
              T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
              decltype(xyellow_reverse), decltype(yyellow_reverse), decltype(zyellow_reverse),
              false, true, false, Coarsen, LinBlkSz, BorderIncl, Workflow>(
              s_data, data_size, g_starts, xyellow_reverse, yyellow_reverse, zyellow_reverse, unit,
              cur_eb_r, cur_ebx2, intp_param.use_natural[level], error,
              NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1), NAncBlkY * AncBlkSzY / unit_y,
              NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
          unit_y /= 2;
        }
        if constexpr (SplDim >= 3) {
          interpolate_stage_att<
              T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
              decltype(xblue_reverse), decltype(yblue_reverse), decltype(zblue_reverse), true,
              false, false, Coarsen, LinBlkSz, BorderIncl, Workflow>(
              s_data, data_size, g_starts, xblue_reverse, yblue_reverse, zblue_reverse, unit,
              cur_eb_r, cur_ebx2, intp_param.use_natural[level], error,
              NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1),
              NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2), NAncBlkZ * AncBlkSzZ / unit_z);
          unit_z /= 2;
        }
      }
      else {
        if constexpr (SplDim >= 3) {
          interpolate_stage_att<
              T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
              decltype(xblue), decltype(yblue), decltype(zblue), true, false, false, Coarsen,
              LinBlkSz, BorderIncl, Workflow>(
              s_data, data_size, g_starts, xblue, yblue, zblue, unit, cur_eb_r, cur_ebx2,
              intp_param.use_natural[level], error, NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1),
              NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2), NAncBlkZ * AncBlkSzZ / unit_z);
          unit_z /= 2;
        }
        if constexpr (SplDim >= 2) {
          interpolate_stage_att<
              T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
              decltype(xyellow), decltype(yyellow), decltype(zyellow), false, true, false, Coarsen,
              LinBlkSz, BorderIncl, Workflow>(
              s_data, data_size, g_starts, xyellow, yyellow, zyellow, unit, cur_eb_r, cur_ebx2,
              intp_param.use_natural[level], error, NAncBlkX * AncBlkSzX / unit_x + (SplDim >= 1),
              NAncBlkY * AncBlkSzY / unit_y, NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
          unit_y /= 2;
        }
        if constexpr (SplDim >= 1) {
          interpolate_stage_att<
              T, FP, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ,
              decltype(xhollow), decltype(yhollow), decltype(zhollow), false, false, true, Coarsen,
              LinBlkSz, BorderIncl, Workflow>(
              s_data, data_size, g_starts, xhollow, yhollow, zhollow, unit, cur_eb_r, cur_ebx2,
              intp_param.use_natural[level], error, NAncBlkX * AncBlkSzX / unit_x,
              NAncBlkY * AncBlkSzY / unit_y + (SplDim >= 2),
              NAncBlkZ * AncBlkSzZ / unit_z + (SplDim >= 3));
          unit_x /= 2;
        }
      }
    }
  }
}

#define SPLATT(Mode)                                                                       \
  psz::spline_layout_interpolate_att<                                                      \
      T, FP, LEVEL, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ, \
      LinBlkSz, Mode>
#define SPLATT_PredAtt SPLATT(Spl3_PredAtt)
#define SPLATT_AbAtt SPLATT(Spl3_AbAtt)

template <
    typename T, typename FP, int LEVEL, int SplDim, int AncBlkSzX, int AncBlkSzY, int AncBlkSzZ,
    int NAncBlkX, int NAncBlkY, int NAncBlkZ, int LinBlkSz>
__global__ void psz::KCU_pa_spl_infprecis_data(
    T* data, dim3 data_size, dim3 data_leap, dim3 sample_starts, dim3 sample_block_grid_sizes,
    dim3 sample_strides, FP eb_r, FP eb_x2, INTERP_PARAMS intp_param, T* errors, bool workflow)
{
  {
    // if constexpr (SplDim == 3)
    // __shared__ struct {
    //     T data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)]
    //     [AncBlkSzY * NAncBlkY + (SplDim >= 2)]
    //     [AncBlkSzX * NAncBlkX + (SplDim >= 1)];
    //     T err[6];
    // } shmem;

    __shared__ T s_data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)][AncBlkSzY * NAncBlkY + (SplDim >= 2)]
                       [AncBlkSzX * NAncBlkX + (SplDim >= 1)];
    __shared__ T s_err[9];

    // if constexpr (SplDim == 2)
    // __shared__ struct {
    //     T data[AncBlkSzZ * NAncBlkZ + (SplDim >= 3)]
    //     [AncBlkSzY * NAncBlkY + (SplDim >= 2)]
    //     [AncBlkSzX * NAncBlkX + (SplDim >= 1)];
    //     T err[1];
    // } shmem;

    dim3 g_starts;
    uint8_t level = 0;
    uint8_t unit = 1;
    pre_compute_att<T, SplDim, LEVEL>(
        sample_starts, sample_block_grid_sizes, sample_strides, g_starts, intp_param, level, unit,
        s_err, workflow);

    global2shmem_data_att<
        T, T, SplDim, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY, NAncBlkZ, LinBlkSz>(
        data, data_size, data_leap, s_data, g_starts, unit);

    if constexpr (SplDim == 3) {
      if (workflow) {
        if (level == 2) {
          uint8_t level3 = 3;
          intp_param.use_natural[3] = false;
          intp_param.use_natural[2] = false;
          intp_param.use_md[3] = false;
          intp_param.reverse[3] = false;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level3, intp_param, s_err);
          intp_param.reverse[3] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level3, intp_param, s_err + 1);
          intp_param.use_md[3] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level3, intp_param, s_err + 2);

          intp_param.use_md[2] = false;
          intp_param.reverse[2] = false;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err + 3);
          intp_param.reverse[2] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err + 4);
          intp_param.use_md[2] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err + 5);
          if (TIX < 6) { atomicAdd(const_cast<T*>(errors + TIX), s_err[TIX]); }
        }
        else if (level == 1) {
          intp_param.use_md[1] = false;
          intp_param.reverse[1] = false;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err);
          intp_param.reverse[1] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err + 1);
          intp_param.use_md[1] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err + 2);

          if (TIX < 3) { atomicAdd(const_cast<T*>(errors + 3 + BIY * 3 + TIX), s_err[TIX]); }
        }
        else {
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err);
          if (TIX == 0) { atomicAdd(const_cast<T*>(errors + 9 + BIY), s_err[0]); }
        }
      }
      else {
        SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err);
        if (TIX == 0) atomicAdd(const_cast<T*>(errors + BIY), s_err[0]);
      }
    }
    if constexpr (SplDim == 2) {
      if (workflow) {
        if (level == 3) {
          uint8_t level5 = 5;
          intp_param.use_natural[5] = false;
          intp_param.use_natural[4] = false;
          intp_param.use_natural[3] = false;
          intp_param.use_md[5] = false;
          intp_param.reverse[5] = false;

          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level5, intp_param, s_err);
          intp_param.reverse[5] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level5, intp_param, s_err + 1);
          intp_param.use_md[5] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level5, intp_param, s_err + 2);

          uint8_t level4 = 4;
          intp_param.use_md[4] = false;
          intp_param.reverse[4] = false;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level4, intp_param, s_err + 3);
          intp_param.reverse[4] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level4, intp_param, s_err + 4);
          intp_param.use_md[4] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level4, intp_param, s_err + 5);

          intp_param.use_md[3] = false;
          intp_param.reverse[3] = false;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err + 6);
          intp_param.reverse[3] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err + 7);
          intp_param.use_md[3] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err + 8);
          if (TIX < 9) { atomicAdd(const_cast<T*>(errors + TIX), s_err[TIX]); }
        }
        else if (level == 2) {
          intp_param.use_md[2] = false;
          intp_param.reverse[2] = false;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err);
          intp_param.reverse[2] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err + 1);
          intp_param.use_md[2] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err + 2);

          if (TIX < 3) { atomicAdd(const_cast<T*>(errors + 6 + BIY * 3 + TIX), s_err[TIX]); }
        }
        else if (level == 1) {
          intp_param.use_md[1] = false;
          intp_param.reverse[1] = false;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err);
          intp_param.reverse[1] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err + 1);
          intp_param.use_md[1] = true;
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err + 2);

          if (TIX < 3) { atomicAdd(const_cast<T*>(errors + 6 + BIY * 3 + TIX), s_err[TIX]); }
        }
        else {
          SPLATT_PredAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err);
          if (TIX == 0) { atomicAdd(const_cast<T*>(errors + 15 + BIY), s_err[0]); }
        }
      }
      else {
        SPLATT_AbAtt(s_data, data_size, g_starts, eb_r, eb_x2, level, intp_param, s_err);
        if (TIX == 0) atomicAdd(const_cast<T*>(errors + BIY), s_err[0]);
      }
    }
  }
}

#undef TIX
#undef TIY
#undef TIZ
#undef BIX
#undef BIY
#undef BIZ
#undef BDX
#undef BDY
#undef BDZ
#undef GDX
#undef GDY
#undef GDZ

#endif
